import os
from os.path import join
import json
import pdb
import sys
import copy
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from slr_network import SLRModel, ModelOutput
import time
import traceback
from torchmetrics.text import BLEUScore, BERTScore

from torch.amp import autocast, GradScaler

from utils.device import GpuDataParallel
import utils
from utils.parameters import ConfigArgs

import os
import torch


def seq_train(
    loader: torch.utils.data.DataLoader,
    model: SLRModel,
    optimizer: utils.Optimizer,
    device: GpuDataParallel,
    epoch_idx: int,
    recorder: utils.Recorder,
):
    model.train()
    losses = []
    scaler = GradScaler(device.output_device)
    learning_rates = [group['lr'] for group in optimizer.optimizer.param_groups]

    for batch_idx, data in enumerate(tqdm(loader, desc=f"Training Epoch {epoch_idx}")):
        try:
            torch.cuda.empty_cache()
            vid, vid_len, label, label_len = [device.data_to_device(x) for x in data[:4]]
            print(label)

            optimizer.zero_grad()
            with autocast(device.output_device):
                ret = model(vid, vid_len, label=label, label_lgt=label_len)
                loss = model.criterion_calculation(ret, label, label_len)

            if not torch.isfinite(loss):
                print(f"Skipping batch {batch_idx} due to invalid loss.")
                continue

            scaler.scale(loss).backward()
            scaler.step(optimizer.optimizer)
            scaler.update()

            losses.append(loss.item())

        except Exception as e:
            print(f"Exception at batch {batch_idx}: {e}")
            print(traceback.format_exc())
            raise

    recorder.log(f'\tEpoch {epoch_idx}: Loss: {np.mean(losses):.6f}, LR: {learning_rates[0]:.6f}')
    print(f'Epoch {epoch_idx}: Loss: {np.mean(losses):.6f}, LR: {learning_rates[0]:.6f}')
    optimizer.scheduler.step()
    return


def indices_to_tokens(idx_tensor, idx2tok):
    """
    idx_tensor : 1-D or 2-D LongTensor on GPU/CPU
    idx2tok    : dict[int -> str]   (e.g. {0:'<BLK>', 1:'HELLO', 2:'WORLD', …})
    returns    : list[str]          (tokens without blanks/pads)
    """
    return [idx2tok[i] for i in idx_tensor if i in idx2tok and idx2tok[i] != "<UNK>"]

def seq_eval(
    cfg: ConfigArgs,
    loader: torch.utils.data.DataLoader,
    model: SLRModel,
    device: GpuDataParallel,
    mode: str,
    epoch: int,
    work_dir: str,
    recorder: utils.Recorder,
    evaluate_tool="python",
):
    model.eval()
    num_batches = len(loader)
    # ── metric objects (streaming) ────────────────────────────────────────
    bleu1 = BLEUScore(n_gram=1)
    bleu2 = BLEUScore(n_gram=2)
    bleu3 = BLEUScore(n_gram=3)
    bleu4 = BLEUScore(n_gram=4)
    bert = BERTScore(lang="en")

    # ── id-to-token map ───────────────────────────────────────────────────
    gloss_file = f"./preprocess/{cfg.dataset}/gloss_dict.npy"
    id2token = (
        {v[0]: k for k, v in np.load(gloss_file, allow_pickle=True).item().items()}
        if os.path.exists(gloss_file) else {}
    )

    # ── validation loop ───────────────────────────────────────────────────
    for _, data in enumerate(tqdm(loader, desc=f"Validating Epoch {epoch}")):
        torch.cuda.empty_cache()                  # frees *unused* GPU blocks

        vid, vid_len, label, label_len = [device.data_to_device(x) for x in data[:4]]

        with torch.no_grad(), autocast(device.output_device):   # mixed precision ⇒ <½ VRAM
            ret   = model(vid, vid_len, label=label, label_lgt=label_len)
            preds = ret["recognized_sents"]                # list[list[(tok, score)]]

        # ---> Immediately move small things to CPU & free the big ones <---
        label = label.cpu()
        preds_cpu = [p for batch in preds for p, _ in batch]
        del vid, ret, preds
        torch.cuda.empty_cache()

        # convert references once, then drop label tensor
        refs_cpu = indices_to_tokens(label.tolist(), id2token)
        del label

        # ── update streaming metrics ─────────────────────────────────────
        bleu1.update(preds_cpu, refs_cpu)
        bleu2.update(preds_cpu, refs_cpu)
        bleu3.update(preds_cpu, refs_cpu)
        bleu4.update(preds_cpu, refs_cpu)
        bert.update(preds_cpu, refs_cpu)     # <── just one line
    # ── final numbers ─────────────────────────────────────────────────────
    bleu_1, bleu_2, bleu_3, bleu_4 = [m.compute().item() for m in (bleu1, bleu2, bleu3, bleu4)]
    bert_P, bert_R, bert_F          = [x.item() for x in bert.compute()]


    print(f"BLEU-1 {bleu_1:.2f}  BLEU-2 {bleu_2:.2f}  BLEU-3 {bleu_3:.2f}  BLEU-4 {bleu_4:.2f}")
    print(f"BERTScore P/R/F : {bert_P:.4f} / {bert_R:.4f} / {bert_F:.4f}")

    # recorder.log(f"Epoch {epoch}, {mode} WER: {score:.2f}%", os.path.join(work_dir, f"{mode}.txt"))
    # print(f"Epoch {epoch}, {mode} WER: {score:.2f}%", os.path.join(work_dir, f"{mode}.txt"))
    return bert_P

def seq_feature_generation(loader, model, device, mode, work_dir, recorder):
    model.eval()

    src_path = os.path.abspath(f"{work_dir}{mode}")
    tgt_path = os.path.abspath(f"./features/{mode}")
    os.makedirs("./features/", exist_ok=True)

    if os.path.islink(tgt_path):
        if work_dir[1:] in os.readlink(tgt_path) and os.path.isabs(os.readlink(tgt_path)):
            return
        os.unlink(tgt_path)

    if os.path.exists(src_path) and len(loader.dataset) == len(os.listdir(src_path)):
        os.symlink(src_path, tgt_path)
        return

    os.makedirs(src_path, exist_ok=True)

    for batch_idx, data in tqdm(enumerate(loader), total=len(loader)):
        vid, vid_len = device.data_to_device(data[0]), device.data_to_device(data[1])
        labels, label_lens, file_infos = data[2], data[3], data[4]

        with torch.no_grad():
            ret = model(vid, vid_len)

        start = 0
        for idx, file_name in enumerate(file_infos):
            end = start + label_lens[idx]
            file_base = file_name.split('|')[0]
            save_path = os.path.join(src_path, f"{file_base}_features.npy")

            np.save(save_path, {
                "label": labels[start:end],
                "features": ret['framewise_features'][idx][:, :vid_len[idx]].T.cpu().numpy(),
            })

            start = end
        assert start == len(labels)

    os.symlink(src_path, tgt_path)

def write2file(path, info, output):
    with open(path, "w") as file:
        for sample_idx, words in enumerate(output):
            for word_idx, word in enumerate(words):
                label = "[EMPTY]" if not word else word[0]
                file.write(f"{info[sample_idx]} 1 {word_idx * 0.01:.2f} {(word_idx + 1) * 0.01:.2f} {label}\n")
