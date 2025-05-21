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
from torchmetrics.text import BLEUScore
from bert_score import BERTScorer
from torchmetrics.text import WordErrorRate

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
    scaler = GradScaler(device.output_device)
    running_loss = 0.0
    num_samples = 0
    lr0 = optimizer.optimizer.param_groups[0]["lr"]
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    pbar = tqdm(loader, position=1, leave=False, desc=f"Epoch {epoch_idx}")
    for batch_idx, data in enumerate(
        pbar
    ):

        vid, vid_len = [device.data_to_device(x) for x in data[:2]]
        txt = data[4]

        optimizer.zero_grad()
        with autocast(device.output_device):
            ret = model(vid, vid_len, txt)
            (loss_t5, contr_loss) = ret["loss"]
            loss = loss_t5 + contr_loss * 0.5
            recorder.log(f"\t\t {ret['pred_text']} --||-- {txt}")

        if not torch.isfinite(loss):
            continue

        scaler.scale(loss).backward()
        scaler.step(optimizer.optimizer)
        scaler.update()
        pbar.set_description(f"Epoch {epoch_idx}; Loss: {loss} {loss_t5} {contr_loss}")

        bs = vid.size(0)
        running_loss += loss.item() * bs
        num_samples += bs

        del vid, vid_len, ret, loss
        torch.cuda.empty_cache()

    mean_loss = running_loss / max(1, num_samples)
    recorder.log(f"Epoch {epoch_idx}: Loss {mean_loss:.6f}, LR {lr0:.6f}")
    # print(f"Epoch {epoch_idx}: Loss {mean_loss:.6f}, LR {lr0:.6f}")
    return mean_loss


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
    bleu1 = BLEUScore(n_gram=1)
    bleu2 = BLEUScore(n_gram=2)
    bleu3 = BLEUScore(n_gram=3)
    bleu4 = BLEUScore(n_gram=4)
    wer = WordErrorRate()

    gloss_path = f"./preprocess/{cfg.dataset}/gloss_dict.npy"
    id2tok = {}
    if os.path.exists(gloss_path):
        id2tok = {
            v[0]: k for k, v in np.load(gloss_path, allow_pickle=True).item().items()
        }

    for data in tqdm(loader, desc=f"Valid Epoch {epoch}", leave=False):

        vid, vid_len = [device.data_to_device(x) for x in data[:2]]
        txt = data[4]

        with torch.no_grad(), autocast(device.output_device):
            ret = model(vid, vid_len, text=txt)
            pred_sent = ret["pred_text"]
            ref_sent = txt

        bleu1.update(pred_sent, ref_sent)
        bleu2.update(pred_sent, ref_sent)
        bleu3.update(pred_sent, ref_sent)
        bleu4.update(pred_sent, ref_sent)
        # wer_score = wer(preds=[pred_sent], target=[ref_sent])
        recorder.log(f"predicted: {pred_sent}")
        recorder.log(f"ground truth: {ref_sent}")

        del vid, vid_len, ret
        torch.cuda.empty_cache()

    b1, b2, b3, b4 = (m.compute().item() for m in (bleu1, bleu2, bleu3, bleu4))
    recorder.log(f"BLEU-1 {b1:.2f} BLEU-2 {b2:.2f} BLEU-3 {b3:.2f} BLEU-4 {b4:.2f}")
    # print(       f"BLEU-1 {b1:.2f} BLEU-2 {b2:.2f} BLEU-3 {b3:.2f} BLEU-4 {b4:.2f}")

    return b4


def seq_feature_generation(loader, model, device, mode, work_dir, recorder):
    model.eval()

    src_path = os.path.abspath(f"{work_dir}{mode}")
    tgt_path = os.path.abspath(f"./features/{mode}")
    os.makedirs("./features/", exist_ok=True)

    if os.path.islink(tgt_path):
        if work_dir[1:] in os.readlink(tgt_path) and os.path.isabs(
            os.readlink(tgt_path)
        ):
            return
        os.unlink(tgt_path)

    if os.path.exists(src_path) and len(loader.dataset) == len(os.listdir(src_path)):
        os.symlink(src_path, tgt_path)
        return

    os.makedirs(src_path, exist_ok=True)

    for batch_idx, data in tqdm(enumerate(loader), total=len(loader), leave=False):
        vid, vid_len = device.data_to_device(data[0]), device.data_to_device(data[1])
        labels, label_lens, file_infos = data[2], data[3], data[4]

        with torch.no_grad():
            ret = model(vid, vid_len)

        start = 0
        for idx, file_name in enumerate(file_infos):
            end = start + label_lens[idx]
            file_base = file_name.split("|")[0]
            save_path = os.path.join(src_path, f"{file_base}_features.npy")

            np.save(
                save_path,
                {
                    "label": labels[start:end],
                    "features": ret["framewise_features"][idx][:, : vid_len[idx]]
                    .T.cpu()
                    .numpy(),
                },
            )

            start = end
        assert start == len(labels)

    os.symlink(src_path, tgt_path)


def write2file(path, info, output):
    with open(path, "w") as file:
        for sample_idx, words in enumerate(output):
            for word_idx, word in enumerate(words):
                label = "[EMPTY]" if not word else word[0]
                file.write(
                    f"{info[sample_idx]} 1 {word_idx * 0.01:.2f} {(word_idx + 1) * 0.01:.2f} {label}\n"
                )
