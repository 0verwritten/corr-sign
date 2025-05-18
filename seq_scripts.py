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
from evaluation.slr_eval.wer_calculation import evaluate
from slr_network import SLRModel, ModelOutput
import time
import traceback

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
    total_info, total_sents, total_conv_sents = [None] * num_batches, [None] * num_batches, [None] * num_batches

    for batch_idx, data in enumerate(tqdm(loader, desc=f"Validating Epoch {epoch}")):
        file_info = data[4]
        torch.cuda.empty_cache()
        vid, vid_len, label, label_len = [device.data_to_device(x) for x in data[:4]]
        with torch.no_grad():
            ret = model(vid, vid_len, label=label, label_lgt=label_len)

        total_info[batch_idx] = [f.split("|")[0] for f in file_info]
        total_sents[batch_idx] = ret['recognized_sents']
        total_conv_sents[batch_idx] = ret['conv_sents']

    if evaluate_tool != "python":
        raise NotImplementedError("Only Python evaluation is supported.")

    try:
        write2file(os.path.join(work_dir, f"output-hypothesis-{mode}.ctm"), total_info, total_sents)
        write2file(os.path.join(work_dir, f"output-hypothesis-{mode}-conv.ctm"), total_info, total_conv_sents)

        score, metrics = evaluate(
            prefix=work_dir,
            mode=mode,
            output_file=f"output-hypothesis-{mode}.ctm",
            evaluate_dir=cfg.dataset_info['evaluation_dir'],
            evaluate_prefix=cfg.dataset_info['evaluation_prefix'],
            output_dir=f"epoch_{epoch}_result/",
            python_evaluate=True,
            triplet=True,
        )

        metrics_path = os.path.join(work_dir, f"output-hypothesis-{mode}-epoch-{epoch}-metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f)

    except Exception:
        print("Evaluation failed.")
        print(traceback.format_exc())
        score = 100.0

    recorder.log(f"Epoch {epoch}, {mode} WER: {score:.2f}%", os.path.join(work_dir, f"{mode}.txt"))
    print(f"Epoch {epoch}, {mode} WER: {score:.2f}%", os.path.join(work_dir, f"{mode}.txt"))
    return score

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
