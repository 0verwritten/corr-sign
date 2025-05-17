# %%

from decord import VideoReader
from decord import gpu
import cv2
import os
import numpy as np
import subprocess
import random

import re
import os
import cv2
import pandas
import argparse
import numpy as np
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool
import re
import nltk

nltk.download("punkt")
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from collections import Counter

stemmer = SnowballStemmer("english")


def csv2dict(anno_path, dataset_type, dataset_root, unknownTokens: set = set()):
    inputs_df = pandas.read_csv(anno_path, sep="\t")
    info_dict = dict()
    info_dict["prefix"] = os.path.join(
        dataset_root, f"{dataset_type}_rgb_front_clips/raw_videos"
    )
    print(f"Generate information dict from {anno_path}")\
    
    # random_images = random.sample(list(inputs_df.iterrows()), 10)
    random_images = list(inputs_df.iterrows())
    # for idx, row in tqdm(inputs_df.iterrows(), total=len(inputs_df)):
    for index, (idx, row) in tqdm(enumerate(random_images), total=len(inputs_df)):
        video_id = str(row["VIDEO_ID"])
        sentence_name = str(row["SENTENCE_NAME"])  # corresponds to video file
        label = str(row["SENTENCE"]).strip()  # full sentence
        signer = "unknown"  # no signer info available

        video_path = os.path.join(info_dict["prefix"], f"{sentence_name}.mp4")
        split_label = str(label).strip()
        tokens = word_tokenize(split_label)

        info_dict[index] = {
            "fileid": sentence_name,
            "folder": f"{dataset_type}_rgb_front_clips/raw_videos/{sentence_name}",
            "signer": signer,
            "label": label,
            "tokens": [
                "<unk>" if x in unknownTokens else x
                for x in [stemmer.stem(gloss) for gloss in tokens]
            ],
            "video_path": video_path,
            "original_info": f"{video_id}|{sentence_name}|{signer}|{label}",
        }
    return info_dict


def generate_gt_stm(info, save_path):
    with open(save_path, "w") as f:
        for k, v in info.items():
            if not isinstance(k, int):
                continue
            f.writelines(
                f"{v['fileid']} 1 {v['signer']} 0.0 1.79769e+308 {v['label']}\n"
            )


def sign_dict_update(total_dict, info):
    for k, v in info.items():
        if not isinstance(k, int):
            continue
        tokens = v["tokens"]

        for gloss in tokens:
            if gloss not in total_dict:
                total_dict[gloss] = 1
            else:
                total_dict[gloss] += 1
    return total_dict


def resize_dataset(video_idx, dsize, info_dict):
    """
    Uses FFmpeg to extract frames from a video with center crop and resize.
    Saves each frame as a JPEG in the output directory.
    """
    info = info_dict[video_idx]
    output_dir = os.path.join(info_dict["prefix"], info["fileid"])
    video_path = f"{output_dir}.mp4"

    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    width, height = tuple(int(x) for x in  dsize.split('x'))  # Tuple like (256, 256)

    # Center crop using shortest side, then resize
    # FFmpeg filter: crop='min(iw\,ih)':min(iw\,ih), then scale=256:256
    crop_filter = (
        f"crop='min(iw,ih)':'min(iw,ih)',scale={width}:{height}"
    )

    output_pattern = os.path.join(output_dir, "%04d.jpg")
    command = [
        "ffmpeg",
        "-hwaccel", "cuda",  # Enable GPU acceleration
        "-i", video_path,
        "-vf", crop_filter,
        "-qscale:v", "2",  # Quality (lower = better), optional
        "-vsync", "0",
        output_pattern,
        "-hide_banner",
        "-loglevel", "error"
    ]

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg failed on {video_path}: {e}")


def run_mp_cmd(processes, process_func, process_args):
    with Pool(processes) as p:
        outputs = list(
            tqdm(p.imap(process_func, process_args), total=len(process_args))
        )
    return outputs


def run_cmd(func, args):
    return func(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Data process for Visual Alignment Constraint for Continuous Sign Language Recognition."
    )
    parser.add_argument("--dataset", type=str, default="how2sign", help="save prefix")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="../dataset/how2sign",
        help="path to the dataset",
    )
    parser.add_argument(
        "--annotation-prefix",
        type=str,
        default="how2sign_realigned_{}.csv",
        help="annotation prefix",
    )
    parser.add_argument(
        "--output-res",
        type=str,
        default="256x256",
        help="resize resolution for image sequence",
    )
    parser.add_argument(
        "--process-image", "-p", action="store_true", help="resize image",   
    )
    parser.add_argument(
        "--multiprocessing",
        "-m",
        action="store_true",
        help="whether adopts multiprocessing to accelate the preprocess",
    )

    args = parser.parse_args()
    mode = ["val", "dev", "train"]
    existing_tokens = (
        [(x[0], x[1]) for x in np.load(f"./{args.dataset}/gloss_dict.npy", allow_pickle = True).tolist().items()]
        if os.path.exists(f"./{args.dataset}/gloss_dict.npy")
        else []
    )
    if len(existing_tokens) == 0:
        print("cached tokens are empty. please run agin!!")
    # print(existing_tokens[:10])
    low_ocurrance_token = set([x[0] for x in existing_tokens if x[1][1] < 50])
    # low_ocurrance_token = set()

    sign_dict = dict()
    if not os.path.exists(f"./{args.dataset}"):
        os.makedirs(f"./{args.dataset}")
    for md in mode:
        # generate information dict
        information = csv2dict(
            f"{args.dataset_root}/{args.annotation_prefix.format(md)}",
            dataset_type=md,
            dataset_root=args.dataset_root,
            unknownTokens=low_ocurrance_token,
        )
        np.save(f"./{args.dataset}/{md}_info.npy", information)
        # update the total gloss dict
        sign_dict_update(sign_dict, information)
        # generate groudtruth stm for evaluation
        generate_gt_stm(
            information, f"./{args.dataset}/{args.dataset}-groundtruth-{md}.stm"
        )
        # resize images
        video_index = np.arange(len(information) - 1)
        print(f"Resize image to {args.output_res}")
        if args.process_image:
            if args.multiprocessing:
                run_mp_cmd(
                    10,
                    partial(
                        resize_dataset, dsize=args.output_res, info_dict=information
                    ),
                    video_index,
                )
            else:
                for idx in tqdm(video_index):
                    run_cmd(
                        partial(
                            resize_dataset, dsize=args.output_res, info_dict=information
                        ),
                        idx,
                    )

    sign_dict = sorted(sign_dict.items(), key=lambda d: d[0])
    top_highest = sorted([x for x in sign_dict if x[0] != '<unk>'], key=lambda x: x[1], reverse=True)

    save_dict = {"<UNK>": [0, 0]}
    for idx, (key, value) in enumerate(sign_dict):
        save_dict[key] = [idx + 1, value]

    # print(save_dict)
    # np.save(f"./{args.dataset}/gloss_dict.npy", save_dict)

    pattern = re.compile(r"^[a-zA-Z0-9]+$")
    invalid_items = [
        f"{s} <---> {count}" for (s, count) in sign_dict if not pattern.match(s)
    ]

    with open(f"./{args.dataset}/invalid_items.txt", "w") as f:
        for x in invalid_items:
            f.write(f"{x}\n")

    # print(top_highest[50:])

    import matplotlib.pyplot as plt


    frequencies = np.array([value for key, value in top_highest if key != '<UNK>'])
    freq_sorted = np.sort(frequencies)[::-1]
    ranks = np.arange(1, len(freq_sorted) + 1)

    # --- Compute metrics ---
    # 1) Zipf’s Law Fit (log–log)
    log_r = np.log(ranks)
    log_f = np.log(freq_sorted)
    slope, intercept = np.polyfit(log_r, log_f, 1)

    # 2) Shannon Entropy
    p = frequencies / frequencies.sum()
    entropy = -(p * np.log(p)).sum()

    # 3) Gini Coefficient
    sorted_f = np.sort(frequencies)
    cum_f = np.cumsum(sorted_f)
    cum_p = cum_f / cum_f[-1]
    lorenz = np.insert(cum_p, 0, 0)
    x = np.linspace(0, 1, len(lorenz))
    gini = 1 - 2 * np.trapz(lorenz, x)

    # 4) Type–Token Ratio
    TTR = len(frequencies) / frequencies.sum()

    # Print metrics
    print(f"Shannon Entropy:        {entropy:.4f}")
    print(f"Gini Coefficient:       {gini:.4f}")
    print(f"Type–Token Ratio (TTR): {TTR:.6f}")

    # --- Plot 1: Zipf plot with fitted line ---
    plt.figure(figsize=(8, 5))
    plt.scatter(ranks, freq_sorted, s=8, label='Data')
    plt.plot(ranks, np.exp(intercept) * ranks**slope, 'r-', label=f'Fit: slope={slope:.2f}')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Rank (log scale)')
    plt.ylabel('Frequency (log scale)')
    plt.title('Zipf Plot of Gloss Frequencies')
    plt.legend()
    plt.grid(True, which='both', ls='--', lw=0.5)
    plt.tight_layout()
    plt.savefig(f"./{args.dataset}/Zipf_Plot_of_Gloss_Frequencies.png")
    # plt.show()

    # --- Plot 2: Cumulative coverage ---
    cum_coverage = np.cumsum(freq_sorted) / freq_sorted.sum()
    plt.figure(figsize=(8, 5))
    plt.plot(ranks, cum_coverage, linewidth=2)
    plt.xlabel('Number of Gloss Types')
    plt.ylabel('Cumulative Coverage')
    plt.title('Cumulative Token Coverage by Top Glosses')
    plt.grid(axis='both', ls='--', lw=0.5)
    plt.tight_layout()
    plt.savefig(f"./{args.dataset}/Cumulative_Token_Coverage_by_Top_Glosses.png")
    # plt.show()

    # Plot the distribution of gloss frequencies without labels
    frequencies = [value for key, value in top_highest]

    plt.figure(figsize=(12, 6))
    plt.plot(frequencies, color="skyblue", marker="o")
    plt.ylabel("Frequencies")
    plt.title("Distribution of Gloss Frequencies")
    plt.tight_layout()

    # Save the plot before showing
    plt.savefig(f"./{args.dataset}/gloss_distribution_no_labels.png")
    # plt.show()

    # import IPython as ipy; ipy.embed()

    # Group top_highest into ranges of 50 (1-50, 51-100, etc.) and calculate total frequency
    range_groups = {}
    for key, value in top_highest:
        # Determine which range this value belongs to
        range_start = 1 if value < 50 else ((value - 1) // 100 * 100 + 1)
        range_end = range_start + 49
        range_key = f"{range_start}-{range_end}"

        # Add the actual frequency value to the total for this range
        if range_key not in range_groups:
            range_groups[range_key] = value
        else:
            range_groups[range_key] += value

    # Sort the ranges naturally
    sorted_ranges = sorted(range_groups.items(), key=lambda x: int(x[0].split("-")[0]))

    # Plot the grouped data
    range_labels, total_frequencies = zip(*sorted_ranges)
    plt.figure(figsize=(12, 6))
    plt.bar(range_labels, total_frequencies, color="skyblue")
    plt.xlabel("Gloss Frequency Range")
    plt.ylabel("Total Frequency (sum of values in range)")
    plt.title("Total Gloss Frequency by Range (Number of times × Value)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    # Save the plot before showing
    plt.savefig(f"./{args.dataset}/gloss_frequency_grouped_ranges_snowball.png")
    # plt.show()

# %%
