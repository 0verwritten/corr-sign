import glob
import pdb
import copy
import numpy as np
from itertools import groupby
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Reference: https://github.com/ustc-slr/DilatedSLR/blob/master/lib/lib_metric.py
# More about ASR evaluation: https://www.nist.gov/system/files/documents/2021/08/03/OpenASR20_EvalPlan_v1_5.pdf


def load_groundtruth(fpath):
    file_info = open(fpath, 'r', encoding='utf-8').readlines()
    gt_dict = dict()
    for line in file_info:
        info = line[:-1].split(" ")[5:]
        info = [*filter(lambda x: len(x), info)]
        gt_dict[line.split(" ")[0]] = info
    return gt_dict


def load_prediction(fpath):
    file_info = open(fpath, 'r', encoding='utf-8').readlines()
    pre_dict = dict()
    for line in file_info:
        split = line[:-1].split(" ")
        file_name = split[0]
        wd = split[-1]
        if file_name not in pre_dict.keys():
            pre_dict[file_name] = [wd]
        else:
            pre_dict[file_name].append(wd)
    return pre_dict


def get_wer_delsubins(ref, hyp, merge_same=False, align_results=False,
                      penalty={'ins': 1, 'del': 1, 'sub': 1}):
    # whether merge glosses before evaluation
    hyp = hyp if not merge_same else [x[0] for x in groupby(hyp)]

    # initialization
    ref_lgt = len(ref) + 1
    hyp_lgt = len(hyp) + 1

    costs = np.ones((ref_lgt, hyp_lgt), dtype=np.int32) * 1e6
    # auxiliary values
    costs[0, :] = np.arange(hyp_lgt) * penalty['ins']
    costs[:, 0] = np.arange(ref_lgt) * penalty['del']

    backtrace = np.zeros((ref_lgt, hyp_lgt), dtype=np.int32)
    # auxiliary indexes, 0, 1, 2, 3 are corresponding to correct, substitute, insert and delete, respectively
    backtrace[0, :] = 2
    backtrace[:, 0] = 3

    # dynamic programming
    for i in range(1, ref_lgt):
        for j in range(1, hyp_lgt):
            if ref[i - 1] == hyp[j - 1]:
                costs[i, j] = min(costs[i - 1, j - 1], costs[i, j])
                backtrace[i, j] = 0
            else:
                sub_cost, ins_cost, del_cost = \
                    costs[i - 1, j - 1] + penalty['sub'], \
                    costs[i - 1, j] + penalty['del'], \
                    costs[i, j - 1] + penalty['ins']
                min_cost = min(del_cost, ins_cost, sub_cost)
                if min_cost < costs[i, j]:
                    costs[i, j] = min_cost
                    backtrace[i, j] = [sub_cost, ins_cost, del_cost].index(costs[i, j]) + 1

    # backtrace pointer
    bt_ptr = np.array([ref_lgt - 1, hyp_lgt - 1])
    bt_path = []
    while bt_ptr.min() > 0:
        if backtrace[bt_ptr[0], bt_ptr[1]] == 0:
            # if correct, move (-1, -1)
            bt_ptr = bt_ptr - 1
            op = 'C'
        elif backtrace[bt_ptr[0], bt_ptr[1]] == 1:
            # if substitute, move (-1, -1)
            bt_ptr = bt_ptr - 1
            op = 'S'
        elif backtrace[bt_ptr[0], bt_ptr[1]] == 2:
            # if delete, move (-1, 0)
            bt_ptr = bt_ptr + (-1, 0)
            op = 'D'
        elif backtrace[bt_ptr[0], bt_ptr[1]] == 3:
            # if insert, move (0, -1)
            bt_ptr = bt_ptr + (0, -1)
            op = 'I'
        else:
            assert "Unexpected Operation"
        bt_path.append((bt_ptr, op))

    # decode path
    aligned_gt = []
    aligned_pred = []
    results = []
    for i in range(bt_path[-1][0][0]):
        aligned_gt.append(ref[i])
        aligned_pred.append('*' * len(ref[i]))
        results.append('D' + ' ' * (len(ref[i]) - 1))
    for i in range(bt_path[-1][0][1]):
        aligned_pred.append(hyp[i])
        aligned_gt.append('*' * len(hyp[i]))
        results.append('I' + ' ' * (len(hyp[i]) - 1))
    for ptr, op in bt_path[::-1]:
        if op in ['C', 'S']:
            if align_results:
                delta_lgt = len(ref[ptr[0]]) - len(hyp[ptr[1]])
                ref_pad = 0 if delta_lgt > 0 else -delta_lgt
                hyp_pad = 0 if delta_lgt < 0 else delta_lgt
                aligned_gt.append(ref[ptr[0]] + ' ' * ref_pad)
                aligned_pred.append(hyp[ptr[1]] + ' ' * hyp_pad)
            else:
                aligned_gt.append(ref[ptr[0]])
                aligned_pred.append(hyp[ptr[1]])
        elif op == 'I':
            aligned_gt.append('*' * len(hyp[ptr[1]]))
            aligned_pred.append(hyp[ptr[1]])
        elif op == 'D':
            aligned_gt.append(ref[ptr[0]])
            aligned_pred.append('*' * len(ref[ptr[0]]))

        if op == 'C':
            results.append(' ' * (len(aligned_gt[-1])))
        else:
            results.append(op + ' ' * (len(aligned_gt[-1]) - 1))
    return aligned_gt, aligned_pred


def calculate_stats(gt, lstm_pred, conv_pred=None):
    stat_ret = {
        'wer_conv': 0,
        'wer_lstm': 0,
        'war': 0,
        'wdr': 0,
        'cnt': 0,
    }
    for i in range(len(gt)):
        if "*" not in gt[i]:
            stat_ret['cnt'] += 1
        if gt[i] != lstm_pred[i]:
            stat_ret['wer_lstm'] += 1
        if conv_pred is not None:
            if gt[i] != conv_pred[i]:
                stat_ret['wer_conv'] += 1
            if conv_pred[i] == gt[i] and lstm_pred[i] != gt[i]:
                stat_ret['wdr'] += 1
            if conv_pred[i] != gt[i] and lstm_pred[i] == gt[i]:
                stat_ret['war'] += 1
    return stat_ret


def sent_evaluation(**kwargs):
    if "conv_prediction" in kwargs.keys():
        ret1 = get_wer_delsubins(kwargs['gt'], kwargs['conv_prediction'],
                                 merge_same=kwargs['merge_same'],
                                 penalty=kwargs['penalty'])
        ret2 = get_wer_delsubins(kwargs['gt'], kwargs['lstm_prediction'],
                                 merge_same=kwargs['merge_same'],
                                 penalty=kwargs['penalty'])
        new_gt = get_wer_delsubins(
            ret1[0], ret2[0],
            merge_same=kwargs['merge_same'],
            penalty=kwargs['penalty'])[0]
        conv_pred = get_wer_delsubins(
            new_gt, kwargs['conv_prediction'],
            align_results=True,
            merge_same=kwargs['merge_same'],
            penalty=kwargs['penalty'])[1]
        lstm_pred = get_wer_delsubins(
            new_gt, kwargs['lstm_prediction'],
            align_results=True,
            merge_same=kwargs['merge_same'],
            penalty=kwargs['penalty'])[1]
        # print(new_gt)
        # print(new_ret1)
        # print(new_ret2)
        return calculate_stats(new_gt, lstm_pred, conv_pred)

    gt, lstm_pred = get_wer_delsubins(kwargs['gt'], kwargs['lstm_prediction'],
                                      merge_same=kwargs['merge_same'],
                                      penalty=kwargs['penalty'])
    
    return calculate_stats(gt, lstm_pred)


def sum_dict(dict_list):
    ret_dict = dict()
    for key in dict_list[0].keys():
        ret_dict[key] = sum([d[key] for d in dict_list])
    return ret_dict


# def wer_calculation(gt_path, primary_pred, auxiliary_pred=None):
#     gt = load_groundtruth(gt_path)
#     pred1 = load_prediction(primary_pred)
#     results_list = []
#     if auxiliary_pred is not None:
#         pred2 = load_prediction(auxiliary_pred)
#         for fileid, sent in gt.items():
#             sent_stat = sent_evaluation(
#                 info=fileid, gt=sent,
#                 merge_same=True,
#                 lstm_prediction=pred1[fileid],
#                 conv_prediction=pred2[fileid],
#                 penalty={'ins': 3, 'del': 3, 'sub': 4},
#             )
#             results_list.append(sent_stat)
#     else:
#         for fileid, sent in gt.items():
#             sent_stat = sent_evaluation(
#                 info=fileid, gt=sent,
#                 merge_same=True,
#                 lstm_prediction=pred1[fileid],
#                 penalty={'ins': 3, 'del': 3, 'sub': 4},
#             )
#             results_list.append(sent_stat)
#     results = sum_dict(results_list)
#     print(
#         f"WER_primary: {results['wer_lstm'] / results['cnt']: 2.2%}\n"
#         f"WER_auxiliary: {results['wer_conv'] / results['cnt']: 2.2%}\n"
#         f"WAR: {results['war'] / results['cnt']: 2.2%}\n"
#         f"WDR: {results['wdr'] / results['cnt']: 2.2%}"
#     )
#     return results['wer_lstm'] / results['cnt'] * 100

def wer_calculation(gt_path, primary_pred, auxiliary_pred=None):
    gt_data = load_groundtruth(gt_path)
    pred1 = load_prediction(primary_pred)
    all_aligned_gt = []
    all_aligned_pred = []
    results_list = []
    if auxiliary_pred is not None:
        pred2 = load_prediction(auxiliary_pred)
        for fileid, sent in gt_data.items():
            ret1 = get_wer_delsubins(sent, pred2[fileid],
                                     merge_same=True,
                                     penalty={'ins': 3, 'del': 3, 'sub': 4})
            ret2 = get_wer_delsubins(sent, pred1[fileid],
                                     merge_same=True,
                                     penalty={'ins': 3, 'del': 3, 'sub': 4})
            new_gt = get_wer_delsubins(ret1[0], ret2[0],
                                        merge_same=True,
                                        penalty={'ins': 3, 'del': 3, 'sub': 4})[0]
            conv_aligned = get_wer_delsubins(new_gt, pred2[fileid],
                                             align_results=True,
                                             merge_same=True,
                                             penalty={'ins': 3, 'del': 3, 'sub': 4})[1]
            lstm_aligned = get_wer_delsubins(new_gt, pred1[fileid],
                                             align_results=True,
                                             merge_same=True,
                                             penalty={'ins': 3, 'del': 3, 'sub': 4})[1]
            all_aligned_gt.extend(new_gt)
            all_aligned_pred.extend(lstm_aligned)
            sent_stat = calculate_stats(new_gt, lstm_aligned, conv_aligned)
            results_list.append(sent_stat)
    else:
        for fileid, sent in gt_data.items():
            gt_aligned, lstm_aligned = get_wer_delsubins(sent, pred1[fileid],
                                                          merge_same=True,
                                                          penalty={'ins': 3, 'del': 3, 'sub': 4})
            all_aligned_gt.extend(gt_aligned)
            all_aligned_pred.extend(lstm_aligned)
            # print(gt_aligned, lstm_aligned)
            sent_stat = calculate_stats(gt_aligned, lstm_aligned)
            results_list.append(sent_stat)
    results = sum_dict(results_list)
    wer = results['wer_lstm'] / results['cnt'] * 100
    print(
        f"WER_primary: {results['wer_lstm'] / results['cnt']: 2.2%}\n"
        f"WER_auxiliary: {results['wer_conv'] / results['cnt']: 2.2%}\n"
        f"WAR: {results['war'] / results['cnt']: 2.2%}\n"
        f"WDR: {results['wdr'] / results['cnt']: 2.2%}"
    )

    # Classic evaluation metrics on the aligned tokens
    # Overall multi-class accuracy (including the blank '*' token)
    overall_accuracy = accuracy_score(all_aligned_gt, all_aligned_pred)
    # For gesture detection, map tokens: '*' -> 0 (no gesture), others -> 1 (gesture)
    gt_binary = [0 if token == '*' else 1 for token in all_aligned_gt]
    pred_binary = [0 if token == '*' else 1 for token in all_aligned_pred]
    binary_accuracy = accuracy_score(gt_binary, pred_binary)
    binary_precision = precision_score(gt_binary, pred_binary, zero_division=0)
    binary_recall = recall_score(gt_binary, pred_binary, zero_division=0)
    binary_f1 = f1_score(gt_binary, pred_binary, zero_division=0)
    # Confusion matrix for all classes
    # conf_matrix = confusion_matrix(all_aligned_gt, all_aligned_pred)
    # Confusion matrix for binary gesture detection: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = confusion_matrix(gt_binary, pred_binary).ravel()

    print("\nClassic Evaluation Metrics:")
    print(f"Overall Accuracy (all classes): {overall_accuracy:.2%}")
    print("\nGesture Detection (binary):")
    print(f"  Accuracy: {binary_accuracy:.2%}")
    print(f"  Precision: {binary_precision:.2%}")
    print(f"  Recall: {binary_recall:.2%}")
    print(f"  F1 Score: {binary_f1:.2%}")
    # print("\nConfusion Matrix (all classes):")
    # print(conf_matrix)
    print("\nFalse Positives (model predicts gesture when none present):", fp)
    print("False Negatives (model misses a gesture):", fn)

    metrics = {
        "wer_primary_percent": wer,
        "wer_auxiliary_percent": results['wer_conv'] / results['cnt'] * 100,
        "war_percent": results['war'] / results['cnt'] * 100,
        "wdr_percent": results['wdr'] / results['cnt'] * 100,
        "overall_accuracy_all_classes": overall_accuracy,
        "binary_accuracy": binary_accuracy,
        "binary_precision": binary_precision,
        "binary_recall": binary_recall,
        "binary_f1": binary_f1,
        # "confusion_matrix_all_classes": conf_matrix.tolist(),
        "confusion_matrix_binary": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}
    }

    return wer, metrics

if __name__ == '__main__':
    wer_calculation('phoenix2014-groundtruth-dev.stm',
                    'out.output-hypothesis-dev.ctm')
    #                     'out.output-hypothesis-dev-conv.ctm')
