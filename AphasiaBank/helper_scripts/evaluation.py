"""
Evaluation functions
"""

import jiwer
from sklearn.metrics import f1_score, recall_score, precision_score
import pandas as pd
import os
import Levenshtein as lev
import json

## TD
def TD_helper_all(true_labels, predicted_labels):
    TTC = 0
    for i in range(len(true_labels)):
        # for paraphasia label
        if true_labels[i] != "c":
            min_distance_for_label = max(i - 0, len(true_labels))
            for j in range(len(predicted_labels)):
                if true_labels[i] == predicted_labels[j]:
                    # check for min distance
                    if abs(i - j) < min_distance_for_label:
                        min_distance_for_label = abs(i - j)

            TTC += min_distance_for_label

    CTT = 0
    for j in range(len(predicted_labels)):
        if predicted_labels[j] != "c":
            min_distance_for_label = max(j - 0, len(predicted_labels))
            for i in range(len(true_labels)):
                if true_labels[i] == predicted_labels[j]:
                    # check for min distance
                    if abs(i - j) < min_distance_for_label:
                        min_distance_for_label = abs(i - j)

            CTT += min_distance_for_label
    tot_d = (TTC + CTT) / len(true_labels)
    return tot_d


def TD_helper_binary(true_labels, predicted_labels):
    # detect any paraphasia
    TTC = 0
    for i in range(len(true_labels)):
        # for paraphasia label
        if true_labels[i] != "c":
            min_distance_for_label = max(i - 0, len(true_labels))
            for j in range(len(predicted_labels)):
                if predicted_labels[j] != "c":
                    # check for min distance
                    if abs(i - j) < min_distance_for_label:
                        min_distance_for_label = abs(i - j)

            TTC += min_distance_for_label

    CTT = 0
    for j in range(len(predicted_labels)):
        if predicted_labels[j] != "c":
            min_distance_for_label = max(j - 0, len(predicted_labels))
            for i in range(len(true_labels)):
                if true_labels[i] != "c":
                    # check for min distance
                    if abs(i - j) < min_distance_for_label:
                        min_distance_for_label = abs(i - j)

            CTT += min_distance_for_label
    tot_d = (TTC + CTT) / len(true_labels)
    return tot_d


def TD_helper_para_sp(true_labels, predicted_labels, pclass):
    # print(f"true_labels: {true_labels}")
    # print(f"predicted_labels: {predicted_labels}")
    # print(f"pclass: {pclass}")

    TTC = 0
    for i in range(len(true_labels)):
        # for paraphasia label
        if true_labels[i] == pclass:
            min_distance_for_label = max(i - 0, len(true_labels))
            for j in range(len(predicted_labels)):
                if true_labels[i] == predicted_labels[j]:
                    # check for min distance
                    if abs(i - j) < min_distance_for_label:
                        min_distance_for_label = abs(i - j)

            TTC += min_distance_for_label

    CTT = 0
    for j in range(len(predicted_labels)):
        if predicted_labels[j] == pclass:
            min_distance_for_label = max(j - 0, len(predicted_labels))
            for i in range(len(true_labels)):
                if true_labels[i] == predicted_labels[j]:
                    # check for min distance
                    if abs(i - j) < min_distance_for_label:
                        min_distance_for_label = abs(i - j)

            CTT += min_distance_for_label
    tot_d = (TTC + CTT) / len(true_labels)
    return tot_d


def compute_temporal_distance(true_labels, predicted_labels, binary_PD):
    # Return list of TDs for each utterance
    TD_per_utt = []
    for true_label, pred_label in zip(true_labels, predicted_labels):
        # Binary PD
        if binary_PD:
            TD_utt = TD_helper_binary(true_label, pred_label)
        # Multiclass
        else:
            TD_utt = TD_helper_all(true_label, pred_label)
        TD_per_utt.append(TD_utt)

    return sum(TD_per_utt) / len(TD_per_utt), TD_per_utt


def compute_temporal_distance_para_sp(
    true_labels, predicted_labels, para_class
):
    # Return list of TDs for each utterance
    TD_per_utt = []
    for true_label, pred_label in zip(true_labels, predicted_labels):
        # para class
        TD_utt = TD_helper_para_sp(true_label, pred_label, para_class)
        TD_per_utt.append(TD_utt)

    return sum(TD_per_utt) / len(TD_per_utt), TD_per_utt


# AWER (need list preprocessing first)
def compute_AWER_lists(list_ytrue, list_ypred):
    """
    Given two arrays of true and pred, compute awer
    return dictionary of number of errors and total words
    """
    wer_details = {"err": [], "tot": []}
    count = 0
    for y_true, y_pred in zip(list_ytrue, list_ypred):
        ytrue_str = " ".join(y_true)
        ypred_str = " ".join(y_pred)

        measures = jiwer.compute_measures(ytrue_str, ypred_str)
        wer_details["err"].append(
            measures["substitutions"]
            + measures["deletions"]
            + measures["insertions"]
        )
        wer_details["tot"].append(len(y_true))

    return wer_details


# Utt-level F1 and recall
def _paraphasia_utt_level(para_type, list_seq):
    return_list = []
    for utt in list_seq:
        if para_type in utt:
            return_list.append(1)
        else:
            return_list.append(0)

    return return_list


def utt_level_statistics(list_ytrue, list_ypred):
    df_list = []
    for ptype in ["p", "n", "s"]:
        p_true = _paraphasia_utt_level(ptype, list_ytrue)
        p_pred = _paraphasia_utt_level(ptype, list_ypred)

        # f1
        f1 = f1_score(p_true, p_pred)
        recall = recall_score(p_true, p_pred, average="binary")

        df = pd.DataFrame({"para": [ptype], "f1": [f1], "recall": [recall],})
        df_list.append(df)

    return pd.concat(df_list)


""" Print and Log """
## generic preprocessing##
def prepare_AWER_seq(asr_para_seq):
    # concat <word> and <para>
    all_seqs = []
    for seq in asr_para_seq:
        new_seq = []
        for w in seq:
            if w.startswith("[") and w.endswith("]"):
                new_seq[-1] = new_seq[-1] + f"/{w[1:-1]}"
            elif w == "<eps>":
                continue
            else:
                new_seq.append(w)
        all_seqs.append(new_seq)
    return all_seqs


def prepare_AWER_disj(asr_para_seq):
    # remove <eps>
    all_seqs = []
    for seq in asr_para_seq:
        new_seq = []
        for w in seq:
            if w == "<eps>":
                continue
            new_seq.append(w)
        all_seqs.append(new_seq)
    return all_seqs


def prepare_AWER_PD(asr_para_seq):
    # remove asr words before paraphasia token
    all_seqs = []
    for seq in asr_para_seq:
        new_seq = []
        for w in seq:
            # if paraphasia, remove previous word
            if w in ["[p]", "[n]"]:
                new_seq.pop(-1)
            elif w == "<eps>":
                continue
            new_seq.append(w)
        all_seqs.append(new_seq)
    return all_seqs


def prepare_WER(asr_para_seq):
    # remove paraphasias
    # for wer eval
    all_seqs = []
    for seq in asr_para_seq:
        new_seq = []
        for w in seq:
            if w not in ["[c]", "[p]", "[n]", "[s]", "<eps>"]:
                new_seq.append(w)
        all_seqs.append(new_seq)
    return all_seqs


def extract_paraphasia_class_labels(word_para_list):
    # get paraphasia predictions only
    all_utts = []
    for word_para in word_para_list:
        para_list = []
        for w in word_para:
            if w.startswith("[") and w.endswith("]"):
                para_list.append(w[1:-1])
            elif w == "<eps>":
                para_list.append("c")
        all_utts.append(para_list)

    return all_utts


def _print_and_log(log_str, w):
    print(log_str)
    w.write(f"{log_str}\n")


def display_and_save_dfs(
    eval_dir, utt_df, fold_df, utt_stats_df, df_utt_stat_sig=None
):
    results_dir = f"{eval_dir}/results"
    os.makedirs(results_dir, exist_ok=True)

    if df_utt_stat_sig is not None:
        df_utt_stat_sig.to_csv(f"{results_dir}/wer_utt_stat_sig.csv")

    with open(f"{results_dir}/Frid_metrics_multi.txt", "w") as w:
        # Utt-level
        _print_and_log("____Combined Utterance____", w)
        for k in ["wer", "awer", "awer_disj", "awer_PD"]:
            err = (
                utt_df[f"{k}-err"].values.sum()
                / utt_df[f"{k}-tot"].values.sum()
            )
            _print_and_log(f"{k}: {err}", w)

        # TD
        for TD_met in ["TD_bin", "TD_multi", "TD_p", "TD_n", "TD_s"]:
            avg_TD = utt_df[TD_met].values.sum() / len(utt_df[TD_met].values)
            _print_and_log(f"{TD_met}: {avg_TD}", w)

        # _print_and_log("____Para Only___",w)
        # for i, row in utt_stats_df.iterrows():
        #     _print_and_log(f"{row['para']}: f1={row['f1']} | recall={row['recall']}",w)

        # # Fold specific stats (mean and std)
        # _print_and_log("____Fold Stats___",w)
        # for k in ['wer','awer','awer_disj', 'awer_PD', 'TD_bin', 'TD_multi']:
        #     log_str = f"{k}: {fold_df[k].values.mean()} ({fold_df[k].values.std()})"
        #     _print_and_log(log_str,w)


## specific ##
def mtl_extract_asr_para(awer_file):
    """
    return lists of [<word0>, <para0>, <word1>, <para1>, ...]
    para are enveloped with [] like [p], [n], [s]
    preserve <eps> for alignment of paraphasia-only evaluation metrics
    return uids
    """

    pred_words = []
    gt_words = []
    uids = []
    with open(awer_file, "r") as r:
        lines = r.readlines()
        switch = 0
        for line in lines:
            line = line.strip()
            if line.startswith("P") and len(line.split()) == 14 and switch == 0:
                utt_id = line.split()[0][:-1]
                uids.append(utt_id)
                switch = 1
            elif switch == 1:
                # ground truth
                words = [w.strip() for w in line.split(";")]
                new_extended_words = []
                for word in words:
                    if "<eps>" in word:
                        continue
                    else:
                        new_para_arr = [
                            word.split("/")[0],
                            f'[{word.split("/")[1].lower()}]',
                        ]
                        new_extended_words.extend(new_para_arr)
                gt_words.append(new_extended_words)
                switch = 2

            elif switch == 2:
                switch = 3
            elif switch == 3:
                # pred
                words = [w.strip() for w in line.split(";")]
                new_extended_words = []
                for word in words:
                    if "<eps>" in word:
                        continue
                    else:
                        new_para_arr = [
                            word.split("/")[0],
                            f'[{word.split("/")[1].lower()}]',
                        ]
                        new_extended_words.extend(new_para_arr)

                pred_words.append(new_extended_words)
                switch = 0

    return gt_words, pred_words, uids


def ss_extract_asr_para(wer_file):
    # Extract word-level paraphasias from transcription WER file
    # Words (no tag -> C)

    # AWER
    y_true = []
    y_pred = []
    uid_list = []
    with open(wer_file, "r") as r:
        lines = r.readlines()
        switch = 0
        for line in lines:
            line = line.strip()
            if line.startswith("P") and len(line.split()) == 14 and switch == 0:
                utt_id = line.split()[0][:-1]
                uid_list.append(utt_id)
                switch = 1
            elif switch == 1:
                # ground truth
                words = [w.strip() for w in line.split(";")]
                words_no_eps = [w for w in words if w != "<eps>"]
                y_true.append(words_no_eps)
                switch = 2

            elif switch == 2:
                switch = 3
            elif switch == 3:
                # pred
                words = [w.strip() for w in line.split(";")]
                words_no_eps = [w for w in words if w != "<eps>"]
                y_pred.append(words_no_eps)
                switch = 0

    return y_true, y_pred, uid_list


def ss_remove_paraphasia_tags(ytrue_list, ypred_list):
    pred_tag_list = []
    true_tag_list = []
    ytrue_list_nowords = []
    ypred_list_nowords = []
    for y_true, y_pred in zip(ytrue_list, ypred_list):

        def tag_helper(word_list):
            tag_dict = {}
            no_tag_word_list = []
            # print(f"word_list: {word_list}")
            for i, w in enumerate(word_list):
                if w.startswith("[") and w.endswith("]"):
                    # tag found
                    paraphasia_word = word_list[i - 1]  # prev word

                    # same tag found => skip (redundant) or first tag is paraphasia
                    if w == paraphasia_word or i == 0:
                        continue

                    ptag = w[1:-1]
                    # print(f"ptag: {ptag}")

                    # add tag
                    # if paraphasia_word not in tag_dict:
                    #     tag_dict[paraphasia_word] = []
                    # pop previous
                    # print(f"tag_dict[{paraphasia_word}]: {tag_dict[paraphasia_word]}")
                    tag_dict[paraphasia_word].pop(-1)
                    tag_dict[paraphasia_word].append(ptag)

                else:

                    if w not in tag_dict:
                        tag_dict[w] = []
                    tag_dict[w].append("c")

                    no_tag_word_list.append(w)
            # print(word_list)
            # print(tag_dict)
            # exit()
            return tag_dict, no_tag_word_list

        pred_tags, pred_words = tag_helper(y_pred)
        true_tags, true_words = tag_helper(y_true)

        pred_tag_list.append(pred_tags)
        true_tag_list.append(true_tags)

        ytrue_list_nowords.append(true_words)
        ypred_list_nowords.append(pred_words)

    return ytrue_list_nowords, ypred_list_nowords, true_tag_list, pred_tag_list


def ss_get_alignment(ytrue_list, ypred_list):
    ytrue_list_align = []
    ypred_list_align = []
    for y_true, y_pred in zip(ytrue_list, ypred_list):
        # print(f"y_true: {y_true}")
        # print(f"y_pred: {y_pred}")
        align_true, align_pred = _align_strings_with_editops(y_true, y_pred)

        # print(f"align_true: {align_true}")
        # print(f"align_pred: {align_pred}")
        ytrue_list_align.append(align_true.split())
        ypred_list_align.append(align_pred.split())

    return ytrue_list_align, ypred_list_align


def _align_strings_with_editops(true_str, pred_str):
    # Split the strings into words
    true_words = true_str
    pred_words = pred_str

    # Get the edit operations to transform pred_str to true_str
    edit_operations = lev.editops(pred_words, true_words)
    # print(edit_operations)
    aligned_true = []
    aligned_pred = []
    i = j = 0

    for op, pred_idx, true_idx in edit_operations:
        # Catch up to the current position
        while i < true_idx or j < pred_idx:
            if i < true_idx:
                aligned_true.append(true_words[i])
                i += 1
            if j < pred_idx:
                aligned_pred.append(pred_words[j])
                j += 1

            if i < true_idx or j < pred_idx:
                aligned_pred.append("<eps>" if i < true_idx else pred_words[j])
                aligned_true.append("<eps>" if j < pred_idx else true_words[i])

        # Apply the edit operation
        if op == "replace":
            aligned_true.append(true_words[true_idx])
            aligned_pred.append(pred_words[pred_idx])
            i += 1
            j += 1
        elif op == "insert":
            aligned_true.append(true_words[true_idx])
            aligned_pred.append("<eps>")
            i += 1
        elif op == "delete":
            aligned_true.append("<eps>")
            aligned_pred.append(pred_words[pred_idx])
            j += 1

    # Add remaining words
    while i < len(true_words):
        aligned_true.append(true_words[i])
        i += 1

    while j < len(pred_words):
        aligned_pred.append(pred_words[j])
        j += 1

    return " ".join(aligned_true), " ".join(aligned_pred)


def ss_reinsert_paraphasia_tags(word_list, tag_list):
    reconstructed_word_list = []
    for words, tag_dict in zip(word_list, tag_list):
        # print(f"words: {words}")
        # print(f"tag: {tag_dict}")

        reconstructed_word = []
        for word in words:
            reconstructed_word.append(word)
            if word in tag_dict:
                # print(f"word: {word}")
                reconstructed_word.append(f"[{tag_dict[word][0]}]")
                tag_dict[word].pop(0)

        # print(f"reconstructed: {reconstructed_word}")
        reconstructed_word_list.append(reconstructed_word)
        # exit()
    return reconstructed_word_list


# get_metrics() EVAL
def mtl_get_metrics(fold_dir, fold_num):
    """
    Compute WER metric for a given fold dir
    Compile list of lists y_true and y_pred for paraphasia analysis
    """
    awer_file = f"{fold_dir}/awer_para.txt"

    # list of <word> <para> for every word (preserved paraphasias)
    y_true, y_pred, uids = mtl_extract_asr_para(awer_file)

    # AWER
    y_true_awer = prepare_AWER_seq(y_true)
    y_pred_awer = prepare_AWER_seq(y_pred)
    awer = compute_AWER_lists(y_true_awer, y_pred_awer)

    # AWERdisj (remove <eps> )
    ytrue_disj = prepare_AWER_disj(y_true)
    ypred_disj = prepare_AWER_disj(y_pred)
    awer_disj = compute_AWER_lists(ytrue_disj, ypred_disj)

    # AWER-PD (remove paraphasia words and remove <eps> )
    ytrue_PD = prepare_AWER_PD(y_true)
    ypred_PD = prepare_AWER_PD(y_pred)
    awer_PD = compute_AWER_lists(ytrue_PD, ypred_PD)

    # prepare for wer (remove paraphasia labels and <eps>)
    ytrue_wer = prepare_WER(y_true)
    ypred_wer = prepare_WER(y_pred)
    wer = compute_AWER_lists(ytrue_wer, ypred_wer)

    # Extract paraphasia sequence from wer.txt
    list_list_ytrue = extract_paraphasia_class_labels(y_true)
    list_list_ypred = extract_paraphasia_class_labels(y_pred)

    # TD computation
    TD_per_utt_bin, TD_list_bin = compute_temporal_distance(
        list_list_ytrue, list_list_ypred, True
    )

    TD_per_utt_multi, TD_list_multi = compute_temporal_distance(
        list_list_ytrue, list_list_ypred, False
    )

    TD_per_utt_p, TD_list_p = compute_temporal_distance_para_sp(
        list_list_ytrue, list_list_ypred, "p"
    )
    TD_per_utt_n, TD_list_n = compute_temporal_distance_para_sp(
        list_list_ytrue, list_list_ypred, "n"
    )
    TD_per_utt_s, TD_list_s = compute_temporal_distance_para_sp(
        list_list_ytrue, list_list_ypred, "s"
    )

    # Combine all utterances (single metric)
    utt_stats_df = pd.DataFrame(
        {
            "wer-err": wer["err"],
            "wer-tot": wer["tot"],
            "awer-err": awer["err"],
            "awer-tot": awer["tot"],
            "awer_disj-err": awer_disj["err"],
            "awer_disj-tot": awer_disj["tot"],
            "awer_PD-err": awer_PD["err"],
            "awer_PD-tot": awer_PD["tot"],
            "TD_bin": TD_list_bin,
            "TD_multi": TD_list_multi,
            "TD_p": TD_list_p,
            "TD_n": TD_list_n,
            "TD_s": TD_list_s,
        }
    )

    # Compute mean and std across all folds
    fold_stats_df = pd.DataFrame(
        {
            "fold": [fold_num],
            "TD_bin": [TD_per_utt_bin],
            "TD_multi": [TD_per_utt_multi],
            "wer": [sum(wer["err"]) / sum(wer["tot"])],
            "awer": [sum(awer["err"]) / sum(awer["tot"])],
            "awer_disj": [sum(awer_disj["err"]) / sum(awer_disj["tot"])],
            "awer_PD": [sum(awer_PD["err"]) / sum(awer_PD["tot"])],
        }
    )

    # statistical significance testing (utt-level)
    df_awer_stat_sig = pd.DataFrame(
        {
            "uids": uids,
            "wer-err": wer["err"],
            "wer-tot": wer["tot"],
            "awer-disj-err": awer_disj["err"],
            "awer-disj-tot": awer_disj["tot"],
            "awer-PD-err": awer_PD["err"],
            "awer-PD-tot": awer_PD["tot"],
            "fold": [fold_num for _ in range(len(wer["err"]))],
        }
    )

    return (
        utt_stats_df,
        fold_stats_df,
        df_awer_stat_sig,
        list_list_ytrue,
        list_list_ypred,
    )


def ss_get_metrics(fold_dir, fold_num):
    """
    Compute WER metric for a given fold dir
    Compile list of lists y_true and y_pred for paraphasia analysis
    """
    wer_file = f"{fold_dir}/wer.txt"

    # Extract paraphasia sequence from wer.txt
    # list_list_unaligned_words, list_list_unaligned_words = extract_word_level_paraphasias(wer_file, paraphasia)

    # remove eps
    noalign_ytrue, noalign_ypred, uids = ss_extract_asr_para(wer_file)

    # remove paraphasia tags
    (
        noalign_ytrue,
        noalign_ypred,
        true_tag_list,
        pred_tag_list,
    ) = ss_remove_paraphasia_tags(noalign_ytrue, noalign_ypred)

    # align non_words
    align_ytrue, align_ypred = ss_get_alignment(noalign_ytrue, noalign_ypred)

    # re-insert paraphasia tags
    y_true = ss_reinsert_paraphasia_tags(align_ytrue, true_tag_list)
    y_pred = ss_reinsert_paraphasia_tags(align_ypred, pred_tag_list)

    # remove paraphasia labels for WER
    y_true_wer = prepare_WER(y_true)
    y_pred_wer = prepare_WER(y_pred)
    wer = compute_AWER_lists(y_true_wer, y_pred_wer)

    # process of AWER-PD (remove recognized words for paraphasias)
    y_true_awerPD = prepare_AWER_PD(y_true)
    y_pred_awerPD = prepare_AWER_PD(y_pred)
    awer_PD = compute_AWER_lists(y_true_awerPD, y_pred_awerPD)

    # AWER
    y_pred_awer = prepare_AWER_seq(y_pred)
    y_true_awer = prepare_AWER_seq(y_true)
    awer = compute_AWER_lists(y_true_awer, y_pred_awer)

    # awer disj
    y_pred_awer_disj = prepare_AWER_disj(y_pred)
    y_true_awer_disj = prepare_AWER_disj(y_true)
    awer_disj = compute_AWER_lists(y_true_awer_disj, y_pred_awer_disj)

    # extract word-level paraphasia lists
    list_list_ytrue = extract_paraphasia_class_labels(y_true)
    list_list_ypred = extract_paraphasia_class_labels(y_pred)

    # check assertion
    for y, p in zip(list_list_ytrue, list_list_ypred):
        assert len(y) == len(p)

    # TD computation
    TD_per_utt_bin, TD_list_bin = compute_temporal_distance(
        list_list_ytrue, list_list_ypred, True
    )

    TD_per_utt_multi, TD_list_multi = compute_temporal_distance(
        list_list_ytrue, list_list_ypred, False
    )

    TD_per_utt_p, TD_list_p = compute_temporal_distance_para_sp(
        list_list_ytrue, list_list_ypred, "p"
    )
    TD_per_utt_n, TD_list_n = compute_temporal_distance_para_sp(
        list_list_ytrue, list_list_ypred, "n"
    )
    TD_per_utt_s, TD_list_s = compute_temporal_distance_para_sp(
        list_list_ytrue, list_list_ypred, "s"
    )
    # exit()

    # Combine all utterances (single metric)
    utt_stats_df = pd.DataFrame(
        {
            "wer-err": wer["err"],
            "wer-tot": wer["tot"],
            "awer-err": awer["err"],
            "awer-tot": awer["tot"],
            "awer_disj-err": awer_disj["err"],
            "awer_disj-tot": awer_disj["tot"],
            "awer_PD-err": awer_PD["err"],
            "awer_PD-tot": awer_PD["tot"],
            "TD_bin": TD_list_bin,
            "TD_multi": TD_list_multi,
            "TD_p": TD_list_p,
            "TD_n": TD_list_n,
            "TD_s": TD_list_s,
        }
    )

    # Compute mean and std across all folds
    fold_stats_df = pd.DataFrame(
        {
            "fold": [fold_num],
            "TD_bin": [TD_per_utt_bin],
            "TD_multi": [TD_per_utt_multi],
            "wer": [sum(wer["err"]) / sum(wer["tot"])],
            "awer": [sum(awer["err"]) / sum(awer["tot"])],
            "awer_disj": [sum(awer_disj["err"]) / sum(awer_disj["tot"])],
            "awer_PD": [sum(awer_PD["err"]) / sum(awer_PD["tot"])],
        }
    )

    # statistical significance testing (utt-level)
    df_awer_stat_sig = pd.DataFrame(
        {
            "uids": uids,
            "wer-err": wer["err"],
            "wer-tot": wer["tot"],
            "awer-disj-err": awer_disj["err"],
            "awer-disj-tot": awer_disj["tot"],
            "awer-PD-err": awer_PD["err"],
            "awer-PD-tot": awer_PD["tot"],
            "fold": [fold_num for _ in range(len(wer["err"]))],
        }
    )

    return (
        utt_stats_df,
        fold_stats_df,
        df_awer_stat_sig,
        list_list_ytrue,
        list_list_ypred,
    )


def para_eval(eval_dir, model_name):
    assert model_name in ["single_seq", "mtl"]
    results_dir = f"{eval_dir}/results"
    os.makedirs(results_dir, exist_ok=True)

    y_true = []  # aggregate list of y_true(list)
    y_pred = []

    utt_stats = []
    fold_stats = []
    stat_sig_utt_df_list = []
    for i in range(1, 13):
        Fold_dir = f"{eval_dir}/Fold-{i}"
        if model_name == "mtl":
            (
                utt_stats_df,
                fold_stats_df,
                df_awer_stat_sig,
                list_list_ytrue,
                list_list_ypred,
            ) = mtl_get_metrics(Fold_dir, i)
        elif model_name == "single_seq":
            (
                utt_stats_df,
                fold_stats_df,
                df_awer_stat_sig,
                list_list_ytrue,
                list_list_ypred,
            ) = ss_get_metrics(Fold_dir, i)
        # Combine over all folds
        y_true.extend(list_list_ytrue)
        y_pred.extend(list_list_ypred)

        # aggregate dfs
        stat_sig_utt_df_list.append(df_awer_stat_sig)
        utt_stats.append(utt_stats_df)
        fold_stats.append(fold_stats_df)

    utt_df = pd.concat(utt_stats)
    fold_df = pd.concat(fold_stats)
    df_utt_stat_sig = pd.concat(stat_sig_utt_df_list)
    _, TD_list_multi = compute_temporal_distance(y_true, y_pred, False)
    _, TD_list_bin = compute_temporal_distance(y_true, y_pred, True)
    _, TD_list_p = compute_temporal_distance_para_sp(y_true, y_pred, "p")
    _, TD_list_n = compute_temporal_distance_para_sp(y_true, y_pred, "n")
    _, TD_list_s = compute_temporal_distance_para_sp(y_true, y_pred, "s")
    df_utt_stat_sig["TD_bin"] = TD_list_bin
    df_utt_stat_sig["TD_multi"] = TD_list_multi
    df_utt_stat_sig["TD_p"] = TD_list_p
    df_utt_stat_sig["TD_n"] = TD_list_n
    df_utt_stat_sig["TD_s"] = TD_list_s

    # print(fold_df)

    # utt-level F1-score
    utt_stats_df = utt_level_statistics(y_true, y_pred)

    display_and_save_dfs(
        eval_dir, utt_df, fold_df, utt_stats_df, df_utt_stat_sig
    )


## GPT
def extract_transcript(wer_path):
    """
    Return dict of transcripts
    """
    transcripts = {}
    gt_transcripts = {}
    switch = 0
    with open(wer_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line.startswith("P") and len(line.split()) == 14 and switch == 0:
                utt_id = line.split()[0][:-1]
                switch = 1
            elif switch == 1:
                switch = 2
                words = [w.strip() for w in line.split(";")]
                gt_transcripts[utt_id] = " ".join(words)
            elif switch == 2:
                switch = 3
            elif switch == 3:
                # pred
                words = [w.strip() for w in line.split(";")]
                transcripts[utt_id] = " ".join(words)
                switch = 0
    return transcripts, gt_transcripts


def compile_predictions_labels_awer(results_dict, labels_dict):
    """
    compile predictions and labels
    results_dict = json predictions
    """

    PARAPHASIA_KEY = {
        "non-paraphasic": "c",
        "phonemic": "p",
        "semantic": "s",
        "neologistic": "n",
    }
    y_true_list = []
    y_pred_list = []
    # for utt_id, result_dict in results_dict.items():

    AWER_dict = {
        "wer-err": [],
        "wer-tot": [],
        "awer-err": [],
        "awer-tot": [],
        "awer_disj-err": [],
        "awer_disj-tot": [],
        "awer_PD-err": [],
        "awer_PD-tot": [],
    }

    for utt_id, label_aug_para in labels_dict.items():
        if utt_id not in results_dict:
            continue

        # WER
        result_dict = results_dict[utt_id]
        pred_WER = [
            f"{k.split('_')[1]}"
            for k, v in result_dict.items()
            if "<eps>" not in k
        ]
        pred_WER_str = " ".join(pred_WER)
        word_recognition_labels = [l.split("/")[0] for l in label_aug_para]
        word_recognition_labels_str = " ".join(word_recognition_labels)

        measures = jiwer.compute_measures(
            word_recognition_labels_str, pred_WER_str
        )
        AWER_dict["wer-err"].append(
            measures["substitutions"]
            + measures["deletions"]
            + measures["insertions"]
        )
        AWER_dict["wer-tot"].append(
            measures["substitutions"]
            + measures["deletions"]
            + measures["insertions"]
            + measures["hits"]
        )

        # AWER
        result_dict = results_dict[utt_id]
        pred_AWER = [
            f"{k.split('_')[1]}/{PARAPHASIA_KEY[v]}"
            for k, v in result_dict.items()
            if "<eps>" not in k
        ]
        pred_AWER_str = " ".join(pred_AWER)
        label_aug_para_str = " ".join(label_aug_para)
        measures = jiwer.compute_measures(label_aug_para_str, pred_AWER_str)
        AWER_dict["awer-err"].append(
            measures["substitutions"]
            + measures["deletions"]
            + measures["insertions"]
        )
        AWER_dict["awer-tot"].append(
            measures["substitutions"]
            + measures["deletions"]
            + measures["insertions"]
            + measures["hits"]
        )

        # AWER_disj
        pred_AWER_disj = [
            f"{k.split('_')[1]} {PARAPHASIA_KEY[v]}"
            for k, v in result_dict.items()
            if "<eps>" not in k
        ]
        pred_AWER_disj_str = " ".join(pred_AWER_disj)
        label_aug_para_disj = [
            f"{p.split('/')[0]} {p.split('/')[1]}" for p in label_aug_para
        ]
        label_aug_para_disj_str = " ".join(label_aug_para_disj)
        measures = jiwer.compute_measures(
            label_aug_para_disj_str, pred_AWER_disj_str
        )
        AWER_dict["awer_disj-err"].append(
            measures["substitutions"]
            + measures["deletions"]
            + measures["insertions"]
        )
        AWER_dict["awer_disj-tot"].append(
            measures["substitutions"]
            + measures["deletions"]
            + measures["insertions"]
            + measures["hits"]
        )

        # AWER PD
        pred_AWER_PD = [
            f"{PARAPHASIA_KEY[v]}"
            if PARAPHASIA_KEY[v] in ["p", "n"]
            else f"{k.split('_')[1]} {PARAPHASIA_KEY[v]}"
            for k, v in result_dict.items()
        ]
        pred_AWER_PD = [x for x in pred_AWER_PD if "<eps>" not in x]
        pred_AWER_PD_str = " ".join(pred_AWER_PD)

        label_aug_para_PD = [
            f"{p.split('/')[1]}"
            if p.split("/")[1] in ["p", "n"]
            else f"{p.split('/')[0]} {p.split('/')[1]}"
            for p in label_aug_para
        ]
        label_aug_para_PD = [x for x in label_aug_para_PD if "<eps>" not in x]
        label_aug_PD_str = " ".join(label_aug_para_PD)
        measures = jiwer.compute_measures(label_aug_PD_str, pred_AWER_PD_str)
        AWER_dict["awer_PD-err"].append(
            measures["substitutions"]
            + measures["deletions"]
            + measures["insertions"]
        )
        AWER_dict["awer_PD-tot"].append(
            measures["substitutions"]
            + measures["deletions"]
            + measures["insertions"]
            + measures["hits"]
        )

        # label_aug_para +
        result_dict = results_dict[utt_id]
        labels_list = [w.split("/")[-1] for w in label_aug_para]

        # Prediction labels
        pred_labels = [
            "c" if k.split("_") == "<eps>" else PARAPHASIA_KEY[v]
            for k, v in result_dict.items()
        ]

        y_pred_list.append(pred_labels)
        y_true_list.append(labels_list)

        assert len(pred_labels) == len(labels_list)

    return y_true_list, y_pred_list, AWER_dict


def extract_labels(label_csv_path, gt_transcript_dict):
    # map label_csv -> gt_transcript alignment (from wer files)
    df = pd.read_csv(label_csv_path)
    labels = {}
    for i, row in df.iterrows():
        if row["ID"] not in gt_transcript_dict:
            continue
        gt_transcript = gt_transcript_dict[row["ID"]]

        # print(row)
        gt_para_labels = []
        aug_para_arr = row["aug_para"].split()

        # go through gt_transcript
        # print(f"gt_transcript: {gt_transcript}")
        for word in gt_transcript.split():
            if word == "<eps>":
                gt_para_labels.append("<eps>/c")
            else:
                # # pop
                next_valid_word = aug_para_arr.pop(0)
                gt_para_labels.append(next_valid_word)

        labels[row["ID"]] = gt_para_labels

    return labels


def extract_labels_oracle(label_csv_path):
    # extract labels from asr wer.txt

    df = pd.read_csv(label_csv_path)
    labels = {}
    for i, row in df.iterrows():
        labels[row["ID"]] = row["aug_para"].split()

    return labels


def GPT_eval(gpt_dir, model_name):
    fold_df_list = []
    utt_df_list = []
    y_true_aggregate = []
    y_pred_aggregate = []
    assert model_name in ["oracle", "asr"]
    for i in range(1, 13):
        wer_filepath = f"{gpt_dir}/fold-{i}_wer.txt"
        json_filepath = f"{gpt_dir}/fold_{i}_output.json"
        LABEL_DIR = "/home/mkperez/scratch/speechbrain/AphasiaBank/data/Fridriksson_para_best_Word"

        # load json
        assert os.path.exists(json_filepath)
        with open(json_filepath, "r") as file:
            results = json.load(file)

        # get transcripts
        transcript_dict, gt_transcript_dict = extract_transcript(wer_filepath)

        # extract labels
        label_csv_path = f"{LABEL_DIR}/Fold_{i}/test_multi.csv"

        if model_name == "oracle":
            labels_dict = extract_labels_oracle(label_csv_path)
        elif model_name == "asr":
            labels_dict = extract_labels(label_csv_path, gt_transcript_dict)

        # Get y_true and y_pred
        y_true_fold, y_pred_fold, AWER_dict = compile_predictions_labels_awer(
            results, labels_dict
        )

        # TD
        TD_per_utt_bin, TD_list_bin = compute_temporal_distance(
            y_true_fold, y_pred_fold, True
        )
        TD_per_utt_multi, TD_list_multi = compute_temporal_distance(
            y_true_fold, y_pred_fold, False
        )

        TD_per_utt_p, TD_list_p = compute_temporal_distance_para_sp(
            y_true_fold, y_pred_fold, "p"
        )
        TD_per_utt_n, TD_list_n = compute_temporal_distance_para_sp(
            y_true_fold, y_pred_fold, "n"
        )
        TD_per_utt_s, TD_list_s = compute_temporal_distance_para_sp(
            y_true_fold, y_pred_fold, "s"
        )

        y_true_aggregate.extend(y_true_fold)
        y_pred_aggregate.extend(y_pred_fold)

        utt_stats_df = pd.DataFrame(
            {
                "wer-err": AWER_dict["wer-err"],
                "wer-tot": AWER_dict["wer-tot"],
                "awer-err": AWER_dict["awer-err"],
                "awer-tot": AWER_dict["awer-tot"],
                "awer_disj-err": AWER_dict["awer_disj-err"],
                "awer_disj-tot": AWER_dict["awer_disj-tot"],
                "awer_PD-err": AWER_dict["awer_PD-err"],
                "awer_PD-tot": AWER_dict["awer_PD-tot"],
                "TD_bin": TD_list_bin,
                "TD_multi": TD_list_multi,
                "TD_p": TD_list_p,
                "TD_n": TD_list_n,
                "TD_s": TD_list_s,
            }
        )

        fold_stats_df = pd.DataFrame(
            {
                "fold": [i],
                "TD_bin": [TD_per_utt_bin],
                "TD_multi": [TD_per_utt_multi],
                "wer": [sum(AWER_dict["wer-err"]) / sum(AWER_dict["wer-tot"])],
                "awer": [
                    sum(AWER_dict["awer-err"]) / sum(AWER_dict["awer-tot"])
                ],
                "awer_disj": [
                    sum(AWER_dict["awer_disj-err"])
                    / sum(AWER_dict["awer_disj-tot"])
                ],
                "awer_PD": [
                    sum(AWER_dict["awer_PD-err"])
                    / sum(AWER_dict["awer_PD-tot"])
                ],
            }
        )

        fold_df_list.append(fold_stats_df)
        utt_df_list.append(utt_stats_df)
    df_fold = pd.concat(fold_df_list)
    utt_df_awer = pd.concat(utt_df_list)
    utt_stats_df = utt_level_statistics(y_true_aggregate, y_pred_aggregate)

    display_and_save_dfs(gpt_dir, utt_df_awer, df_fold, utt_stats_df)
