'''
Statistical significance test for various models
'''
import os
import seaborn as sns
import json
import pandas as pd
import statsmodels
import jiwer
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import pairwise_tukeyhsd

## GPT##
def extract_transcript(wer_path):
    '''
    Return dict of transcripts
    '''
    transcripts = {}
    gt_transcripts = {}
    switch=0
    with open(wer_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line.startswith("P") and len(line.split()) == 14 and switch == 0:
                utt_id = line.split()[0][:-1]
                switch=1
            elif switch == 1:
                switch = 2
                words = [w.strip() for w in line.split(";")]
                gt_transcripts[utt_id] = " ".join(words)
            elif switch == 2:
                switch = 3
            elif switch ==3:
                # pred
                words = [w.strip() for w in line.split(";")]
                transcripts[utt_id] = " ".join(words)
                switch = 0
    return transcripts, gt_transcripts


def compile_predictions_labels_awer(results_dict, labels_dict):
    '''
    compile predictions and labels
    results_dict = json predictions
    '''

    PARAPHASIA_KEY = {'non-paraphasic':'c', 'phonemic':'p','semantic':'s', 'neologistic':'n'}
    y_true_list = []
    y_pred_list = []
    # for utt_id, result_dict in results_dict.items():

    AWER_dict = {
                'wer-err':0, 'wer-tot':0,
                'awer-err':0, 'awer-tot':0,
                 'awer_disj-err':0, 'awer_disj-tot':0,
                 'awer_PD-err':0, 'awer_PD-tot':0,}
    
    for utt_id, label_aug_para in labels_dict.items():
        if utt_id not in results_dict:
            continue
        

        # WER
        result_dict = results_dict[utt_id]
        pred_WER = [f"{k.split('_')[1]}" for k,v in result_dict.items() if '<eps>' not in k]
        pred_WER_str = " ".join(pred_WER)
        word_recognition_labels = [l.split("/")[0] for l in label_aug_para]
        word_recognition_labels_str = " ".join(word_recognition_labels)

        # print(utt_id)
        # print(f"result_dict: {result_dict}")
        # print(f"label_aug_para: {label_aug_para}")
        # print(f"pred: {pred_WER_str}")
        # print(f"true: {word_recognition_labels_str}")
        # exit()
        measures = jiwer.compute_measures(word_recognition_labels_str, pred_WER_str)
        AWER_dict['wer-err']+= measures['substitutions'] + measures['deletions'] + measures['insertions']
        AWER_dict['wer-tot']+= measures['substitutions'] + measures['deletions'] + measures['insertions'] + measures['hits']

        # AWER
        result_dict = results_dict[utt_id]
        pred_AWER = [f"{k.split('_')[1]}/{PARAPHASIA_KEY[v]}" for k,v in result_dict.items() if '<eps>' not in k]
        pred_AWER_str = " ".join(pred_AWER)
        label_aug_para_str = " ".join(label_aug_para)
        measures = jiwer.compute_measures(label_aug_para_str, pred_AWER_str)
        AWER_dict['awer-err']+= measures['substitutions'] + measures['deletions'] + measures['insertions']
        AWER_dict['awer-tot']+= measures['substitutions'] + measures['deletions'] + measures['insertions'] + measures['hits']

        # AWER_disj
        pred_AWER_disj = [f"{k.split('_')[1]} {PARAPHASIA_KEY[v]}" for k,v in result_dict.items() if '<eps>' not in k]
        pred_AWER_disj_str = " ".join(pred_AWER_disj)
        label_aug_para_disj = [f"{p.split('/')[0]} {p.split('/')[1]}" for p in label_aug_para]
        label_aug_para_disj_str = " ".join(label_aug_para_disj)
        measures = jiwer.compute_measures(label_aug_para_disj_str, pred_AWER_disj_str)
        AWER_dict['awer_disj-err']+= measures['substitutions'] + measures['deletions'] + measures['insertions']
        AWER_dict['awer_disj-tot']+= measures['substitutions'] + measures['deletions'] + measures['insertions'] + measures['hits']

        # AWER PD
        pred_AWER_PD = [f"{PARAPHASIA_KEY[v]}" if PARAPHASIA_KEY[v] in ['p','n'] else f"{k.split('_')[1]} {PARAPHASIA_KEY[v]}" for k,v in result_dict.items()]
        pred_AWER_PD = [x for x in pred_AWER_PD if '<eps>' not in x]
        pred_AWER_PD_str = " ".join(pred_AWER_PD)

        label_aug_para_PD = [f"{p.split('/')[1]}" if p.split('/')[1] in ['p','n'] else f"{p.split('/')[0]} {p.split('/')[1]}" for p in label_aug_para]
        label_aug_para_PD = [x for x in label_aug_para_PD if '<eps>' not in x]
        label_aug_PD_str = " ".join(label_aug_para_PD)
        measures = jiwer.compute_measures(label_aug_PD_str, pred_AWER_PD_str)
        AWER_dict['awer_PD-err']+= measures['substitutions'] + measures['deletions'] + measures['insertions']
        AWER_dict['awer_PD-tot']+= measures['substitutions'] + measures['deletions'] + measures['insertions'] + measures['hits']


        # label_aug_para + 
        result_dict = results_dict[utt_id]
        labels_list = [w.split("/")[-1] for w in label_aug_para]

        # Prediction labels 
        pred_labels = ['c' if k.split("_") == '<eps>' else PARAPHASIA_KEY[v] for k,v in result_dict.items()]
        
        y_pred_list.append(pred_labels)
        y_true_list.append(labels_list)

        assert len(pred_labels) == len(labels_list)
    
    return y_true_list, y_pred_list, AWER_dict

def extract_labels(label_csv_path, gt_transcript_dict):
    # map label_csv -> gt_transcript alignment (from wer files)
    df = pd.read_csv(label_csv_path)
    labels = {}
    for i, row in df.iterrows():
        if row['ID'] not in gt_transcript_dict:
            continue
        gt_transcript = gt_transcript_dict[row['ID']]

        # print(row)
        gt_para_labels = []
        aug_para_arr = row['aug_para'].split()

        # go through gt_transcript
        # print(f"gt_transcript: {gt_transcript}")
        for word in gt_transcript.split():
            if word == "<eps>":
                gt_para_labels.append('<eps>/c')
            else:
                # # pop 
                next_valid_word = aug_para_arr.pop(0)
                gt_para_labels.append(next_valid_word)
        
        labels[row['ID']] = gt_para_labels

    return labels


## TD eval ##
def TD_helper(true_labels, predicted_labels):
    TTC = 0
    for i in range(len(true_labels)):
        # for paraphasia label
        if true_labels[i] != 'c':
            min_distance_for_label = max(i-0,len(true_labels)-i)
            for j in range(len(predicted_labels)):
                if true_labels[i] == predicted_labels[j]:
                    # check for min distance
                    if abs(i - j) < min_distance_for_label:
                        min_distance_for_label = abs(i - j)

            TTC += min_distance_for_label


    CTT = 0
    for j in range(len(predicted_labels)):
        if predicted_labels[j] != 'c':
            min_distance_for_label = max(j-0,len(predicted_labels)-j)
            for i in range(len(true_labels)):
                if true_labels[i] == predicted_labels[j]:
                    # check for min distance
                    if abs(i - j) < min_distance_for_label:
                        min_distance_for_label = abs(i - j)

            CTT += min_distance_for_label
    return TTC + CTT

def compute_temporal_distance(true_labels, predicted_labels):
    # Return list of TDs for each utterance
    TD_per_utt = []
    for true_label, pred_label in zip(true_labels, predicted_labels):
        TD_utt = TD_helper(true_label, pred_label)
        TD_per_utt.append(TD_utt)

    return sum(TD_per_utt) / len(TD_per_utt), TD_per_utt



def extract_labels_oracle(label_csv_path):
    # extract labels from asr wer.txt
    
    df = pd.read_csv(label_csv_path)
    labels = {}
    for i, row in df.iterrows():
        labels[row['ID']] = row['aug_para'].split()

    return labels

def extract_GPT_data_oracle(gpt_dir):
    awer_df_list = []
    y_true_list = []
    y_pred_list = []
    df_list = []
    for i in range(1,13):
        wer_filepath = f"{gpt_dir}/fold-{i}_wer.txt"
        json_filepath = f"{gpt_dir}/fold_{i}_output.json"
        LABEL_DIR="/home/mkperez/scratch/speechbrain/AphasiaBank/data/Fridriksson_para_best_Word"

        # load json
        assert os.path.exists(json_filepath)
        with open(json_filepath, 'r') as file:
            results = json.load(file)

        # get transcripts
        transcript_dict, gt_transcript_dict = extract_transcript(wer_filepath)

        # extract labels
        label_csv_path = f"{LABEL_DIR}/Fold_{i}/test_multi.csv"
        labels_dict = extract_labels_oracle(label_csv_path)

        # Get y_true and y_pred
        y_true_fold, y_pred_fold, awer_df_fold = compile_predictions_labels_awer(results, labels_dict)

        # TD
        TD_per_utt, TD_utt_list = compute_temporal_distance(y_true_fold, y_pred_fold)

        df_loc = pd.DataFrame({
            'model': ['GPT-Oracle'],
            'speaker': [i],
            'wer': [awer_df_fold['wer-err']/awer_df_fold['wer-tot']],
            'awer': [awer_df_fold['awer-err']/awer_df_fold['awer-tot']],
            'awer-disj': [awer_df_fold['awer_disj-err']/awer_df_fold['awer_disj-tot']],
            'awer-PD': [awer_df_fold['awer_PD-err']/awer_df_fold['awer_PD-tot']],
            'TD': [TD_per_utt]
        })
        # print(df_loc)
        # exit()
        df_list.append(df_loc)
    df = pd.concat(df_list)
    return df

def extract_GPT_data(gpt_dir):
    df_list = []
    for i in range(1,13):
        wer_filepath = f"{gpt_dir}/fold-{i}_wer.txt"
        json_filepath = f"{gpt_dir}/fold_{i}_output.json"
        LABEL_DIR="/home/mkperez/scratch/speechbrain/AphasiaBank/data/Fridriksson_para_best_Word"

        # load json
        assert os.path.exists(json_filepath)
        with open(json_filepath, 'r') as file:
            results = json.load(file)

        # get transcripts
        transcript_dict, gt_transcript_dict = extract_transcript(wer_filepath)

        # extract labels
        label_csv_path = f"{LABEL_DIR}/Fold_{i}/test_multi.csv"
        labels_dict = extract_labels(label_csv_path,gt_transcript_dict)

        # Get y_true and y_pred
        y_true_fold, y_pred_fold, awer_df_fold = compile_predictions_labels_awer(results, labels_dict)

        # TD
        TD_per_utt, TD_utt_list = compute_temporal_distance(y_true_fold, y_pred_fold)

        df_loc = pd.DataFrame({
            'model': ['GPT'],
            'speaker': [i],
            'wer': [awer_df_fold['wer-err']/awer_df_fold['wer-tot']],
            'awer': [awer_df_fold['awer-err']/awer_df_fold['awer-tot']],
            'awer-disj': [awer_df_fold['awer_disj-err']/awer_df_fold['awer_disj-tot']],
            'awer-PD': [awer_df_fold['awer_PD-err']/awer_df_fold['awer_PD-tot']],
            'TD': [TD_per_utt]
        })
        # print(df_loc)
        # exit()
        df_list.append(df_loc)
    df = pd.concat(df_list)
    return df


def extract_SB_json_spk(sb_dir, model_name):
    json_filepath = f"{sb_dir}/speaker_out.json"
    
    # load json
    assert os.path.exists(json_filepath)
    with open(json_filepath, 'r') as file:
        results = json.load(file)

    if 'utt-f1' in results:
        del results['utt-f1']
    df = pd.DataFrame(results)
    df['speaker'] = range(len(df))
    df['model'] = model_name
    return df


def anova_test(df):
    scores = ['wer', 'awer-disj', 'awer-PD', 'TD']
    sig_test = []
    for score in scores:
        # print(score)
        df_score = df[[score, 'speaker','model']]
        stats = df.groupby('model')[score].agg(['mean', 'std']).reset_index()
        print(f"--------------\n{score} | mean: {stats}\n-------------------")
        # Perform the Repeated Measures ANOVA
        anova = AnovaRM(data=df_score, depvar=score, subject='speaker', within=['model'])
        result = anova.fit()

        p_value = result.anova_table["Pr > F"][0]
        if p_value < 0.05:
            sig_test.append(score)
    
            # Tukey test
            score_list = df_score[score]
            model_list = df_score['model']
            tukey_results = pairwise_tukeyhsd(score_list, model_list, 0.05)
            print(tukey_results)
            print("\n")



if __name__ == "__main__":
    ROOT_DIR = "/home/mkperez/scratch/speechbrain/AphasiaBank/statistical_analysis/results/multi"
    SPKR_FLAG = True
    all_models = []

    MTL_EXP = "/home/mkperez/scratch/speechbrain/AphasiaBank/ISresults/full_FT_MTL_Scripts/MTL-loss_S2S-hubert-Transformer-500"
    SS_EXP = "/home/mkperez/scratch/speechbrain/AphasiaBank/ISresults/full_FT_Transcription_Scripts/bpe_ES_S2S-hubert-Transformer-500"
    GPT_EXP = "/home/mkperez/scratch/speechbrain/AphasiaBank/statistical_analysis/results/multi/GPT"
    GPT_ORACLE_EXP = "/home/mkperez/scratch/speechbrain/AphasiaBank/statistical_analysis/results/multi/GPT-Oracle"

    for sdir in os.listdir(ROOT_DIR):
        df_GPT = extract_GPT_data(GPT_EXP)
        all_models.append(df_GPT)


        df_GPT_oracle = extract_GPT_data_oracle(GPT_ORACLE_EXP)
        all_models.append(df_GPT_oracle)
        df_SS = extract_SB_json_spk(SS_EXP, "single_seq")
        all_models.append(df_SS)
        df_MTL = extract_SB_json_spk(MTL_EXP, "MTL")
        all_models.append(df_MTL)
        
        

    df = pd.concat(all_models, ignore_index=True)
    # print(df)
    # exit()
    # statistical significance test
    anova_test(df)