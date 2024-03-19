'''
Statistical significance test for various models
'''
import os
import sys
import seaborn as sns
import json
import pandas as pd
import statsmodels
import jiwer
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import pairwise_tukeyhsd
sys.path.append('/home/mkperez/scratch/speechbrain/AphasiaBank/helper_scripts')
from evaluation import *

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

    y_true_aggregate = []
    y_pred_aggregate = []
    df_list = []
    for utt_id, label_aug_para in labels_dict.items():
        if utt_id not in results_dict:
            continue

        # remove eps
        label_aug_para = [l for l in label_aug_para if '<eps>' not in l]

        # WER
        result_dict = results_dict[utt_id]
        pred_WER = [f"{k.split('_')[1]}" for k,v in result_dict.items() if '<eps>' not in k]
        word_recognition_labels = [l.split("/")[0] for l in label_aug_para if l != '<eps>']
        wer = compute_AWER_lists([word_recognition_labels], [pred_WER])
        
        # AWER
        result_dict = results_dict[utt_id]
        pred_AWER = [f"{k.split('_')[1]}/{PARAPHASIA_KEY[v]}" for k,v in result_dict.items() if '<eps>' not in k]
        awer = compute_AWER_lists([label_aug_para], [pred_AWER])
        
        # AWER_disj
        pred_AWER_disj = [f"{k.split('_')[1]} {PARAPHASIA_KEY[v]}" for k,v in result_dict.items() if '<eps>' not in k]
        label_aug_para_disj = [f"{p.split('/')[0]} {p.split('/')[1]}" for p in label_aug_para]
        pred_AWER_disj = " ".join(pred_AWER_disj).split()
        label_aug_para_disj = " ".join(label_aug_para_disj).split()
        # exit()
        awer_disj = compute_AWER_lists([label_aug_para_disj], [pred_AWER_disj])

        # AWER PD
        pred_AWER_PD = [f"{PARAPHASIA_KEY[v]}" if PARAPHASIA_KEY[v] in ['p','n'] else f"{k.split('_')[1]} {PARAPHASIA_KEY[v]}" for k,v in result_dict.items()]
        pred_AWER_PD = [x for x in pred_AWER_PD if '<eps>' not in x]
        label_aug_para_PD = [f"{p.split('/')[1]}" if p.split('/')[1] in ['p','n'] else f"{p.split('/')[0]} {p.split('/')[1]}" for p in label_aug_para]
        label_aug_para_PD = [x for x in label_aug_para_PD if '<eps>' not in x]
        pred_AWER_PD = " ".join(pred_AWER_PD).split()
        label_aug_para_PD = " ".join(label_aug_para_PD).split()
        awer_PD = compute_AWER_lists([label_aug_para_PD], [pred_AWER_PD])

        y_pred_para =  [f"{PARAPHASIA_KEY[v]}" for k,v in result_dict.items() if '<eps>' not in k]
        y_true_para = [[f"{p.split('/')[1]}" for p in label_aug_para]]
    
        y_pred_aggregate.append(y_pred_para)
        y_true_aggregate.append(y_true_para)


        df_loc = pd.DataFrame({
            'uids': [utt_id],
            'wer-err': wer['err'],
            'wer-tot': wer['tot'],
            'awer-err': awer['err'],
            'awer-tot': awer['tot'],
            'awer_disj-err': awer_disj['err'],
            'awer_disj-tot': awer_disj['tot'],
            'awer_PD-err': awer_PD['err'],
            'awer_PD-tot': awer_PD['tot'],
        })

        df_list.append(df_loc)
    df = pd.concat(df_list)
    return y_true_aggregate, y_pred_aggregate, df

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
        TD_per_utt_bin, TD_list_bin = compute_temporal_distance(y_true_fold, y_pred_fold, True)                                 
        TD_per_utt_multi, TD_list_multi = compute_temporal_distance(y_true_fold, y_pred_fold, False)

        TD_per_utt_p, TD_list_p = compute_temporal_distance_para_sp(y_true_fold, y_pred_fold, 'p')
        TD_per_utt_n, TD_list_n = compute_temporal_distance_para_sp(y_true_fold, y_pred_fold, 'n')
        TD_per_utt_s, TD_list_s = compute_temporal_distance_para_sp(y_true_fold, y_pred_fold, 's')


        df_loc = pd.DataFrame({
            'model': ['GPT-ASR' for _ in range(len(TD_list_bin))],
            'uids': awer_df_fold['uids'],
            'wer': awer_df_fold['wer-err']/awer_df_fold['wer-tot'],
            'awer': awer_df_fold['awer-err']/awer_df_fold['awer-tot'],
            'awer-disj': awer_df_fold['awer_disj-err']/awer_df_fold['awer_disj-tot'],
            'awer-PD': awer_df_fold['awer_PD-err']/awer_df_fold['awer_PD-tot'],
            'TD_bin': TD_list_bin,
            'TD_multi': TD_list_multi,
            'TD_p': TD_list_p,
            'TD_n': TD_list_n,
            'TD_s': TD_list_s,
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
        TD_per_utt_bin, TD_list_bin = compute_temporal_distance(y_true_fold, y_pred_fold, True)                                 
        TD_per_utt_multi, TD_list_multi = compute_temporal_distance(y_true_fold, y_pred_fold, False)

        TD_per_utt_p, TD_list_p = compute_temporal_distance_para_sp(y_true_fold, y_pred_fold, 'p')
        TD_per_utt_n, TD_list_n = compute_temporal_distance_para_sp(y_true_fold, y_pred_fold, 'n')
        TD_per_utt_s, TD_list_s = compute_temporal_distance_para_sp(y_true_fold, y_pred_fold, 's')


        df_loc = pd.DataFrame({
            'model': ['GPT-ASR' for _ in range(len(TD_list_bin))],
            'uids': awer_df_fold['uids'],
            'wer': awer_df_fold['wer-err']/awer_df_fold['wer-tot'],
            'awer': awer_df_fold['awer-err']/awer_df_fold['awer-tot'],
            'awer-disj': awer_df_fold['awer_disj-err']/awer_df_fold['awer_disj-tot'],
            'awer-PD': awer_df_fold['awer_PD-err']/awer_df_fold['awer_PD-tot'],
            'TD_bin': TD_list_bin,
            'TD_multi': TD_list_multi,
            'TD_p': TD_list_p,
            'TD_n': TD_list_n,
            'TD_s': TD_list_s,
        })
        # print(df_loc)
        # exit()
        df_list.append(df_loc)
    df = pd.concat(df_list)
    return df



# Extract csv

def extract_SB_json_spk(sb_dir, model_name):
    df = pd.read_csv(f"{sb_dir}/results/wer_utt_stat_sig.csv")
    df['model'] = model_name
    for stat in ['wer', 'awer-disj', 'awer-PD']:
        df[f'{stat}'] = df[f'{stat}-err'] / df[f'{stat}-tot']
    df = df[['wer','awer-disj', 'awer-PD','TD_bin', 'TD_multi', 'TD_p', 'TD_n', 'TD_s', 'model', 'uids']]
    return df


def anova_test(df):
    # scores = ['wer', 'awer-disj', 'awer-PD', 'TD_bin', 'TD_multi', 'TD_']
    scores = ['TD_bin', 'TD_multi', 'TD_p', 'TD_n', 'TD_s']
    sig_test = []
    for score in scores:
        print(score)
        df_score = df[[score, 'uids','model']]
        agg_df = df_score.groupby(['uids', 'model'], as_index=False)[score].mean()
        # Perform the Repeated Measures ANOVA
        anova = AnovaRM(data=agg_df, depvar=score, subject='uids', within=['model'])
        result = anova.fit()

        p_value = result.anova_table["Pr > F"][0]
        bonferroni_corrected = 0.05 / len(set(df_score['model'].unique()))
        if p_value < bonferroni_corrected:
            sig_test.append(score)
    
            # Tukey test
            score_list = df_score[score]
            model_list = df_score['model']
            tukey_results = pairwise_tukeyhsd(score_list, model_list, 0.05)
            print(tukey_results)
            # print(tukey_results.p-adj)
            print("\n")



if __name__ == "__main__":
    ROOT_DIR = "/home/mkperez/scratch/speechbrain/AphasiaBank/statistical_analysis/results/multi"
    all_models = []

    MTL_EXP = "/home/mkperez/scratch/speechbrain/AphasiaBank/ISresults/full_FT_MTL_Scripts/MTL-weighted_para/reduce-w_asr_w-0.6_S2S-hubert-Transformer-500"
    SS_EXP = "/home/mkperez/scratch/speechbrain/AphasiaBank/ISresults/full_FT_Transcription_Scripts/balanced_para_S2S-hubert-Transformer-500"
    # GPT_EXP = "/home/mkperez/scratch/speechbrain/AphasiaBank/statistical_analysis/results/multi/GPT"
    # GPT_ORACLE_EXP = "/home/mkperez/scratch/speechbrain/AphasiaBank/statistical_analysis/results/multi/GPT-Oracle"
    GPT_EXP = "/home/mkperez/scratch/speechbrain/AphasiaBank/statistical_analysis/results/multi/GPT4"
    GPT_ORACLE_EXP = "/home/mkperez/scratch/speechbrain/AphasiaBank/statistical_analysis/results/multi/GPT4-Oracle"

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

    # filter GPT uids which are not in the other sets
    all_models = set(df['model'].unique())
    filtered_df = df.groupby('uids').filter(lambda x: all_models.issubset(x['model'].unique()))

    # statistical significance test
    anova_test(filtered_df)