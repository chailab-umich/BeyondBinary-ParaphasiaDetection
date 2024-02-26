'''
Process wer data for bootstrap test
'''
import os
import seaborn as sns
import json
import pandas as pd
import statsmodels
import jiwer
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from tqdm import tqdm
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


def compile_predictions_labels_awer(results_dict, labels_dict):
    '''
    compile predictions and labels
    results_dict = json predictions
    '''

    PARAPHASIA_KEY = {'non-paraphasic':'c', 'phonemic':'p','semantic':'s', 'neologistic':'n'}

    df_list = []
    for utt_id, label_aug_para in labels_dict.items():
        if utt_id not in results_dict:
            continue
        
        # WER
        result_dict = results_dict[utt_id]
        pred_WER = [f"{k.split('_')[1]}" for k,v in result_dict.items() if '<eps>' not in k]
        pred_WER_str = " ".join(pred_WER)
        word_recognition_labels = [l.split("/")[0] for l in label_aug_para]
        word_recognition_labels_str = " ".join(word_recognition_labels)
        measures = jiwer.compute_measures(word_recognition_labels_str, pred_WER_str)
        wer_err = measures['substitutions'] + measures['deletions'] + measures['insertions']
        wer_tot = measures['substitutions'] + measures['deletions'] + measures['insertions'] + measures['hits']
        
        

        # AWER
        result_dict = results_dict[utt_id]
        pred_AWER = [f"{k.split('_')[1]}/{PARAPHASIA_KEY[v]}" for k,v in result_dict.items() if '<eps>' not in k]
        pred_AWER_str = " ".join(pred_AWER)
        label_aug_para_str = " ".join(label_aug_para)
        measures = jiwer.compute_measures(label_aug_para_str, pred_AWER_str)
        awer_err = measures['substitutions'] + measures['deletions'] + measures['insertions']
        awer_tot = measures['substitutions'] + measures['deletions'] + measures['insertions'] + measures['hits']

        # AWER_disj
        pred_AWER_disj = [f"{k.split('_')[1]} {PARAPHASIA_KEY[v]}" for k,v in result_dict.items() if '<eps>' not in k]
        pred_AWER_disj_str = " ".join(pred_AWER_disj)
        label_aug_para_disj = [f"{p.split('/')[0]} {p.split('/')[1]}" for p in label_aug_para]
        label_aug_para_disj_str = " ".join(label_aug_para_disj)
        measures = jiwer.compute_measures(label_aug_para_disj_str, pred_AWER_disj_str)
        awer_disj_err = measures['substitutions'] + measures['deletions'] + measures['insertions']
        awer_disj_tot = measures['substitutions'] + measures['deletions'] + measures['insertions'] + measures['hits']

        # AWER PD
        pred_AWER_PD = [f"{PARAPHASIA_KEY[v]}" if PARAPHASIA_KEY[v] in ['p','n'] else f"{k.split('_')[1]} {PARAPHASIA_KEY[v]}" for k,v in result_dict.items()]
        pred_AWER_PD = [x for x in pred_AWER_PD if '<eps>' not in x]
        pred_AWER_PD_str = " ".join(pred_AWER_PD)
        label_aug_para_PD = [f"{p.split('/')[1]}" if p.split('/')[1] in ['p','n'] else f"{p.split('/')[0]} {p.split('/')[1]}" for p in label_aug_para]
        label_aug_para_PD = [x for x in label_aug_para_PD if '<eps>' not in x]
        label_aug_PD_str = " ".join(label_aug_para_PD)
        measures = jiwer.compute_measures(label_aug_PD_str, pred_AWER_PD_str)
        awer_pd_err = measures['substitutions'] + measures['deletions'] + measures['insertions']
        awer_pd_tot = measures['substitutions'] + measures['deletions'] + measures['insertions'] + measures['hits']


        df_loc = pd.DataFrame({
            'uids': [utt_id],
            'wer-err': [wer_err],
            'wer-tot': [wer_tot],
            'awer-err': [awer_err],
            'awer-tot': [awer_tot],
            'awer-disj-err': [awer_disj_err],
            'awer-disj-tot': [awer_disj_tot],
            'awer-PD-err': [awer_pd_err],
            'awer-PD-tot': [awer_pd_tot],
        })
        df_list.append(df_loc)
    return pd.concat(df_list)


def compute_fold_stats(df):
    df = df.drop(columns=['model','uids'])
    tot_stats = df.groupby('fold').sum()

    metrics = ['wer', 'awer','awer-disj', 'awer-PD']
    for k in metrics:
        tot_stats[k] = tot_stats[f'{k}-err'] / tot_stats[f'{k}-tot']
        print(f"{k}: {tot_stats[k].values.mean()} ({tot_stats[k].values.std()})")


    exit()


def extract_gpt_df(gpt_dir, model_name):
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

        # df with uids, wer-err, tot 
        df_fold = compile_predictions_labels_awer(results, labels_dict)
        df_fold['fold'] = i
        df_list.append(df_fold)
    df = pd.concat(df_list)
    df['model'] = model_name
    return df

## Extract df from SB csv
def extract_df(sb_dir, model_name):
    stat_sig_csv = f"{sb_dir}/results/wer_utt_stat_sig.csv"

    df = pd.read_csv(stat_sig_csv)
    df['model'] = model_name

    return df
    


## prepare file
def prepare_bootstrap_file(df1, df2):
    ROOT_STAT = "/home/mkperez/scratch/asr_stat_significance/AphasiaBank"
    model_1, model_2 = df1['model'].values[0], df2['model'].values[0]

    # sort by uids
    df1 = df1.sort_values(by='uids')
    df2 = df2.sort_values(by='uids')

    for stat in ['wer', 'awer-disj', 'awer-PD']:
        filename = f"{ROOT_STAT}/{model_1}_{model_2}_{stat}.txt"
        with open(filename, 'w') as w:
            assert len(df1) == len(df2)
            for i in tqdm(range(len(df1))):
                df1_row = df1.iloc[i]
                df2_row = df2.iloc[i]
                # print(df1_row)
                # print(df2_row)
                assert df1.iloc[i]['fold'] == df2.iloc[i]['fold']
                assert df1.iloc[i][f'uids'] == df2.iloc[i][f'uids']
                w.write(f"{df1.iloc[i][f'{stat}-err']}|{df2.iloc[i][f'{stat}-err']}|{df2.iloc[i][f'{stat}-tot']}|{df2.iloc[i]['fold']}\n")


        


if __name__ == "__main__":
    ROOT_DIR = "/home/mkperez/scratch/speechbrain/AphasiaBank/statistical_analysis/results/multi"


    MTL_EXP = "/home/mkperez/scratch/speechbrain/AphasiaBank/ISresults/full_FT_MTL_Scripts/MTL-loss_S2S-hubert-Transformer-500"
    SS_EXP = "/home/mkperez/scratch/speechbrain/AphasiaBank/ISresults/full_FT_Transcription_Scripts/bpe_ES_S2S-hubert-Transformer-500"
    ORACLE_EXP = "/home/mkperez/scratch/speechbrain/AphasiaBank/statistical_analysis/results/multi/GPT"


    df_ss = extract_df(SS_EXP, "SS")
    df_mtl = extract_df(MTL_EXP, "MTL")
    df_oracle = extract_gpt_df(ORACLE_EXP, "GPT")

    compute_fold_stats(df_oracle)
    exit()

    prepare_bootstrap_file(df_ss, df_mtl)
    prepare_bootstrap_file(df_ss, df_oracle)
    prepare_bootstrap_file(df_mtl, df_oracle)

