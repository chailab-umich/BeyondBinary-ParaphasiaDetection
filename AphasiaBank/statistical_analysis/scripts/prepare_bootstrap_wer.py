'''
Process wer data for bootstrap test
/home/mkperez/scratch/asr_stat_significance
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
import shutil
import sys
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


        df_loc = pd.DataFrame({
            'uids': [utt_id],
            'wer-err': wer['err'],
            'wer-tot': wer['tot'],
            'awer-err': awer['err'],
            'awer-tot': awer['tot'],
            'awer-disj-err': awer_disj['err'],
            'awer-disj-tot': awer_disj['tot'],
            'awer-PD-err': awer_PD['err'],
            'awer-PD-tot': awer_PD['tot'],
        })

        df_list.append(df_loc)
    return pd.concat(df_list)



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
def prepare_bootstrap_file(wdir, df1, df2, speaker_bool):
    model_1, model_2 = df1['model'].values[0], df2['model'].values[0]

    # sort by uids
    df1 = df1.sort_values(by='uids')
    df2 = df2.sort_values(by='uids')
    # print(df1)
    # exit()
    # for stat in ['wer', 'awer-disj', 'awer-PD']:
    for stat in ['TD_bin', 'TD_multi', 'TD_p', 'TD_n','TD_s']:
        filename = f"{wdir}/{model_1}_{model_2}_{stat}.txt"
        print(f"stat: {stat}")
        with open(filename, 'w') as w:
            assert len(df1) == len(df2)
            
            for i in tqdm(range(len(df1))):
                df1_row = df1.iloc[i]
                df2_row = df2.iloc[i]
                # print(df1_row)
                # print(df2_row)
                assert df1.iloc[i]['fold'] == df2.iloc[i]['fold']
                assert df1.iloc[i][f'uids'] == df2.iloc[i][f'uids']
                assert df1.iloc[i][f'{stat}-tot'] == df2.iloc[i][f'{stat}-tot'], f"uid: {df1.iloc[i]['uids']} | df1: {df1.iloc[i][f'{stat}-tot']} | df2: {df2.iloc[i][f'{stat}-tot']}"
                if speaker_bool:
                    w.write(f"{df1.iloc[i][f'{stat}-err']}|{df2.iloc[i][f'{stat}-err']}|{df2.iloc[i][f'{stat}-tot']}|{df2.iloc[i]['fold']}\n")
                else:
                    w.write(f"{df1.iloc[i][f'{stat}-err']}|{df2.iloc[i][f'{stat}-err']}|{df2.iloc[i][f'{stat}-tot']}\n")

            # print(f"{stat}")
            # print(f"{model_1}: {df1[f'{stat}-err'].values.sum() / df1[f'{stat}-tot'].values.sum()}" )
            # print(f"{model_2}: {df2[f'{stat}-err'].values.sum() / df2[f'{stat}-tot'].values.sum()}" )
        


if __name__ == "__main__":
    ROOT_DIR = "/home/mkperez/scratch/speechbrain/AphasiaBank/statistical_analysis/results/multi"


    MTL_EXP = "/home/mkperez/scratch/speechbrain/AphasiaBank/ISresults/full_FT_MTL_Scripts/MTL-weighted_para/reduce-w_asr_w-0.6_S2S-hubert-Transformer-500"
    SS_EXP = "/home/mkperez/scratch/speechbrain/AphasiaBank/ISresults/full_FT_Transcription_Scripts/balanced_para_S2S-hubert-Transformer-500"
    ORACLE_EXP = "/home/mkperez/scratch/speechbrain/AphasiaBank/statistical_analysis/results/multi/GPT"
    ROOT_STAT = "/home/mkperez/scratch/asr_stat_significance/AphasiaBank"
    
    # clear output dir
    if os.path.exists(ROOT_STAT):
        shutil.rmtree(ROOT_STAT)
    os.makedirs(ROOT_STAT)

    df_ss = extract_df(SS_EXP, "SS")
    df_mtl = extract_df(MTL_EXP, "MTL")
    df_oracle = extract_gpt_df(ORACLE_EXP, "GPT")

    speaker_fold_bool = True
    prepare_bootstrap_file(ROOT_STAT,df_ss, df_mtl, speaker_fold_bool)
    prepare_bootstrap_file(ROOT_STAT,df_ss, df_oracle, speaker_fold_bool)
    prepare_bootstrap_file(ROOT_STAT,df_mtl, df_oracle, speaker_fold_bool)

