'''
Run through all folds of Fridriksson
S2S model jointly optimized for both ASR and paraphasia detection
Model is first trained on proto dataset and PD is on 'pn'
'''
import os
import shutil
import subprocess
import time
import datetime
import pandas as pd
from sklearn.metrics import f1_score, recall_score, precision_score
from collections import Counter
import pickle
from scipy import stats
import re
import socket
from tqdm import tqdm
import jiwer
import json

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]



TOT_EPOCHS=120

def train_log_check(train_log_file, last_epoch):
    with open(train_log_file, 'r') as file:
        last_line = file.readlines()[-1].strip()

        if int(last_line.split()[2]) == last_epoch:
            return True
        else:
            print(f"Error, last line epoch = {last_line.split()[2]}")
            return False
      
def compute_maj_class(fold,para_type):
    # compute majority class for naive baseline
    data = f"/home/mkperez/speechbrain/AphasiaBank/data/Fridriksson_para/Fold_{fold}/train_{para_type}.csv"
    df = pd.read_csv(data)
    PARA_DICT = {'P':1, 'C':0}
    utt_tr = []
    word_tr = []
    for utt in df['aug_para']:
        utt_arr = []
        for p in utt.split():
            utt_arr.append(PARA_DICT[p.split("/")[-1]])
        utt_tr.append(max(utt_arr))
    
    utt_counter = Counter(utt_tr)
    maj_class_utt = utt_counter.most_common()[0][0]
    return maj_class_utt

## TD for multi
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

        # print(f"true_label: {true_label}")
        # print(f"pred_label: {pred_label}")
        # print(f"TD_utt: {TD_utt}\n")
        # print(f"TD_utt: {TD_utt}")
        # if TD_utt > 200:
        #     print(f"true_label: {true_label}")
        #     print(f"pred_label: {pred_label}")
        #     exit()
    

    return sum(TD_per_utt) / len(TD_per_utt), TD_per_utt
    


## TTR
def compute_time_tolerant_scores(true_labels, predicted_labels, n=0):
    # Input: true_labels and predicted labels is a list of lists of labels
    # Output: number of TP, FN, FP
    assert len(true_labels) == len(predicted_labels), "Length of true_labels and predicted_labels must be the same"

    TP = 0  # True Positives
    FN = 0  # False Negatives
    FP = 0  # False Positives

    
    for utt_true, utt_pred in zip(true_labels, predicted_labels):
        loc_TP=0
        loc_FN=0
        for i, (true_label, predicted_label) in enumerate(zip(utt_true, utt_pred)):
            neighborhood = utt_pred[max(i-n, 0):min(i+n+1, len(utt_pred))]

            if true_label != 'c' and true_label != '<eps>':
                if any(label == true_label for label in neighborhood):
                    TP += 1
                    loc_TP += 1
                else:
                    FN += 1
                    loc_FN += 1

            elif any(label in ['p','n','s']  for label in neighborhood):
                FP += 1

    # Calculating precision and recall
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    # Calculating F1-score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1_score, recall

def extract_wer(wer_file):
    with open(wer_file, 'r') as r:
        first_line = r.readline()
        wer = float(first_line.split()[1])
        err = int(first_line.split()[3])
        total = int(first_line.split()[5][:-1])
        
        wer_details = {'wer': wer, 'err': err, 'tot': total}


    wer_details_list = []
    with open(wer_file, 'r') as r:
        lines = r.readlines()
        switch = 0
        for line in lines:
            line = line.strip()
            if line.startswith("P") and len(line.split()) == 14 and switch == 0:
                utt_id = line.split()[0][:-1]
                switch=1
 

                err = int(line.split()[4])
                tot = int(line.split()[6][:-1])
                wer = err/tot
                wer_details_list.append(wer)

            elif switch == 1:
                # ground truth
                words = [w.strip() for w in line.split(";")]
                switch = 2


            elif switch == 2:
                switch = 3
            elif switch ==3:
                # pred
                words = [w.strip() for w in line.split(";")]
                switch = 0
         
    return wer_details, wer_details_list

def transcription_helper_MTL(words):
    # For a given list of words, return list of paraphasias (strings)
    paraphasia_list = []
    for i,w in enumerate(words):
        if '/' in w:
            paraphasia_list.append(w.split('/')[-1].lower())
        elif w == '<eps>':
            paraphasia_list.append('c')


    return paraphasia_list
    
def extract_word_level_paraphasias(wer_file):
    # Extract word-level paraphasias from transcription WER file
    # Words (no tag -> C)

    # AWER
    y_true = []
    y_pred = []
    with open(wer_file, 'r') as r:
        lines = r.readlines()
        switch = 0
        for line in lines:
            line = line.strip()
            if line.startswith("P") and len(line.split()) == 14 and switch == 0:
                utt_id = line.split()[0][:-1]
                switch=1
            elif switch == 1:
                # ground truth
                words = [w.strip() for w in line.split(";")]
                # print(f"gt: {words}")
                paraphasia_list = transcription_helper_MTL(words)
                # print(f"gt_para: {paraphasia_list}")
                y_true.append(paraphasia_list)
                switch = 2


            elif switch == 2:
                switch = 3
            elif switch ==3:
                # pred
                words = [w.strip() for w in line.split(";")]
                # print(f"PRED words: {words}")
                paraphasia_list = transcription_helper_MTL(words)
                # print(f"pred_para: {paraphasia_list}\n")
                y_pred.append(paraphasia_list)
                switch = 0
                # assert len(pred) == len(gt)
    return y_true, y_pred

## utt-level
def compute_utt_f1(list_list_ytrue, list_list_ypred):
    '''
    Compute utt-level f1 score
    '''
    all_ytrue = []
    all_ypred = []
    for utt_ytrue, utt_ypred in zip(list_list_ytrue, list_list_ypred):
        true_bin_utt = [0 if x == 'c' else 1 for x in utt_ytrue]
        pred_bin_utt = [0 if x == 'c' else 1 for x in utt_ypred]

        all_ytrue.append(max(true_bin_utt))
        all_ypred.append(max(pred_bin_utt))

    # f1-score
    f1 = f1_score(all_ytrue, all_ypred, average='macro')
    return f1

## AWER

def compute_AWER_disjoint(awer_file):
    '''
    return dictionary of number of errors and total words
    return list of wer's on utterance level for significance testing 
    '''

    def _disjoint_paraphasia_words(word_list):
        # separate words from paraphasias
        combined = []
        for w in words:
            w = w.lower()
            if '/' in w:
                combined.extend(w.split('/'))
            elif w == '<eps>':
                continue
            else:
                combined.append(w)
        return ' '.join(combined)

    wer_details = {'err':0, 'tot':0}
    wer_details_list = []
    with open(awer_file, 'r') as r:
        lines = r.readlines()
        switch = 0
        for line in lines:
            line = line.strip()
            if line.startswith("P") and len(line.split()) == 14 and switch == 0:
                utt_id = line.split()[0][:-1]
                switch=1
            elif switch == 1:
                # ground truth
                words = [w.strip() for w in line.split(";")]
                true_word_para_disjoint = _disjoint_paraphasia_words(words)
                # print(f"line: {line}")
                # print(f"word_para_disjoint: {word_para_disjoint}")
                # y_true.append(paraphasia_list)
                switch = 2


            elif switch == 2:
                switch = 3
            elif switch ==3:
                # pred
                words = [w.strip() for w in line.split(";")]
                pred_word_para_disjoint = _disjoint_paraphasia_words(words)
                switch = 0
                # assert len(pred) == len(gt)

                # jiwer
                measures = jiwer.compute_measures(true_word_para_disjoint,pred_word_para_disjoint)
                wer_details['err']+= measures['substitutions'] + measures['deletions'] + measures['insertions']
                wer_details['tot']+= measures['substitutions'] + measures['deletions'] + measures['insertions'] + measures['hits']
                wer_details_list.append(measures['wer'])

    return wer_details, wer_details_list


def get_metrics(fold_dir):
    '''
    Compute WER metric for a given fold dir
    Compile list of lists y_true and y_pred for paraphasia analysis
    '''
    awer_file = f"{fold_dir}/awer_para.txt"
    awer_details,_ = extract_wer(awer_file)


    wer_file = f"{fold_dir}/asr_wer.txt"
    wer_details,_ = extract_wer(wer_file)
    awer_disjoint, _ = compute_AWER_disjoint(awer_file)

    # Extract paraphasia sequence from wer.txt
    list_list_ytrue, list_list_ypred = extract_word_level_paraphasias(awer_file)

    result_df = pd.DataFrame({
        'wer-err': [wer_details['err']],
        'wer-tot': [wer_details['tot']],
        'awer-err': [awer_details['err']],
        'awer-tot': [awer_details['tot']],
        'awer_disj-err': [awer_disjoint['err']],
        'awer_disj-tot': [awer_disjoint['tot']],
    })

    return result_df, list_list_ytrue, list_list_ypred
    
## json output (statistical significance)
def output_json_statistical_significance(fold_dir, results_dir, spkr_flag):
    '''
    Compute speaker awer, TD, utt-f1
    '''

    if spkr_flag:
        json_out = f"{results_dir}/speaker_out.json"
    else:
        json_out = f"{results_dir}/utterance_out.json"
    if os.path.exists(json_out):
        with open(json_out) as f:
            results_dict = json.load(f)
    else:
        results_dict = {}

    # AWER
    awer_file = f"{fold_dir}/awer_para.txt"
    awer_details, awer_list = extract_wer(awer_file)
    awer_disjoint, awer_disjoint_list = compute_AWER_disjoint(awer_file)



    # Extract paraphasia sequence from wer.txt
    list_list_ytrue, list_list_ypred = extract_word_level_paraphasias(awer_file)


    # check assertion
    for y,p in zip(list_list_ytrue, list_list_ypred):
        assert len(y) == len(p)


    # TD
    TD_per_utt, TD_list = compute_temporal_distance(list_list_ytrue, list_list_ypred)


    # initialize dict
    metric_keys = ['awer', 'awer-disj', 'TD']
    for mk in metric_keys:
        if mk not in results_dict:
            results_dict[mk] = []
        
    if spkr_flag:
        # add metrics
        results_dict[f"awer"].append(awer_details['err']/awer_details['tot'])
        results_dict[f"awer-disj"].append(awer_disjoint['err']/awer_disjoint['tot'])
        results_dict[f"TD"].append(TD_per_utt)
        # results_dict[f"utt-f1"].append(utt_f1)
    else:
        # add metrics utt
        results_dict[f"awer"].extend(awer_list)
        results_dict[f"awer-disj"].extend(awer_disjoint_list)
        results_dict[f"TD"].extend(TD_list)
    assert len(awer_list) == len(awer_disjoint_list) == len(TD_list)

    # write
    with open(json_out, 'w') as f:
        json.dump(results_dict, f, indent=4)



def clean_FT_model_save(path):
    # keep only 1 checkpoint, remove optimizer
    save_dir = f"{path}/save"
    abs_directory = os.path.abspath(save_dir)


    files = os.listdir(abs_directory)

    # Filter files that start with 'CKPT'
    ckpt_files = [f for f in files if f.startswith('CKPT')]

    # If no CKPT files, return
    if not ckpt_files:
        print("No CKPT files found.")
        return

    # Sort files lexicographically, this works because the timestamp format is year to second
    ckpt_files.sort(reverse=True)

    # The first file in the list is the latest, assuming the naming convention is consistent
    latest_ckpt = ckpt_files[0]
    print(f"Retaining the latest CKPT file: {latest_ckpt}")


    # Remove all other CKPT files
    for ckpt in ckpt_files[1:]:
        shutil.rmtree(os.path.join(abs_directory, ckpt))
        print(f"Deleted CKPT file: {ckpt}")

    # remove optimizer
    optim_file = f"{abs_directory}/{latest_ckpt}/optimizer.ckpt"
    os.remove(optim_file)

def change_yaml(yaml_src,yaml_target,data_fold_dir,frid_fold,output_neurons,output_dir,base_model,freeze_arch_bool,mtl_asr_weight):
    # copy src to tgt
    shutil.copyfile(yaml_src,yaml_target)

    # edit target file
    train_flag = True
    reset_LR = True # if true, start lr with init_LR
    output_dir = f"{output_dir}/Fold-{frid_fold}"
    lr = 5.0e-4 # 1e-3 for frozen arch
    
    
    # copy original file over to new dir
    if not os.path.exists(output_dir):
        print("copying dir")
        shutil.copytree(base_model,output_dir, ignore_dangling_symlinks=True)
        clean_FT_model_save(output_dir)

        
        
    # replace with raw text
    with open(yaml_target) as fin:
        filedata = fin.read()
        filedata = filedata.replace('data_dir_PLACEHOLDER', f"{data_fold_dir}")
        filedata = filedata.replace('train_flag_PLACEHOLDER', f"{train_flag}")
        filedata = filedata.replace('FT_start_PLACEHOLDER', f"{reset_LR}")
        filedata = filedata.replace('epochs_PLACEHOLDER', f"{TOT_EPOCHS}")
        filedata = filedata.replace('frid_fold_PLACEHOLDER', f"{frid_fold}")
        filedata = filedata.replace('output_PLACEHOLDER', f"{output_dir}")
        filedata = filedata.replace('output_neurons_PLACEHOLDER', f"{output_neurons}")
        filedata = filedata.replace('lr_PLACEHOLDER', f"{lr}")
        filedata = filedata.replace('freeze_ARCH_PLACEHOLDER', f"{freeze_arch_bool}")
        filedata = filedata.replace('mtl_asr_weight_PLACEHOLDER', f"{mtl_asr_weight}")


        with open(yaml_target,'w') as fout:
            fout.write(filedata)

    return output_dir

if __name__ == "__main__":
    DATA_ROOT = "/home/mkperez/speechbrain/AphasiaBank/data/Fridriksson_para_best_Word"

    TRAIN_FLAG = True
    EVAL_FLAG = True
    OUTPUT_NEURONS=500
    FREEZE_ARCH = False
    MTL_ASR_WEIGHT = 0.7


    if FREEZE_ARCH:
        BASE_MODEL = f"ISresults/MTL_proto/S2S-hubert-Transformer-500"
        EXP_DIR = f"ISresults/MTL_Scripts/S2S-hubert-Transformer-500"
    else:
        BASE_MODEL = f"/home/mkperez/speechbrain/AphasiaBank/ISresults/full_FT_MTL_proto/S2S-hubert-Transformer-500"
        EXP_DIR = f"/home/mkperez/speechbrain/AphasiaBank/ISresults/full_FT_MTL_Scripts/Target-decoder/asr_weight-{MTL_ASR_WEIGHT}_S2S-hubert-Transformer-500"

    if TRAIN_FLAG:
        yaml_src = "/home/mkperez/speechbrain/AphasiaBank/hparams/dev/MTL_target_base.yml"
        yaml_target = "/home/mkperez/speechbrain/AphasiaBank/hparams/dev/MTL_target_fold.yml"
        start = time.time()
        
        i=1
        count=0
        while i <=12:
            data_fold_dir = f"{DATA_ROOT}/Fold_{i}"

            change_yaml(yaml_src,yaml_target,data_fold_dir,i,OUTPUT_NEURONS,EXP_DIR,BASE_MODEL,FREEZE_ARCH,MTL_ASR_WEIGHT)

            # # launch experiment
            # multi-gpu
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = '0'
            port = find_free_port()  # Get a free port.
            print(f"free port: {port}")
            cmd = ['python', '-m', 'torch.distributed.launch',
                   f'--master_port={str(port)}', 
                'train_target_MTL.py', f'{yaml_target}']
            
            p = subprocess.run(cmd, env=env)

            # p = subprocess.run(cmd)
            count+=1
            print(f"p.returncode: {p.returncode} | retry: {count}")
            # exit()

            if p.returncode == 0:
                i+=1
                count-=1
            if count >=5:
                print("too many fails")
                exit()


        end = time.time()
        elapsed = end-start
        print(f"Total Train runtime: {datetime.timedelta(seconds=elapsed)}")

    ##  Stat computation
    if EVAL_FLAG:
        results_dir = f"{EXP_DIR}/results"
        os.makedirs(results_dir, exist_ok=True)

        json_out = f"{results_dir}/utterance_out.json"
        if os.path.exists(json_out):
            os.remove(json_out)

        json_out = f"{results_dir}/speaker_out.json"
        if os.path.exists(json_out):
            os.remove(json_out)

        df_list = []
        y_true = [] # aggregate list of y_true(list)
        y_pred = []
        SPKR_FLAG_STAT_SIG=True
        for i in range(1,13):
        # for i in range(1,2):
            Fold_dir = f"{EXP_DIR}/Fold-{i}"
            result_df, list_list_ytrue, list_list_ypred = get_metrics(Fold_dir)
            output_json_statistical_significance(Fold_dir,results_dir,SPKR_FLAG_STAT_SIG)
            # Combine over all folds
            y_true.extend(list_list_ytrue)
            y_pred.extend(list_list_ypred)
            df_list.append(result_df)
 
        df = pd.concat(df_list)

        # TD
        TD_per_utt, _ = compute_temporal_distance(y_true, y_pred)

        # utt-level F1-score
        utt_f1 = compute_utt_f1(y_true, y_pred)

        # Recall-f1 localization
        zero_f1, zero_recall = compute_time_tolerant_scores(y_true, y_pred, n=0)
        one_f1, one_recall = compute_time_tolerant_scores(y_true, y_pred, n=1)
        two_f1, two_recall = compute_time_tolerant_scores(y_true, y_pred, n=2)


        with open(f"{results_dir}/Frid_metrics_multi.txt", 'w') as w:
            

            print("Time Tolerant Recall:")
            print(f"0: {zero_recall}")
            print(f"1: {one_recall}")
            print(f"2: {two_recall}")

            print("Time Tolerant F1")
            print(f"0: {zero_f1}")
            print(f"1: {one_f1}")
            print(f"2: {two_f1}")

            for k in ['wer','awer','awer_disj']:
                wer = df[f'{k}-err'].sum()/df[f'{k}-tot'].sum()
                print(f"{k}: {wer}")
                w.write(f"{k}: {wer}\n")
            print(f"TD per utt: {TD_per_utt}")  
            print(f"utt-level F1-score: {utt_f1}")

            w.write(f"Time Tolerant Recall:\n")
            w.write(f"0: {zero_recall}\n")
            w.write(f"1: {one_recall}\n")
            w.write(f"2: {two_recall}\n\n")

            w.write(f"Time Tolerant F1:\n")
            w.write(f"0: {zero_f1}\n")
            w.write(f"1: {one_f1}\n")
            w.write(f"2: {two_f1}\n\n")

            w.write(f"TD per utt: {TD_per_utt}\n\n")
            w.write(f"utt-level F1-score: {utt_f1}\n\n")


        

        

        


