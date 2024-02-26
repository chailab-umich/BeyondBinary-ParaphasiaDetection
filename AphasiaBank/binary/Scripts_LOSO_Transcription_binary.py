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
import Levenshtein as lev
import json

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

TOT_EPOCHS=100

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

    return sum(TD_per_utt) / len(TD_per_utt)
    


## TTR
def compute_time_tolerant_scores(true_labels, predicted_labels, n=0):
    # Input: true_labels and predicted labels is a list of lists of labels
    # Output: number of TP, FN, FP
    assert len(true_labels) == len(predicted_labels), "Length of true_labels and predicted_labels must be the same"

    TP = 0  # True Positives
    FN = 0  # False Negatives
    FP = 0  # False Positives

    for utt_true, utt_pred in zip(true_labels, predicted_labels):
        # print(f"utt_true: {utt_true}")
        # print(f"utt_pred: {utt_pred}")
        loc_TP = 0
        loc_FN = 0
        for i, (true_label, predicted_label) in enumerate(zip(utt_true, utt_pred)):
            neighborhood = utt_pred[max(i-n, 0):min(i+n+1, len(utt_pred))]

            if true_label != 'c' and true_label != '<eps>':
                if any(label == true_label for label in neighborhood):
                    TP += 1
                    loc_TP+=1
                else:
                    FN += 1
                    loc_FN+=1

            elif any(label in ['p','n','s']  for label in neighborhood):
                FP += 1
        # print(f"TP: {loc_TP} | FN: {loc_FN}\n")
    # exit()
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


    return wer_details
  
def transcription_helper(words):
    # For a given list of words, return list of paraphasias
    paraphasia_list = []
    for i,w in enumerate(words):
        if w.startswith("[") and w.endswith("]"):
            # paraphasia found for previous word
            paraphasia_list.pop(-1)
            
            # replace previous word label with paraphasia label
            paraphasia_list.append(w[1:-1])
        elif w == "<eps>":
            continue
        else:
            # paraphasia_list.append('C')
            paraphasia_list.append('c')

    return paraphasia_list
    
def extract_word_level_paraphasias(wer_file,paraphasia):
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
                words_no_eps = [w for w in words if w != '<eps>']
                # print(f"GT words: {words}")
                # print(f"GT words_no_eps: {words_no_eps}")
                # paraphasia_list = transcription_helper(words)
                # true_words = words
                # utt_true = paraphasia_list
                # print(f"GT paraphasia_list: {paraphasia_list}")
                y_true.append(words_no_eps)
                switch = 2


            elif switch == 2:
                switch = 3
            elif switch ==3:
                # pred
                words = [w.strip() for w in line.split(";")]
                words_no_eps = [w for w in words if w != '<eps>']
                # print(f"PRED words: {words}")
                # paraphasia_list = transcription_helper(words)
                # pred_words = words
                # utt_pred = paraphasia_list
                # print(f"PRED paraphasia_list: {paraphasia_list}")
                y_pred.append(words_no_eps)
                switch = 0

                # if len(utt_true) != len(utt_pred):
                #     print(f"words - gt: {true_words}")
                #     print(f"words - pred: {pred_words}")
                #     print(f"paraphasia - gt: {utt_true}")
                #     print(f"paraphasia - pred: {utt_pred}")
                #     exit()

    return y_true, y_pred


### word-level paraphasia extraction ###
def remove_paraphasia_tags(ytrue_list, ypred_list):
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
                if w.startswith('[') and w.endswith(']'):
                    # tag found
                    paraphasia_word = word_list[i-1] # prev word

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
                    tag_dict[w].append('c')

                    no_tag_word_list.append(w)
            # print(word_list)
            # print(tag_dict)
            # exit()
            return tag_dict,no_tag_word_list

        pred_tags, pred_words = tag_helper(y_pred)
        true_tags, true_words = tag_helper(y_true)

        pred_tag_list.append(pred_tags)
        true_tag_list.append(true_tags)

        ytrue_list_nowords.append(true_words)
        ypred_list_nowords.append(pred_words)

    return ytrue_list_nowords, ypred_list_nowords, true_tag_list, pred_tag_list
                
def get_alignment(ytrue_list, ypred_list):
    ytrue_list_align = []
    ypred_list_align = []
    for y_true, y_pred in zip(ytrue_list, ypred_list):
        # print(f"y_true: {y_true}")
        # print(f"y_pred: {y_pred}")
        align_true, align_pred = align_strings_with_editops(y_true, y_pred)

        # print(f"align_true: {align_true}")
        # print(f"align_pred: {align_pred}")
        ytrue_list_align.append(align_true.split())
        ypred_list_align.append(align_pred.split())

    
    return ytrue_list_align, ypred_list_align

def align_strings_with_editops(true_str, pred_str):
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
                aligned_pred.append('<eps>' if i < true_idx else pred_words[j])
                aligned_true.append('<eps>' if j < pred_idx else true_words[i])

        # Apply the edit operation
        if op == 'replace':
            aligned_true.append(true_words[true_idx])
            aligned_pred.append(pred_words[pred_idx])
            i += 1
            j += 1
        elif op == 'insert':
            aligned_true.append(true_words[true_idx])
            aligned_pred.append('<eps>')
            i += 1
        elif op == 'delete':
            aligned_true.append('<eps>')
            aligned_pred.append(pred_words[pred_idx])
            j += 1

    # Add remaining words
    while i < len(true_words):
        aligned_true.append(true_words[i])
        i += 1

    while j < len(pred_words):
        aligned_pred.append(pred_words[j])
        j += 1

    return ' '.join(aligned_true), ' '.join(aligned_pred)

def reinsert_paraphasia_tags(word_list, tag_list):
    reconstructed_word_list = []
    for words, tag_dict in zip(word_list, tag_list):
        # print(f"words: {words}")
        # print(f"tag: {tag_dict}")

        reconstructed_word = []
        for word in words:
            reconstructed_word.append(word)
            if word in tag_dict:
                # print(f"word: {word}")
                reconstructed_word.append(f'[{tag_dict[word][0]}]')
                tag_dict[word].pop(0)

        
        # print(f"reconstructed: {reconstructed_word}")
        reconstructed_word_list.append(reconstructed_word)
        # exit()
    return reconstructed_word_list


def extract_paraphasia_class_labels(word_para_list):
    all_utts = []
    for word_para in word_para_list:
        para_list = []
        for w in word_para:
            if w.startswith('[') and w.endswith(']'):
                para_list.append(w[1:-1])
            elif w == "<eps>":
                para_list.append('c')
        all_utts.append(para_list)

    return all_utts
        

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
def compute_AWER(list_ytrue, list_ypred):
    '''
    return dictionary of number of errors and total words
    '''

    def _combine_paraphasia_for_awer(para_lst):
        return_list = []
        for i,w in enumerate(para_lst):
            if w.startswith('[') and w.endswith(']'):
                return_list[-1] = return_list[-1]+f'/{w[1:-1]}'
            else:
                return_list.append(w)
        return ' '.join(return_list)
    wer_details = {'err':0, 'tot':0}
    for y_true, y_pred in zip(list_ytrue, list_ypred):
        ytrue_str = _combine_paraphasia_for_awer(y_true)
        ypred_str = _combine_paraphasia_for_awer(y_pred)

        measures = jiwer.compute_measures(ytrue_str,ypred_str)

        wer_details['err']+= measures['substitutions'] + measures['deletions'] + measures['insertions']
        wer_details['tot']+= measures['substitutions'] + measures['deletions'] + measures['insertions'] + measures['hits']

    return wer_details

def compute_AWER_disjoint(list_ytrue, list_ypred):
    '''
    return dictionary of number of errors and total words
    '''
    wer_details = {'err':0, 'tot':0}
    for y_true, y_pred in zip(list_ytrue, list_ypred):
        ytrue_str = " ".join(y_true)
        ypred_str = " ".join(y_pred)

        measures = jiwer.compute_measures(ytrue_str,ypred_str)

        wer_details['err']+= measures['substitutions'] + measures['deletions'] + measures['insertions']
        wer_details['tot']+= measures['substitutions'] + measures['deletions'] + measures['insertions'] + measures['hits']

    return wer_details

def get_metrics(fold_dir, paraphasia):
    '''
    Compute WER metric for a given fold dir
    Compile list of lists y_true and y_pred for paraphasia analysis
    '''
    wer_file = f"{fold_dir}/wer.txt"
    wer_details = extract_wer(wer_file)


    # Extract paraphasia sequence from wer.txt
    # list_list_unaligned_words, list_list_unaligned_words = extract_word_level_paraphasias(wer_file, paraphasia)

    # remove eps
    noalign_ytrue, noalign_ypred = extract_word_level_paraphasias(wer_file, paraphasia)
    # print(f"y_true[0]: {noalign_ytrue[i]}")
    # print(f"y_pred[0]: {noalign_ypred[i]}")

    # remove paraphasia tags
    noalign_ytrue, noalign_ypred, true_tag_list, pred_tag_list = remove_paraphasia_tags(noalign_ytrue, noalign_ypred)

    #align non_words
    align_ytrue, align_ypred = get_alignment(noalign_ytrue, noalign_ypred)

    # re-insert paraphasia tags
    y_true = reinsert_paraphasia_tags(align_ytrue,true_tag_list)
    y_pred = reinsert_paraphasia_tags(align_ypred,pred_tag_list)

    awer_details = compute_AWER(y_true, y_pred)
    awer_disjoint = compute_AWER_disjoint(y_true, y_pred)

    # extract word-level paraphasia lists
    list_list_ytrue = extract_paraphasia_class_labels(y_true)
    list_list_ypred = extract_paraphasia_class_labels(y_pred)

    # check assertion
    for y,p in zip(list_list_ytrue, list_list_ypred):
        assert len(y) == len(p)


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
def output_json_statistical_significance(fold_dir, paraphasia, results_dir, fold_num):
    '''
    Compute speaker awer, TD, utt-f1
    '''

    json_out = f"{results_dir}/speaker_out.json"
    if os.path.exists(json_out):
        with open(json_out) as f:
            results_dict = json.load(f)
    else:
        results_dict = {}

    wer_file = f"{fold_dir}/wer.txt"
    wer_details = extract_wer(wer_file)


    # Extract paraphasia sequence from wer.txt
    # list_list_unaligned_words, list_list_unaligned_words = extract_word_level_paraphasias(wer_file, paraphasia)

    # remove eps
    noalign_ytrue, noalign_ypred = extract_word_level_paraphasias(wer_file, paraphasia)

    # remove paraphasia tags
    noalign_ytrue, noalign_ypred, true_tag_list, pred_tag_list = remove_paraphasia_tags(noalign_ytrue, noalign_ypred)

    #align non_words
    align_ytrue, align_ypred = get_alignment(noalign_ytrue, noalign_ypred)

    # re-insert paraphasia tags
    y_true = reinsert_paraphasia_tags(align_ytrue,true_tag_list)
    y_pred = reinsert_paraphasia_tags(align_ypred,pred_tag_list)

    awer_details = compute_AWER(y_true, y_pred)
    awer_disjoint = compute_AWER_disjoint(y_true, y_pred)

    # extract word-level paraphasia lists
    list_list_ytrue = extract_paraphasia_class_labels(y_true)
    list_list_ypred = extract_paraphasia_class_labels(y_pred)

    # check assertion
    for y,p in zip(list_list_ytrue, list_list_ypred):
        assert len(y) == len(p)


    # TD
    TD_per_utt = compute_temporal_distance(list_list_ytrue, list_list_ypred)

    # utt-level F1-score
    utt_f1 = compute_utt_f1(list_list_ytrue, list_list_ypred)

    # initialize dict
    metric_keys = ['awer', 'awer-disj', 'TD', 'utt-f1']
    for mk in metric_keys:
        if mk not in results_dict:
            results_dict[mk] = []
        
    # add metrics
    results_dict[f"awer"].append(awer_details['err']/awer_details['tot'])
    results_dict[f"awer-disj"].append(awer_disjoint['err']/awer_disjoint['tot'])
    results_dict[f"TD"].append(TD_per_utt)
    results_dict[f"utt-f1"].append(utt_f1)

    # write
    with open(json_out, 'w') as f:
        json.dump(results_dict, f, indent=4)

# Model Run
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

def change_yaml(yaml_src,yaml_target,data_fold_dir,frid_fold,output_neurons,output_dir,base_model,freeze_arch_bool, para_type):
    # copy src to tgt
    shutil.copyfile(yaml_src,yaml_target)

    # edit target file
    train_flag = True
    reset_LR = True # if true, start lr with init_LR
    viz_utts = ['P5_P2_SW_C4-2','P8_B2_SA_C1-3','P12_6W_SW_C6-4','P8_T2_SW_C3-3','P2_T6_SW_C3-3','P2_B2_SE_C1-4'] # fold 1
    output_dir = f"{output_dir}/Fold-{frid_fold}"
    lr = 5.0e-4


    
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
        filedata = filedata.replace('vis_dev_PLACEHOLDER', f"{viz_utts}")
        filedata = filedata.replace('para_type_PLACEHOLDER', f"{para_type}")

        with open(yaml_target,'w') as fout:
            fout.write(filedata)

    return output_dir

if __name__ == "__main__":
    DATA_ROOT = "/home/mkperez/speechbrain/AphasiaBank/data/Fridriksson_para_best_Word"

    TRAIN_FLAG = False
    EVAL_FLAG = True
    OUTPUT_NEURONS=500
    FREEZE_ARCH = False
    PARA_TYPE = ['p','n','pn'][2]

    if FREEZE_ARCH:
        EXP_DIR = f"/home/mkperez/speechbrain/AphasiaBank/ISresults/Transcription_Scripts/vis_dev_S2S-hubert-Transformer-500"
        BASE_MODEL = f"/home/mkperez/speechbrain/AphasiaBank/ISresults/full_FT_Transcription_Proto/bpe_ES_S2S-hubert-Transformer-500"
    else:
        EXP_DIR = f"/home/mkperez/speechbrain/AphasiaBank/ISresults/full_FT_Transcription_Scripts/binary/{PARA_TYPE}_S2S-hubert-Transformer-500"
        BASE_MODEL = f"/home/mkperez/speechbrain/AphasiaBank/ISresults/full_FT_Transcription_Proto/binary_S2S-hubert-Transformer-500"

    if TRAIN_FLAG:
        yaml_src = "/home/mkperez/speechbrain/AphasiaBank/hparams/Scripts/binary_transcription_base.yml"
        yaml_target = f"/home/mkperez/speechbrain/AphasiaBank/hparams/Scripts/binary_transcription_{PARA_TYPE}.yml"
        start = time.time()
        
        i=1
        count=0
        while i <=12:
            data_fold_dir = f"{DATA_ROOT}/Fold_{i}"

            change_yaml(yaml_src,yaml_target,data_fold_dir,i,OUTPUT_NEURONS,EXP_DIR,BASE_MODEL,FREEZE_ARCH,PARA_TYPE)

            # # launch experiment
            # multi-gpu
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = '0'
            port = find_free_port()  # Get a free port.
            cmd = ['python', '-m', 'torch.distributed.launch',
                   f'--master_port={str(port)}', 
                'train_Transcription_binary.py', f'{yaml_target}',
                '--distributed_launch', '--distributed_backend=nccl', '--find_unused_parameters']
            
            p = subprocess.run(cmd, env=env)

            # p = subprocess.run(cmd)
            count+=1
            print(f"p.returncode: {p.returncode} | retry: {count}")

            if count >=5:
                print("Too many retries")
                exit()

            if p.returncode == 0:
                i+=1
                count = 0



        end = time.time()
        elapsed = end-start
        print(f"Total Train runtime: {datetime.timedelta(seconds=elapsed)}")

    ##  Stat computation
    if EVAL_FLAG:
        PARAPHASIA_EVAL = ['p', 'n', 's', 'multi']
        results_dir = f"{EXP_DIR}/results"
        os.makedirs(results_dir, exist_ok=True)

        df_list = []
        list_list_ytrue_aggregate = []
        list_list_ypred_aggregate = []

        for i in range(1,13):
            Fold_dir = f"{EXP_DIR}/Fold-{i}"
            result_df, list_list_ytrue, list_list_ypred = get_metrics(Fold_dir,PARAPHASIA_EVAL)
            output_json_statistical_significance(Fold_dir,PARAPHASIA_EVAL,results_dir, i)
            list_list_ytrue_aggregate.extend(list_list_ytrue)
            list_list_ypred_aggregate.extend(list_list_ypred)
            # print(f"list_list_ytrue_aggregate len: {len(list_list_ytrue_aggregate)}")
            df_list.append(result_df)
        df = pd.concat(df_list)


        # Recall-f1 localization
        zero_f1, zero_recall = compute_time_tolerant_scores(list_list_ytrue_aggregate, list_list_ypred_aggregate, n=0)
        one_f1, one_recall = compute_time_tolerant_scores(list_list_ytrue_aggregate, list_list_ypred_aggregate, n=1)
        two_f1, two_recall = compute_time_tolerant_scores(list_list_ytrue_aggregate, list_list_ypred_aggregate, n=2)

        # TD
        TD_per_utt = compute_temporal_distance(list_list_ytrue_aggregate, list_list_ypred_aggregate)

        # utt-level F1-score
        utt_f1 = compute_utt_f1(list_list_ytrue_aggregate, list_list_ypred_aggregate)

        with open(f"{results_dir}/Frid_metrics_{PARAPHASIA_EVAL}.txt", 'w') as w:

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


        

        

        


