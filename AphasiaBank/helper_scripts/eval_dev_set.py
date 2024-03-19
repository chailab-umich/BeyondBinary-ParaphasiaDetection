import os
import shutil
import subprocess
from evaluation import *

BASE="/home/mkperez/scratch/speechbrain/AphasiaBank/ISresults/full_FT_MTL_Scripts/MTL-weighted_para"

def copy_awer_files(exp_dir):
    shutil.copy(f"{exp_dir}/awer_para.txt",f"{exp_dir}/awer_para_test.txt")


def edit_hyparam(yaml_file):
    with open(yaml_file) as fin:
        filedata = fin.read()
        # filedata = filedata.replace('test_multi.csv', "dev_multi.csv")
        filedata = filedata.replace('dev_multi.csv', "test_multi.csv")
        filedata = filedata.replace('train_flag: true', "train_flag: false")

    
    with open(yaml_file,'w') as fout:
        fout.write(filedata)

def edit_python_script(python_filename):
    with open(python_filename, 'r') as file:
        lines = file.readlines()
        modified_lines = []
        for i,line in enumerate(lines):
            if i+1 >= 871 and i+1 <= 875:
                modified_lines.append('#' + line)
            else:
                modified_lines.append(line)
    
    with open(python_filename, 'w') as file:
        file.writelines(modified_lines)



if __name__ == '__main__':
    # for w in [0.3,0.4,0.5,0.6,0.7]:
    # for w in [0.7]:
    #     EXP_DIR = f"{BASE}/reduce-w_asr_w-{w}_S2S-hubert-Transformer-500"

    #     # for fold in range(1,13):
    #     for fold in range(1,2):
    #         print(f"weight: {w} | fold: {fold}")
    #         FOLD_DIR = f"{EXP_DIR}/Fold-{fold}"

    #         hyperparam = f"{FOLD_DIR}/hyperparams.yaml"
    #         edit_hyparam(hyperparam)

    #         # copy awer
    #         copy_awer_files(FOLD_DIR)

    #         # python file
    #         edit_python_script(f'{FOLD_DIR}/train_MTL.py')

    #         # exit()

    #         # run eval
    #         env = os.environ.copy()
    #         env['CUDA_VISIBLE_DEVICES'] = '0'
    #         cmd = ['python', f'{FOLD_DIR}/train_MTL.py', hyperparam]
    #         p = subprocess.run(cmd, env=env)
    #         exit()
            
    
    for w in [0.3,0.4,0.5,0.6,0.7]:
        EXP_DIR = f"{BASE}/reduce-w_asr_w-{w}_S2S-hubert-Transformer-500"
        for fold in range(1,13):
            print(f"weight: {w} | fold: {fold}")
            FOLD_DIR = f"{EXP_DIR}/Fold-{fold}"

            # copy dev
            shutil.copy(f"{FOLD_DIR}/awer_para.txt",f"{FOLD_DIR}/awer_para_dev.txt")

            # copy test back to main
            shutil.copy(f"{FOLD_DIR}/awer_para_test.txt",f"{FOLD_DIR}/awer_para.txt")


        para_eval(EXP_DIR, 'mtl')

        