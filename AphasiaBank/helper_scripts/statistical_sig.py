'''
Compute evaluations for a given directory
'''
import os
from evaluation import *


if __name__ == '__main__':
    # MODEL_DIR = "/home/mkperez/scratch/speechbrain/AphasiaBank/ISresults/full_FT_MTL_Scripts/MTL-loss_S2S-hubert-Transformer-500"
    # MODEL_DIR = "/home/mkperez/scratch/speechbrain/AphasiaBank/ISresults/full_FT_MTL_Scripts/MTL-weighted_para/asr_w-0.7_S2S-hubert-Transformer-500"
    # para_eval(MODEL_DIR, 'mtl')

    MODEL_DIR = "/home/mkperez/scratch/speechbrain/AphasiaBank/ISresults/full_FT_Transcription_Scripts/bpe_ES_S2S-hubert-Transformer-500"
    para_eval(MODEL_DIR, 'single_seq')

    # GPT_DIR = "/home/mkperez/scratch/speechbrain/AphasiaBank/statistical_analysis/results/multi/GPT"
    # GPT_eval(GPT_DIR,'asr')
    # GPT_DIR = "/home/mkperez/scratch/speechbrain/AphasiaBank/statistical_analysis/results/multi/GPT-Oracle"
    # GPT_eval(GPT_DIR,'oracle')