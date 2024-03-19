'''
Compute evaluations for a given directory
For GPT or SB models
'''
import os
from evaluation import *


if __name__ == '__main__':

    ## GPT - 3.5 ##
    # print("\nASR + GPT-3.5")
    # GPT_DIR = "/home/mkperez/scratch/speechbrain/AphasiaBank/statistical_analysis/results/multi/GPT"
    # GPT_eval(GPT_DIR,'asr')
    # print("\n Oracle + GPT-3.5")
    # GPT_DIR = "/home/mkperez/scratch/speechbrain/AphasiaBank/statistical_analysis/results/multi/GPT-Oracle"
    # GPT_eval(GPT_DIR,'oracle')

    ## GPT - 4 ##
    # print("\nASR + GPT-4")
    # GPT_DIR = "/home/mkperez/scratch/speechbrain/AphasiaBank/statistical_analysis/results/multi/GPT4"
    # GPT_eval(GPT_DIR,'asr')
    # print("\nORACLE + GPT-4")
    # GPT_DIR = "/home/mkperez/scratch/speechbrain/AphasiaBank/statistical_analysis/results/multi/GPT4-Oracle"
    # GPT_eval(GPT_DIR,'oracle')


    # print("\nMTL")
    # MODEL_DIR = "/home/mkperez/scratch/speechbrain/AphasiaBank/ISresults/full_FT_MTL_Scripts/MTL-weighted_para/reduce-w_asr_w-0.6_S2S-hubert-Transformer-500"
    # para_eval(MODEL_DIR, 'mtl')

    # ## SS
    print("\nSS")
    # MODEL_DIR = "/home/mkperez/scratch/speechbrain/AphasiaBank/ISresults/full_FT_Transcription_Scripts/balanced_para_S2S-hubert-Transformer-500"
    MODEL_DIR = "/home/mkperez/scratch/speechbrain/AphasiaBank/ISresults/full_FT_Transcription_Scripts/bpe_ES_S2S-hubert-Transformer-500"
    para_eval(MODEL_DIR, 'single_seq')
