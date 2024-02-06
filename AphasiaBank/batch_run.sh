#!/bin/sh
#SBATCH --partition spgpu
#SBATCH --account=emilykmp1
#SBATCH --job-name=bin_pr_trans
#SBATCH --output=batch_logs/bin_pr_trans.txt
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --mem=45G

cd /home/mkperez/scratch/speechbrain/AphasiaBank
source /home/mkperez/.pyenv/versions/miniconda3-latest/etc/profile.d/conda.sh
set -x
conda activate sb



## Proto train
# python train_Transcription.py hparams/SSL_transcription_proto.yml
# --output=ISresults/Transcription_Proto/S2S-hubert-Transformer-500/sbatch.log

# Scripts train
# python Scripts_LOSO_Transcription.py
# --output=ISresults/Transcription_Scripts/S2S-hubert-Transformer-500/sbatch.log



# binary
python train_Transcription_binary.py hparams/binary/SSL_transcription_proto.yml
# python Scripts_LOSO_Transcription.py