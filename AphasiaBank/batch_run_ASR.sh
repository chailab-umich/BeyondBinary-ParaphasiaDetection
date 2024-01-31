#!/bin/sh
#SBATCH --partition spgpu
#SBATCH --account=emilykmp1
#SBATCH --job-name=asr-only
#SBATCH --output=ISresults/ASR-only_Scripts/S2S-hubert-Transformer-500/sbatch.log
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --mem=45G

cd /home/mkperez/scratch/speechbrain/AphasiaBank
source /home/mkperez/.pyenv/versions/miniconda3-latest/etc/profile.d/conda.sh
set -x
conda activate sb


## Proto
# python train_ASR-only.py hparams/ASR_only.yml
# --output=ISresults/ASR-only_Proto/S2S-hubert-Transformer-500/sbatch.log


## Scripts
python Scripts_LOSO_ASR-only.py
# --output=ISresults/ASR-only_Scripts/S2S-hubert-Transformer-500/sbatch.log