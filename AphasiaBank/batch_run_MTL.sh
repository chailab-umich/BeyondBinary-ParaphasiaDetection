#!/bin/sh
#SBATCH --partition spgpu
#SBATCH --account=emilykmp1
#SBATCH --job-name=FT_mtl_script
#SBATCH --output=batch_logs/FT_mtl_script.log
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
# python train_MTL.py hparams/MTL_proto.yml

# Scripts train
python Scripts_LOSO_MTL.py
#--output=ISresults/Transcription_Scripts/S2S-hubert-Transformer-500/sbatch.log