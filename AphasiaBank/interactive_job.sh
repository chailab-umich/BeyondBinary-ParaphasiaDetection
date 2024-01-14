#!/bin/bash                                                                                  
srun \
 --job-name=interactive \
 --mail-user=mkperez@umich.edu \
 --mail-type=NONE \
 --ntasks=1 \
 --cpus-per-task=4 \
 --gres=gpu:1 \
 --mem-per-cpu=11G \
 --time=10-00:00:00 \
 --partition=spgpu \
 --account=emilykmp1 \
 --pty bash