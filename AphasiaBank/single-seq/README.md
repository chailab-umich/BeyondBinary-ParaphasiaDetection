# Recipe for single-sequence model

## Pretrain with Protocol Dataset
```
python train_single-seq.py hparams/pretrain_Proto.yml
```

## Finetune with Scripts Dataset
```run_Scripts_FT.py``` performs finetuning and evaluation across all folds of the Scripts dataset. It then aggregates and computes final evaluation metrics across all folds.
```
python run_Scripts_FT.py
```