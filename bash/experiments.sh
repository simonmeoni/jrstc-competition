# Baseline
python run.py -m experiment=baseline.yaml ++trainer.gpus=1 current_fold=0,1,2,3,4
python run.py -m experiment=more_batch.yaml ++trainer.gpus=1 current_fold=0,1,2,3,4
python run.py -m experiment=cosine_lr.yaml ++trainer.gpus=1 current_fold=0,1,2,3,4