python run.py -m experiment=baseline.yaml ++trainer.gpus=1 current_fold=0,1,2,3,4
python run.py -m experiment=more_batch.yaml ++trainer.gpus=1 current_fold=0,1,2,3,4
python run.py -m experiment=cosine_lr.yaml ++trainer.gpus=1 current_fold=0,1,2,3,4
python run.py -m experiment=synthetic_data.yaml ++trainer.gpus=1 current_fold=0
python run.py -m experiment=less_data.yaml ++trainer.gpus=1 current_fold=0,1,2,3,4
python run.py -m experiment=concatenate_heads.yaml ++trainer.gpus=1 current_fold=0,1,2,3,4
python run.py -m experiment=jigsaw_toxic_comment_dataset.yaml ++trainer.gpus=1 current_fold=0,1,2,3,4 model.concatenate_heads=false,true
python run.py -m experiment=mlp.yaml ++trainer.gpus=1 current_fold=0,1,2,3,4 model.concatenate_heads=false,true
python run.py -m hparams_search=jigsaw_optuna experiment=mlp hydra.sweeper.n_trials=30
