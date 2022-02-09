python run.py -m experiment=1_freezing_vote ++trainer.gpus=1 current_fold=0,1,2,3,4 experiment_group=freezing
python run.py -m experiment=1_freezing_vote ++trainer.gpus=1 current_fold=0,1,2,3,4 experiment_group=llrd model.freeze=false model.remove_dropout=true
