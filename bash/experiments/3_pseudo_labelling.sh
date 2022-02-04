python run.py -m experiment=3_pseudo_labelling ++trainer.gpus=1 current_fold=0,1,2,3,4 transformer_model=microsoft/deberta-v3-base
python run.py -m experiment=3_pseudo_labelling ++trainer.gpus=1 current_fold=0,1,2,3,4 transformer_model=microsoft/deberta-v3-large model.hidden_size=1024
python run.py -m experiment=3_pseudo_labelling ++trainer.gpus=1 current_fold=0,1,2,3,4 transformer_model=unitary/unbiased-toxic-roberta
python run.py -m experiment=3_pseudo_labelling ++trainer.gpus=1 current_fold=0,1,2,3,4 transformer_model=roberta-large model.hidden_size=1024
python run.py -m experiment=3_pseudo_labelling ++trainer.gpus=1 current_fold=0,1,2,3,4 transformer_model=google/electra-large-discriminator model.hidden_size=1024
python run.py -m experiment=3_pseudo_labelling ++trainer.gpus=1 current_fold=0,1,2,3,4 transformer_model=xlnet-large-cased model.hidden_size=1024 model.llrd=False
