# vinai/bertweet-base
python run.py -m experiment=kfold_tranformers transformer_model=vinai/bertweet-base architecture_model=concat_regression ++trainer.gpus=1 model.loss=aux_loss model.lr=5e-5 datamodule.train_batch_size=64 model.remove_dropout=false current_fold=0,1,2,3,4 #68.95
python run.py -m experiment=kfold_tranformers transformer_model=vinai/bertweet-base architecture_model=simple_mlp ++trainer.gpus=1 model.loss=margin_loss model.lr=2e-4 datamodule.train_batch_size=128 model.remove_dropout=true current_fold=0,1,2,3,4 # 69.19
python run.py -m experiment=kfold_tranformers transformer_model=vinai/bertweet-base architecture_model=tfidf_transformer ++trainer.gpus=1 model.loss=margin_loss model.lr=8e-5 datamodule.train_batch_size=128 model.remove_dropout=true current_fold=0,1,2,3,4 # 68.95

# unitary/toxic-bert
python run.py -m experiment=kfold_tranformers transformer_model=unitary/toxic-bert architecture_model=simple_regression ++trainer.gpus=1 model.loss=margin_loss model.lr=5e-5 datamodule.train_batch_size=64 model.remove_dropout=true current_fold=0,1,2,3,4 #68.91
python run.py -m experiment=kfold_tranformers transformer_model=unitary/toxic-bert architecture_model=simple_mlp ++trainer.gpus=1 model.loss=margin_loss model.lr=8e-5 datamodule.train_batch_size=128 model.remove_dropout=true current_fold=0,1,2,3,4 # 68.95
python run.py -m experiment=kfold_tranformers transformer_model=unitary/toxic-bert architecture_model=tfidf_transformer ++trainer.gpus=1 model.loss=margin_loss model.lr=5e-5 datamodule.train_batch_size=64 model.remove_dropout=true current_fold=0,1,2,3,4 # 68.99

# unitary/unbiased-toxic-roberta
python run.py -m experiment=kfold_tranformers transformer_model=unitary/unbiased-toxic-roberta architecture_model=simple_regression ++trainer.gpus=1 model.loss=margin_loss model.lr=5e-5 datamodule.train_batch_size=128 model.remove_dropout=true current_fold=0,1,2,3,4 #69.15
python run.py -m experiment=kfold_tranformers transformer_model=unitary/unbiased-toxic-roberta architecture_model=concat_regression ++trainer.gpus=1 model.loss=margin_loss model.lr=5e-5 datamodule.train_batch_size=128 model.remove_dropout=true current_fold=0,1,2,3,4 # 69.09
python run.py -m experiment=kfold_tranformers transformer_model=unitary/unbiased-toxic-roberta architecture_model=simple_mlp ++trainer.gpus=1 model.loss=margin_loss model.lr=1e-4 datamodule.train_batch_size=128 model.remove_dropout=true current_fold=0,1,2,3,4 # 68.98
python run.py -m experiment=kfold_tranformers transformer_model=unitary/unbiased-toxic-roberta architecture_model=tfidf_transformer ++trainer.gpus=1 model.loss=margin_loss model.lr=5e-5 datamodule.train_batch_size=64 model.remove_dropout=true current_fold=0,1,2,3,4 # 68.99

# google/electra-base-discriminator
python run.py -m experiment=kfold_tranformers transformer_model=google/electra-base-discriminator architecture_model=simple_mlp ++trainer.gpus=1 model.loss=margin_loss model.lr=1e-4 datamodule.train_batch_size=128 model.remove_dropout=true current_fold=0,1,2,3,4 #68.69

# xlnet-base-cased
python run.py -m experiment=kfold_tranformers transformer_model=xlnet-base-cased architecture_model=simple_regression ++trainer.gpus=1 model.loss=margin_loss model.lr=1e-4 datamodule.train_batch_size=64 model.remove_dropout=true current_fold=0,1,2,3,4 #68.69

# TF-IDF regression
python run.py -m experiment=kfold_tranformers architecture_model=tfidf_regression ++trainer.gpus=1 model.loss=margin_loss model.lr=4e-4 datamodule.train_batch_size=16 model.remove_dropout=true current_fold=0,1,2,3,4 #68.69

# TF-IDF mlp
python run.py -m experiment=kfold_tranformers architecture_model=tfidf_mlp ++trainer.gpus=1 model.loss=margin_loss model.lr=1e-4 datamodule.train_batch_size=16 model.remove_dropout=true current_fold=0,1,2,3,4 #67.1
