include .env
export

docker:
	docker build . -t  smeoni1/jrstc-competition:latest \
				   --build-arg KAGGLE_USERNAME=$(KAGGLE_USERNAME) \
                   --build-arg KAGGLE_KEY=$(KAGGLE_KEY)
	docker push smeoni1/jrstc-competition:latest
get_dataset:
	bash/get_dataset.sh
create_synthetic_dataset:
	bash/get_dataset.sh
test:
	bash/fast_dev_run.sh $(experiment)
upload:
	bash/upload_monster_ensemble.sh
experiment:
	 nohup python run.py -m experiment=$(experiment) ++trainer.gpus=1 &
upload-checkpoints:
	python upload_checkpoints.py \
        --wandb_project simonmeoni/jrstc-competition \
        --kaggle_dataset jigsawcheckpoints \
        --checkpoints_path models/checkpoints \
        --wandb_groups $(wandb_groups)
upload-ensemble-monster:
	python upload_checkpoints.py \
        --wandb_project simonmeoni/jrstc-competition \
        --kaggle_dataset jigsawcheckpoints \
        --checkpoints_path models/checkpoints \
        --wandb_groups concat_regression/microsoft/deberta-v3-large/pseudo-labelling\
                       concat_regression/xlnet-large-cased/pseudo-labelling\
                       concat_regression/google/electra-large-discriminator/pseudo-labelling\
                       concat_regression/roberta-large/pseudo-labelling\
                       concat_regression/microsoft/deberta-v3-base/pseudo-labelling\
                       concat_regression/unitary/unbiased-toxic-roberta/pseudo-labelling
