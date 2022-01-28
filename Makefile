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
	bash/fast_dev_run.sh $(experiments)
upload:
	bash/upload_monster_ensemble.sh
experiment:
	nohup bash/experiments/$(experiments).sh &
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
        --wandb_groups simple_regression-unitary/unbiased-toxic-roberta, \
                       simple_regression-unitary/toxic-bert, \
                       simple_regression-xlnet-base-cased, \
                       concat_regression-vinai/bertweet-base, \
                       concat_regression-unitary/toxic-bert, \
                       concat_regression-unitary/unbiased-toxic-roberta, \
                       simple_mlp-google/electra-base-discriminator, \
                       simple_mlp-vinai/bertweet-base, \
                       simple_mlp-unitary/toxic-bert, \
                       simple_mlp-unitary/unbiased-toxic-roberta