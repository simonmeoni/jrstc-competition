docker:
	bash/docker.sh
get_dataset:
	bash/get_dataset.sh
test:
	bash/fast_dev_run.sh $(experiments)
upload:
	bash/upload_monster_ensemble.sh
experiment:
	nohup bash/experiments/$(experiments).sh &
