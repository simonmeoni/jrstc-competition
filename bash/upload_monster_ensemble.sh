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