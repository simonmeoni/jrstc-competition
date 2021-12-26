# With Transformers arch
declare -a archs=("simple_regression" "concat_regression" "simple_mlp" "concat_mlp" "tfidf_transformer")
# shellcheck disable=SC2054
declare -a models=(
"roberta-base" "unitary/toxic-bert" "unitary/unbiased-toxic-roberta"
"vinai/bertweet-base"  "microsoft/deberta-v3-base"
"google/electra-base-discriminator")

for model in "${models[@]}"
do
   for arch in "${archs[@]}"
  do
     echo "$arch"@"$model"
     python run.py -m hparams_search=jigsaw_transformers_optuna experiment=hparams_tranformers transformer_model="$model" architecture_model="$arch" ++trainer.gpus=1 hydra.sweeper.n_trials=12
  done
done
# TFIDF arch
declare -a archs=("tfidf_regression" "tfidf_mlp")

for arch in "${archs[@]}"; do
  echo "$arch"@tfidf
  python run.py -m experiment=hparams_tfidf model.architecture="$arch" ++trainer.gpus=1 hydra.sweeper.n_trials=30
done
