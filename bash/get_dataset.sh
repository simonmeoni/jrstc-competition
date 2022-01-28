# Dataset
kaggle datasets download -d simonmeoni/jigsaw-classification-voting-cleaning -p data/jigsaw-classification-voting-cleaning --unzip
kaggle datasets download -d simonmeoni/pseudo-jigsaw-severity -p data/pseudo-jigsaw-severity --unzip
kaggle datasets download -d vaby667/toxictask -p data/toxictask --unzip
kaggle datasets download -d julian3833/jigsaw-unintended-bias-in-toxicity-classification -p data/jigsaw-unintended-bias-in-toxicity-classification --unzip

# Competition Dataset
kaggle competitions download -c jigsaw-toxic-severity-rating -p data
kaggle competitions download -c jigsaw-toxic-comment-classification-challenge -p data
unzip -o data/jigsaw-toxic-severity-rating.zip -d data
unzip -o data/jigsaw-toxic-comment-classification-challenge.zip -d data/jigsaw-toxic-comment-classification-challenge
unzip -o data/jigsaw-toxic-severity-rating.zip -d data/jigsaw-toxic-severity-rating
unzip -o 'data/jigsaw-toxic-comment-classification-challenge/*.zip' -d data/jigsaw-toxic-comment-classification-challenge
rm -r data/jigsaw-toxic-comment-classification-challenge/*.zip
rm data/*.zip
rm data/*.csv