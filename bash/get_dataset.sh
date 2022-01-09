mkdir data/jigsaw-classification-voting-cleaning
kaggle datasets download -d simonmeoni/jrstcsyntheticdata -p data --unzip
kaggle datasets download -d simonmeoni/jigsaw-classification-voting-cleaning -p data/jigsaw-classification-voting-cleaning --unzip
kaggle competitions download -c jigsaw-toxic-severity-rating -p data
unzip -o data/jigsaw-toxic-severity-rating.zip -d data/jigsaw-toxic-severity-rating && rm data/*.zip
