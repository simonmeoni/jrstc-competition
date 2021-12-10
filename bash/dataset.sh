kaggle datasets download -d simonmeoni/jrstcsyntheticdata -p data --unzip
kaggle competitions download -c jigsaw-toxic-severity-rating -p data
unzip -o data/jigsaw-toxic-severity-rating.zip -d data/jigsaw-toxic-severity-rating && rm data/*.zip