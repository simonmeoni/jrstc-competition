docker build . -t  smeoni1/jrstc-competition:latest --build-arg KAGGLE_USERNAME="$KAGGLE_USERNAME" \
                   --build-arg KAGGLE_KEY="$KAGGLE_KEY"
docker push smeoni1/jrstc-competition:latest