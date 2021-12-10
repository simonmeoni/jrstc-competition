FROM pytorch/pytorch:latest
ARG KAGGLE_USERNAME
ARG KAGGLE_KEY
ENV TOKENIZERS_PARALLELISM=False
ENV KAGGLE_USERNAME=$KAGGLE_USERNAME
ENV KAGGLE_KEY=$KAGGLE_KEY
COPY requirements.txt ./
RUN apt update
RUN apt install -y git screen unzip
RUN pip install --no-cache-dir -r requirements.txt && rm requirements.txt
RUN pip install --no-cache-dir kaggle
RUN mkdir "jrstc-competition"
COPY . .
RUN cd jrstc-competition
RUN bash/dataset.sh