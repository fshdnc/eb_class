# Automatic Essay Scoring

Automatic essay scoring using BERT embeddings

## Getting started

You can install the dependencies using Poetry:

    $ poetry install

Or else use the Docker image / `Dockerfile`. The Docker image has been tested
mainly under Singularity.

## Training/evaluation

Training:

    $ python -m finnessayscore.train \
      --epochs 1 \
      --batch_size=5 \
      --model_type whole_essay \
      --jsons /path/to/data.json

A confusion matrix and scores on the development set are printed at the end of
training.

Results on tensorboard

    $ tensorboard --logdir lightning_logs/ --port <port_number>


## Explainability

TODO
