# Automatic Essay Scoring

Automatic essay scoring using BERT embeddings

## Getting started

You can install the dependencies using Poetry:

    $ poetry install

Or else use the Docker image / `Dockerfile`. The Docker image has been tested
mainly under Singularity.

## Data format

Input data is a JSON array of objects [{...}, {...}] with one object per essay.
It should have at least the keys:

 * "essay": an array of strings for each line of the essay e.g. ["Lorem ipsum --", "dolar"]
 * "lab_grade": the grade as an string e.g. "3"

## Preprocessing

Convert TKP exam data to the JSON format by using:

    $ python -m finnessayscore.process_tkp tkp.xls tkp.json

Some models need parsed data. In this case, further preprocessing should be
done like so:

    $ python -m finnessayscore.parse example.json example_parse.json

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
