#!/usr/bin/env python3
import numpy as np
import json
from pprint import pprint

from argparse import ArgumentParser
from common import encode, read_conll, process_sentences, load_ner_model
from config import DEFAULT_BATCH_SIZE


def mk_argparser():
    argparser = ArgumentParser()
    argparser.add_argument(
        '--batch_size', type=int, default=DEFAULT_BATCH_SIZE,
        help='Batch size for training'
    )
    argparser.add_argument(
        '--ner_model_dir', default=None,
        help='Trained NER model directory'
    )
    argparser.add_argument(
        'input_file',
        help='File to read predicted inputs from'
    )
    argparser.add_argument(
        'output_file',
        help='File to write predicted outputs to'
    )
    return argparser


def read_json(args):
    with open(args.json, "rt", encoding="utf-8") as f:
        data = json.load(f)

    essay_mask = []
    words = []
    labels = []
    for essay_idx, essay in enumerate(data):
        essay_mask.append(essay_idx)
        for sent in essay["sents"]:
            words.append(sent)
            for tok in sent:
                labels.append("O")
    return essay_mask, words, labels


def filter_json(args):
    with open(args.json, "rt", encoding="utf-8") as f:
        data = json.load(f)


def main():
    parser = mk_argparser()
    args = parser.parse_args()

    ner_model, tokenizer, labels, config = load_ner_model(args.ner_model_dir)
    max_seq_len = config['max_seq_length']

    label_map = {t: i for i, t in enumerate(labels)}
    inv_label_map = {v: k for k, v in label_map.items()}

    mask, words, labels = read_json(args.input_file)
    sents_data = process_sentences(
        words,
        labels,
        tokenizer,
        max_seq_len
    )

    test_x = encode(sents_data.combined_tokens, tokenizer, max_seq_len)

    probs = ner_model.predict(test_x, batch_size=args.batch_size)

    pred_labels = []
    preds = np.argmax(probs, axis=-1)
    for i, pred in enumerate(preds):
        pred_labels.append([inv_label_map[t] for t in
                            pred[1:len(sents_data.tokens[i])+1]])

    pprint(sents_data.words)
    pprint(sents_data.lengths)
    pprint(sents_data.tokens)
    pprint(sents_data.labels)
    pprint(pred_labels)


if __name__ == "__main__":
    main()
