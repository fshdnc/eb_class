#!/usr/bin/env python3
import sys
import numpy as np
import json

from argparse import ArgumentParser
from common import encode, process_sentences, load_ner_model
from config import DEFAULT_BATCH_SIZE


def mk_argparser():
    argparser = ArgumentParser()
    argparser.add_argument(
        '--batch_size', type=int, default=DEFAULT_BATCH_SIZE,
        help='Batch size for training'
    )
    argparser.add_argument(
        '--ner_model_dir', default="/usr/src/app/combined-ext-model",
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


def read_json(input_file):
    with open(input_file, "rt", encoding="utf-8") as f:
        data = json.load(f)

    essay_mask = []
    words = []
    labels = []
    for essay_idx, essay in enumerate(data):
        for sent in essay["surfs"]:
            essay_mask.append(essay_idx)
            words.append(sent)
            labels.append(["O"] * len(sent))
    return essay_mask, words, labels


def filter_json(input_file, output_file, indexed_labels):
    with open(input_file, "rt", encoding="utf-8") as f:
        data = json.load(f)

    for essay_idx, labels in indexed_labels.items():
        data[essay_idx]["ner"] = labels

    with open(output_file, "wt", encoding="utf-8") as f:
        json.dump(data, f)


def main():
    parser = mk_argparser()
    args = parser.parse_args()

    ner_model, tokenizer, labels, config = load_ner_model(args.ner_model_dir)
    max_seq_len = config['max_seq_length']

    label_map = {t: i for i, t in enumerate(labels)}
    inv_label_map = {v: k for k, v in label_map.items()}

    mask, all_words, dummy_labels = read_json(args.input_file)
    sents_data = process_sentences(
        all_words,
        dummy_labels,
        tokenizer,
        max_seq_len
    )

    test_x = encode(sents_data.combined_tokens, tokenizer, max_seq_len)
    probs = ner_model.predict(test_x, batch_size=args.batch_size)
    preds = np.argmax(probs, axis=-1)
    indexed_labels = {}
    zipped = zip(mask, preds, sents_data.words, sents_data.tokens)
    lengths = sents_data.lengths
    length_idx = 0
    for essay_idx, pred, words, tokens in zipped:
        pred_slice = pred[1: len(tokens) + 1]
        pred_label_tok = [inv_label_map[t] for t in pred_slice]
        pred_label_word = []
        idx = 0
        for word in words:
            tok = tokens[idx]
            if not (word.startswith(tok) or tok == '[UNK]'):
                print('tokenization mismatch: "{}" vs "{}"'.format(
                    word, tok), file=sys.stderr)
                sys.exit(-1)
            pred_label_word.append(pred_label_tok[idx])
            idx += lengths[length_idx]
            length_idx += 1
        indexed_labels.setdefault(essay_idx, []).append(pred_label_word)
    filter_json(args.input_file, args.output_file, indexed_labels)


if __name__ == "__main__":
    main()
