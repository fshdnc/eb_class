import argparse
from functools import partial

from .utils import filter_json


def clean_data(level, all_data):
    if level == "sentences":
        keep = ["essay", "lemma", "sents"]
    else:
        keep = ["essay"]
    for data in all_data:
        remove = []
        for k in data.keys():
            if k in keep or k.startswith("lab_"):
                pass
            else:
                remove.append(k)
        for k in remove:
            del data[k]
    return all_data


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('inp')
    parser.add_argument('out')
    parser.add_argument('--level', choices=['sentences', 'essays'])
    args = parser.parse_args()

    filter_json(args.inp, args.out, partial(clean_data, args.level))


if __name__ == "__main__":
    main()
