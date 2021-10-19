import argparse
import json
from pathlib import Path
from os import makedirs
from os.path import join as pjoin
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('json_in', type=Path)
    parser.add_argument('splits_out', type=Path)

    args = parser.parse_args()
    with open(args.json_in) as inf:
        data = json.load(inf)

    random.shuffle(data)

    dev_start = int(len(data) * 0.8)
    test_start = int(len(data) * 0.9)
    train = data[:dev_start]
    dev = data[dev_start:test_start]
    test = data[test_start:]

    makedirs(args.splits_out, exist_ok=True)
    splits = ((train, "train.json"), (dev, "val.json"), (test, "test.json"))
    for data_split, fn in splits:
        with open(pjoin(args.splits_out, fn), "w") as outf:
            json.dump(data_split, outf, indent=4, ensure_ascii=False, sort_keys=True)


if __name__ == "__main__":
    main()
