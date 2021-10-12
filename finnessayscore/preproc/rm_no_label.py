import argparse

from .utils import filter_json


def remove_no_label(all_data):
    print("Removing data points without label(s)...")
    new = []
    for data in all_data:
        add = True
        for k in data.keys():
            if k.startswith("lab_"):
                if not data[k]:
                    add = False
        if add:
            new.append(data)
    return new


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('inp')
    parser.add_argument('out')
    args = parser.parse_args()

    filter_json(args.inp, args.outp, remove_no_label)


if __name__ == "__main__":
    main()
