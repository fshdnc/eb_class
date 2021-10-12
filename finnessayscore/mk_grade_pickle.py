import argparse
from pathlib import Path
import pickle
from .grade_scale import mk_grade


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('grade_scale', choices=["fivehigh", "outof20"])
    parser.add_argument('out', type=Path)

    args = parser.parse_args()
    res = mk_grade(args.grade_scale)

    with open(args.out, "wb") as outf:
        pickle.dump(res, outf)


if __name__ == "__main__":
    main()
