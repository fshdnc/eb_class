import argparse
from pathlib import Path
import pickle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('grade_scale', choices=["fivehigh", "outof20"])
    parser.add_argument('out', type=Path)

    args = parser.parse_args()

    if args.grade_scale == "fivehigh":
        res = {"lab_grade": [str(grade) for grade in range(1, 6)]}
    elif args.grade_scale == "outof20":
        res = {"lab_grade": [str(grade) for grade in range(21)]}

    with open(args.out, "wb") as outf:
        pickle.dump(res, outf)


if __name__ == "__main__":
    main()
