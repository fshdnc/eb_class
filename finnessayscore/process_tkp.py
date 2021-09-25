import argparse
import pandas as pd
import json
from pathlib import Path
from pattern.web import plaintext


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('inf', type=Path)
    parser.add_argument('outf', type=Path)

    args = parser.parse_args()

    sheets = pd.read_excel(args.inf, sheet_name=None)
    output = []
    is_html = False
    for course, df in sheets.items():
        for _, points, answer in df.itertuples():
            if is_html:
                answer = plaintext(answer)
            output.append({
                "essay": answer.split("\n"),
                "lab_grade": str(points),
            })
        is_html = True
    with open(args.outf, "w") as outf:
        json.dump(output, outf)


if __name__ == "__main__":
    main()
