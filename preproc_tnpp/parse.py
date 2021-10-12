#!/usr/bin/env python3

import types
import json
import argparse

from tnparser.pipeline import read_pipelines, Pipeline


DEFAULT_PIPELINES_YAML = "/usr/src/app/models_fi_tdt_dia/pipelines.yaml"
ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC = range(10)


def parse(pipeline, txt):
    # txt be a paragraph
    txt_parsed = pipeline.parse(txt)
    sents = []
    lemma_sents = []
    surfs = []
    upos = []
    txt_parsed = txt_parsed.split("\n\n")
    for sent_parsed in txt_parsed:
        surf_sent = []
        upos_sent = []
        lemma_sent = []
        for line in sent_parsed.split("\n"):
            line = line.strip()
            if not line:
                continue
            if line.startswith("# text = "):
                sents.append(line[9:])
                continue
            elif line.startswith("#"):
                continue
            cols = line.split("\t")
            if "-" in cols[ID]:
                continue  # multiword token or multitoken word
            lemma_sent.append(cols[LEMMA])
            surf_sent.append(cols[FORM])
            upos_sent.append(cols[UPOS])
        lemma_sents.append(" ".join(lemma_sent))
        surfs.append(surf_sent)
        upos.append(upos_sent)
    lemma_sents = [lemma for lemma in lemma_sents if lemma]  # remove empty

    return sents, lemma_sents, surfs, upos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true", required=False, default=False, help="Use GPU")
    parser.add_argument("input_json", type=str, help="Essay json file")
    parser.add_argument("output_json", type=str, help="Processed essay json file")
    args = parser.parse_args()

    extra_args = types.SimpleNamespace()
    if args.gpu:
        # simulates someone giving a --device 0 parameter to Udify
        extra_args.__dict__["udify_mod.device"] = "0"
        extra_args.__dict__["lemmatizer_mod.device"] = "0"
    else:
        extra_args.__dict__["udify_mod.device"] = "-1"
        extra_args.__dict__["lemmatizer_mod.device"] = "-1"
    # Unclear whether this is needed (is the lemmatiser compatible with the GPU
    # extra_args.__dict__["lemmatizer_mod.device"] = "-1"

    # {pipeline_name -> its steps}
    available_pipelines = read_pipelines(DEFAULT_PIPELINES_YAML)
    # launch the pipeline from the steps
    pipeline = Pipeline(available_pipelines["parse_plaintext"], extra_args=extra_args)

    with open(args.input_json, "rt", encoding="utf-8") as f:
        data = json.load(f)

    for essay in data:
        sents, lemma_sents, surfs, upos = parse(
            pipeline,
            " ".join(essay["essay"])
        )

        essay["lemma"] = lemma_sents
        essay["sents"] = sents
        essay["surfs"] = surfs
        essay["upos"] = upos

    with open(args.output_json, "wt") as f:
        json.dump(data, f, indent=4, ensure_ascii=False, sort_keys=True)


if __name__ == "__main__":
    main()
