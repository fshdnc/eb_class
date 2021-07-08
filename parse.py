#!/usr/bin/env python3

import sys
import json
import tqdm
import argparse

sys.path.append("/home/jmnybl/git_checkout/Turku-neural-parser-pipeline-modularize")
from tnparser.pipeline import read_pipelines, Pipeline


ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC = range(10)

# GPU
import types
extra_args = types.SimpleNamespace()
extra_args.__dict__["udify_mod.device"] = "0" #simulates someone giving a --device 0 parameter to Udify
extra_args.__dict__["lemmatizer_mod.device"] = "0"

available_pipelines = read_pipelines("models_fi_tdt_v2.7/pipelines.yaml") # {pipeline_name -> its steps}
p = Pipeline(available_pipelines["parse_plaintext"]) # launch the pipeline from the steps

def parse(txt):

    txt_parsed = p.parse(txt) # txt be a paragraph
    sents = []
    tokens = []
    lemmas = []
    txt_parsed = txt_parsed.split("\n\n")
    for sent_parsed in txt_parsed:
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
                continue # multiword token or multitoken word
            #tokens.append(cols[FORM])
            lemma_sent.append(cols[LEMMA])
        lemmas.append(" ".join(lemma_sent))
    lemmas = [l for l in lemmas if l] # remove empty

    return lemmas, sents

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, required=True, help="Essay json file")
    args = parser.parse_args()

    with open(args.json, "rt", encoding="utf-8") as f:
        data = json.load(f)

    for essay in data:
        lemmas, sents = parse(" ".join(essay["essay"]))
        essay["lemma"] = lemmas
        essay["sents"] = sents

    with open(args.json[:-5]+"-parsed.json", "wt") as f:
        json.dump(data, f, indent=4, ensure_ascii=False, sort_keys=True)
