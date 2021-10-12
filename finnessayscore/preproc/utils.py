import json
import os
from os.path import join as pjoin


def filter_json_file(inp, out, filter_func):
    with open(inp, "r") as inf:
        data = json.load(inf)
    data = filter_func(data)
    with open(out, "w") as outf:
        json.dump(data, outf)


def filter_json(inp, out, filter_func):
    if os.path.isdir(inp):
        os.makedirs(out, exist_ok=True)
        for fn in os.listdir(inp):
            if not fn.lower().endswith(".json"):
                continue
            filter_json_file(pjoin(inp, fn), pjoin(out, fn), filter_func)
    else:
        filter_json_file(inp, out, filter_func)
