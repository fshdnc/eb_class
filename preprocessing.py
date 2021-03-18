#!/usr/bin/env python3

import math

def seg_by_char_index(text, n):
    """
    Simple split by character index
    """
    pcs = math.ceil(len(text)/n)
    frags = [text[i*n:(i+1)*n] for i in range(pcs)]
    return frags
