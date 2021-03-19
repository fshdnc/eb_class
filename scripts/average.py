#!/usr/bin/env python3

import sys
import numpy as np

if __name__=="__main__":
    numbers = sys.stdin
    numbers = [float(number.strip()) for number in numbers]
    print("mean\tmin\tmax")
    print("{}\t{}\t{}".format(np.mean(numbers), np.min(numbers), np.max(numbers)))
