#!/usr/bin/env python3
"""
Script for summarizing the results from the comparison output.
Reads from stdin, gathers the data and writes the result on stdout.
"""

import sys
from statistics import stdev, mean

def stringypad(x, pad = 0):
    return str(int(x)).ljust(pad)

mappy = {}

outputs = []

for line in sys.stdin:
    parts = line.split('--')
    if mappy.get(parts[1]) is None:
        mappy[parts[1]] = []
    mappy[parts[1]].append( (int(parts[2]), int(parts[3])) )

for key, value in mappy.items():
    meany = mean( [x for x,_ in value] )
    dev  = stdev( [x for x,_ in value] )
    meandiff = mean( [y-x for x,y in value] )
    
    outputs.append(key + " " + stringypad(meany, 10) + " +- " + stringypad(dev, 10) +  " diff: " + stringypad(meandiff))

for string in sorted(outputs):
    print(string)
