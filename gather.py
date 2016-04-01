#!/usr/bin/env python3

import sys

mappy = {}

outputs = []

for line in sys.stdin:
    parts = line.split('--')
    if mappy.get(parts[1]) is None:
        mappy[parts[1]] = []
    mappy[parts[1]].append( (int(parts[2]), int(parts[3])) )

for key, value in mappy.items():
    withoutprep = sum( [x for x,_ in value] ) / len(value)
    withprep = sum( [x for _,x in value] ) / len(value)
    outputs.append(key + " " + str(int(withoutprep)).ljust(10) + " " + str(int(withprep)   ).ljust(10) )

for string in sorted(outputs):
    print(string)
