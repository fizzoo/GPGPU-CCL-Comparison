#!/usr/bin/env python3

import sys
from statistics import stdev, mean


mappy = {}

outputs = []

for line in sys.stdin:
    parts = line.split('--')
    if mappy.get(parts[1]) is None:
        mappy[parts[1]] = []
    mappy[parts[1]].append( (int(parts[2]), int(parts[3])) )

for key, value in mappy.items():
    withoutprep = [x for x,_ in value]
    withprep = [x for _,x in value]

    mean_wo = int(mean( withoutprep ))
    mean_w  = int(mean( withprep ))

    dev_wo = int(stdev( withoutprep ))
    dev_w  = int(stdev( withprep ))

    string_meandiff = str(int(mean( [x-y for x,y in value] )))
    
    string_wo = str(mean_wo).ljust(10) + " ( " + str(dev_wo).ljust(10) + ")"
    string_w  = str(mean_w).ljust(10)  + " ( " + str(dev_w).ljust(10) + ")"
    
    outputs.append(key + " " + string_wo + " " + string_w + " {{ " + string_meandiff)

for string in sorted(outputs):
    print(string)
