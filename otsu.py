#!/usr/bin/env python3
"""
Script for performing otsu's method (smart thresholding).
Treats all arguments as files to be treated and writes output to a file of the same name + ".png".
"""

from sys import argv

from skimage.filters import threshold_otsu
from skimage.io import imsave, imread
from skimage.color import rgb2grey

for index in range(1, len(argv)):
    filename = argv[index]
    colorimage = imread(filename)
    image = rgb2grey(colorimage)
    thresh = threshold_otsu(image)
    binary = image > thresh
    maxed = binary * 255
    imsave(filename + ".png", maxed)
