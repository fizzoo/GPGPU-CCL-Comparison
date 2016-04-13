# GPGPU-CCL-Comparison

A comparison of a few algorithms that perform connected components labeling.
Each algorithm is implemented as a "Strategy" and is tested and timed at the same time as the rest.

## Usage
The program regards each argument passed to it as an image to be labeled.
It automatically performs some thresholding and labels the resulting data, after which it outputs the result into out/.
Timings are written to stdout, while some informative information (and debug info, when applicable) is output to stderr.

Two python scripts are provided for easy handling of the data.
  * otsu.py performs a more sophisticated thresholding.
  * gather.py summarizes the results from stdout of the regular program.
