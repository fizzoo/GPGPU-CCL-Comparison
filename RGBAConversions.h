#ifndef RGBACONVERSIONS_H
#define RGBACONVERSIONS_H

#include "defines.h"

/**
 * A 1 if all of r, g, and b is above 100
 */
bool rgb_above_100(unsigned char r, unsigned char g, unsigned char b,
                   unsigned char);

/**
 * Returns 255,255,255,255 whenever input is nonzero.
 */
RGBA max_if_nonzero(LABELTYPE in);

#endif /* end of include guard: RGBACONVERSIONS_H */
