#include "RGBAConversions.h"

bool rgb_above_100(unsigned char r, unsigned char g, unsigned char b,
                   unsigned char) {
  if (r > 100 && g > 100 && b > 100) {
    return true;
  }
  return false;
}

RGBA max_if_nonzero(LABELTYPE in) {
  RGBA out;
  if (in) {
    out.r = out.g = out.b = out.a = 255;
  } else {
    out.r = out.g = out.b = out.a = 0;
  }
  return out;
}
