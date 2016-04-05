#include "RGBAConversions.h"

bool rgb_above_128(unsigned char r, unsigned char g, unsigned char b,
                   unsigned char) {
  if (r > 128 && g > 128 && b > 128) {
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

RGBA mod8(LABELTYPE in) {
  RGBA out;
  out.a = 255;
  if (in == 1) {
    out.r = out.g = out.b = 255;
  } else if (in == 0) {
    out.r = out.g = out.b = 0;
  } else {
    char choice = in % 8;
    switch (choice) {
    case 0:
      out.r = 255;
      out.g = 0;
      out.b = 0;
      break;
    case 1:
      out.r = 0;
      out.g = 255;
      out.b = 0;
      break;
    case 2:
      out.r = 0;
      out.g = 0;
      out.b = 255;
      break;
    case 3:
      out.r = 255;
      out.g = 255;
      out.b = 0;
      break;
    case 4:
      out.r = 255;
      out.g = 0;
      out.b = 255;
      break;
    case 5:
      out.r = 0;
      out.g = 255;
      out.b = 255;
      break;
    case 6:
      out.r = 100;
      out.g = 160;
      out.b = 255;
      break;
    case 7:
      out.r = 255;
      out.g = 80;
      out.b = 80;
      break;
    }
  }
  return out;
}
