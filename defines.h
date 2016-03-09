#ifndef DEFINES_H
#define DEFINES_H

#include <iostream>

#define LABELTYPE int32_t

#ifndef NDEBUG
#define CHECKERR                                                               \
  if (err) {                                                                   \
    std::cerr << "UNEXPECTED ERROR (" << err << ") on " << __FILE__ << ":"     \
              << __LINE__ << std::endl;                                        \
  }
#else
#define CHECKERR
#endif /* NDEBUG */

template <class T> void fail(T message, int err = 0) {
  std::cerr << message;
  if (err) {
    std::cerr << " ERR #: " << err;
  }
  std::cerr << std::endl;
  exit(-1);
}

struct RGBA {
  unsigned char r, g, b, a;
};

struct XY {
  size_t x, y;
  XY(size_t x, size_t y) : x(x), y(y) {}
};

#endif /* end of include guard: DEFINES_H */
