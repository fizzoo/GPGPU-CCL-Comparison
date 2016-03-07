#include "Strategy.h"

void CPUOnePass::explore_component(unsigned int xinit, unsigned int yinit,
                                   LabelData *l, unsigned int nr) {
  auto w = l->width;
  auto h = l->height;
  auto d = l->data;

  d[w * yinit + xinit] = nr;

  std::vector<XY> xys;
  xys.emplace_back(xinit, yinit);
  while (!xys.empty()) {
    auto xy = xys.back();
    xys.pop_back();
    auto x = xy.x;
    auto y = xy.y;

    if (x + 1 < w && d[w * y + (x + 1)] == 1) {
      d[w * y + (x + 1)] = nr;
      xys.emplace_back(x + 1, y);
    }
    if (x - 1 < w && d[w * y + (x - 1)] == 1) {
      d[w * y + (x - 1)] = nr;
      xys.emplace_back(x - 1, y);
    }
    if (y + 1 < h && d[w * (y + 1) + x] == 1) {
      d[w * (y + 1) + x] = nr;
      xys.emplace_back(x, y + 1);
    }
    if (y - 1 < h && d[w * (y - 1) + x] == 1) {
      d[w * (y - 1) + x] = nr;
      xys.emplace_back(x, y - 1);
    }
  }
}

void CPUOnePass::execute(LabelData *l) {
  unsigned int nr = 2;
  for (unsigned int x = 0; x < l->width; ++x) {
    for (unsigned int y = 0; y < l->height; ++y) {
      if (l->data[l->width * y + x] == 1) {
        explore_component(x, y, l, nr);
        ++nr;
#ifndef NDEBUG
        if (nr > 1 << 15) {
          std::cerr << "Labelnr close to max!" << std::endl;
        }
#endif /* NDEBUG */
      }
    }
  }
}
