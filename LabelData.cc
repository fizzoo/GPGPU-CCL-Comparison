#include "LabelData.h"

LabelData::LabelData(iml::Image *img, bool (*threshold_function)(
                                          unsigned char r, unsigned char g,
                                          unsigned char b, unsigned char a))
    : width(img->width()), height(img->height()) {
  data = new LABELTYPE[height * width];
  auto *in = img->data;
  auto *end = img->data + height * width * 4;
  auto *out = data;
  while (in != end) {
    *out = threshold_function(in[0], in[1], in[2], in[3]);
    in += 4;
    out += 1;
  }
}

LabelData::LabelData(const LabelData &rhs)
    : width(rhs.width), height(rhs.height) {
  data = new LABELTYPE[height * width];
  std::copy(rhs.data, rhs.data + height * width, data);
}

LabelData &LabelData::operator=(const LabelData &rhs) noexcept {
  if (this != &rhs) {
    delete[] data;
    width = rhs.width;
    height = rhs.height;
    data = new LABELTYPE[height * width];
    std::copy(rhs.data, rhs.data + height * width, data);
  }
  return *this;
}

LabelData::LabelData(LabelData &&rhs) {
  width = rhs.width;
  height = rhs.height;
  data = rhs.data;
  rhs.width = 0;
  rhs.height = 0;
  rhs.data = 0;
}

LabelData &LabelData::operator=(LabelData &&rhs) noexcept {
  if (this != &rhs) {
    width = rhs.width;
    height = rhs.height;
    data = rhs.data;
    rhs.width = 0;
    rhs.height = 0;
    rhs.data = 0;
  }
  return *this;
}

LabelData::LabelData(size_t width, size_t height)
    : width(width), height(height) {
  data = new LABELTYPE[width * height];
}

LabelData::LabelData() : width(0), height(0) {}

LabelData::~LabelData() { delete[] data; }

void LabelData::copy_to_image(unsigned char *img_data,
                              RGBA (*img_fun)(LABELTYPE in)) const {
  auto *in = data;
  auto *end = data + width * height;
  auto *out = img_data;
  while (in != end) {
    auto rgba = img_fun(*in);
    out[0] = rgba.r;
    out[1] = rgba.g;
    out[2] = rgba.b;
    out[3] = rgba.a;
    in += 1;
    out += 4;
  }
}

void LabelData::clear() {
  auto *in = data;
  auto *end = data + width * height;
  while (in != end) {
    *in = 0;
    ++in;
  }
}

bool equivalent_result(LabelData *a, LabelData *b) {
  if (a->width != b->width || a->height != b->height) {
    std::cerr << "Mismatched sizes" << std::endl;
    return false;
  }

  for (size_t y = 0; y < a->height; ++y) {
    for (size_t x = 0; x < a->width; ++x) {
      auto cura = a->data[a->width * y + x];
      auto curb = b->data[b->width * y + x];
      if ((cura == 0 && curb != 0) || (curb == 0 && cura != 0)) {
        std::cerr << "Component on one labeling but none on the other"
                  << std::endl;
        return false;
      }
    }
  }

  return true;
}

bool valid_result(LabelData *l) {
  auto w = l->width;
  auto h = l->height;
  const LABELTYPE *d = l->data;

  for (size_t y = 0; y < h; ++y) {
    for (size_t x = 0; x < w; ++x) {
      auto curlabel = d[w * y + x];

      if (curlabel > 1 << 24) {
        std::cerr << "Labelnr above 1 << 24!" << std::endl;
      }

      if (curlabel == 1) {
        std::cerr << "Unlabeled pixel at x:" << x << " y:" << y << std::endl;
        return false;
      }

      if (curlabel != 0) {
        bool closefault = false;
        closefault |= x + 1 < w && d[w * (y) + (x + 1)] != 0 &&
                      d[w * (y) + (x + 1)] != curlabel;
        closefault |= x - 1 < w && d[w * (y) + (x - 1)] != 0 &&
                      d[w * (y) + (x - 1)] != curlabel;
        closefault |= y + 1 < h && d[w * (y + 1) + (x)] != 0 &&
                      d[w * (y + 1) + (x)] != curlabel;
        closefault |= y - 1 < h && d[w * (y - 1) + (x)] != 0 &&
                      d[w * (y - 1) + (x)] != curlabel;

        if (closefault) {
          std::cerr << "Connected components with different labels at x:" << x
                    << " y:" << y << std::endl;
          return false;
        }
      }
    }
  }

  LabelData tmp(*l);
  std::set<LABELTYPE> prev;
  for (size_t y = 0; y < h; ++y) {
    for (size_t x = 0; x < w; ++x) {
      auto curlabel = tmp.data[w * y + x];

      // Only do anything if it's some component
      if (curlabel > 1) {
        // Another component already used the label
        if (prev.count(curlabel)) {
          std::cerr << "Multiple components with same label: " << curlabel
                    << std::endl;
          return false;
        }

        // Record that this component uses this label
        prev.insert(curlabel);

        // Set to zero to ignore this component in next iteration
        mark_explore(x, y, &tmp, curlabel, 0);
      }
    }
  }

  return true;
}

void mark_explore(size_t xinit, size_t yinit, LabelData *l, LABELTYPE from,
                  LABELTYPE to) {
  auto w = l->width;
  auto h = l->height;
  auto d = l->data;

  if (d[w * yinit + xinit] != from) {
    return;
  }
  d[w * yinit + xinit] = to;

  std::vector<XY> xys;
  xys.emplace_back(xinit, yinit);
  while (!xys.empty()) {
    auto xy = xys.back();
    xys.pop_back();
    auto x = xy.x;
    auto y = xy.y;

    if (x + 1 < w && d[w * y + (x + 1)] == from) {
      d[w * y + (x + 1)] = to;
      xys.emplace_back(x + 1, y);
    }
    if (x - 1 < w && d[w * y + (x - 1)] == from) {
      d[w * y + (x - 1)] = to;
      xys.emplace_back(x - 1, y);
    }
    if (y + 1 < h && d[w * (y + 1) + x] == from) {
      d[w * (y + 1) + x] = to;
      xys.emplace_back(x, y + 1);
    }
    if (y - 1 < h && d[w * (y - 1) + x] == from) {
      d[w * (y - 1) + x] = to;
      xys.emplace_back(x, y - 1);
    }
  }
}
