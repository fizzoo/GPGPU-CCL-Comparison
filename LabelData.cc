#include "LabelData.h"

LabelData::LabelData(iml::Image *img, bool (*threshold_function)(
                                          unsigned char r, unsigned char g,
                                          unsigned char b, unsigned char a))
    : width(img->width()), height(img->height()) {
  data = new label_type[height * width];
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
  data = new label_type[height * width];
  std::copy(rhs.data, rhs.data + height * width, data);
}

LabelData &LabelData::operator=(const LabelData &rhs) noexcept {
  if (this != &rhs) {
    delete data;
    width = rhs.width;
    height = rhs.height;
    data = new label_type[height * width];
    std::copy(rhs.data, rhs.data + height * width, data);
  }
  return *this;
}

LabelData::~LabelData() { delete[] data; }

void LabelData::copy_to_image(unsigned char *img_data,
                              RGBA (*img_fun)(label_type in)) {
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
