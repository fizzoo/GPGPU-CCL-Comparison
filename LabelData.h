#ifndef LABELDATA_H
#define LABELDATA_H

#include "Image.h"

struct RGBA {
  unsigned char r, g, b, a;
};

/**
 * Containing a 2D structure where each element is a single value, such as a
 * label or 0/1 for a binary image.
 */
class LabelData {
public:
  using label_type = int32_t;
  size_t width;
  size_t height;
  label_type *data = nullptr;

  /**
   * Allocate data, copy over from image by thresholding.
   */
  LabelData(iml::Image *img,
            bool (*threshold_function)(unsigned char r, unsigned char g,
                                       unsigned char b, unsigned char a));

  /**
   * Copy everything, allocate anew.
   */
  LabelData(const LabelData &rhs);
  LabelData& operator=(const LabelData &rhs) noexcept;

  /**
   * Copies the label data to an images data with a function deciding the
   * resulting values. Assumes equally sized data portions, in elements.
   */
  void copy_to_image(unsigned char *img_data, RGBA (*img_fun)(label_type in));

  /**
   * Deallocate data.
   */
  ~LabelData();
};

#endif /* end of include guard: LABELDATA_H */
