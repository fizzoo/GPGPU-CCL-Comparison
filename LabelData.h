#ifndef LABELDATA_H
#define LABELDATA_H

#include "Image.h"

/**
 * Containing a 2D structure where each element is a single value, such as a
 * label or 0/1 for a binary image.
 */
class LabelData {
public:
  using label_type = unsigned char;
  size_t width;
  size_t height;
  label_type *data;

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

  /**
   * Deallocate data.
   */
  ~LabelData();
};

#endif /* end of include guard: LABELDATA_H */
