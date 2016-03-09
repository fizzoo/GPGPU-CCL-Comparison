#ifndef LABELDATA_H
#define LABELDATA_H

#include <set>
#include <vector>
#include "Image.h"
#include "defines.h"

/**
 * Containing a 2D structure where each element is a single value, such as a
 * label or 0/1 for a binary image.
 */
class LabelData {
public:
  size_t width;
  size_t height;
  LABELTYPE *data = nullptr;

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
  LabelData &operator=(const LabelData &rhs) noexcept;

  /**
   * Copies the label data to an images data with a function deciding the
   * resulting values. Assumes equally sized data portions, in elements.
   */
  void copy_to_image(unsigned char *img_data, RGBA (*img_fun)(LABELTYPE in));

  /**
   * Resets data to 0.
   */
  void clear();

  /**
   * Deallocate data.
   */
  ~LabelData();
};

///////////////////////////////////////
//   UTILITY concerning labeldatas   //
///////////////////////////////////////

void mark_explore(unsigned int x, unsigned int y, LabelData *l, LABELTYPE from,
                  LABELTYPE to);

/**
 * Returns whether the labeldatas are equivalent, where labels may correspond to
 * a different number in the other labeldata.
 */
bool equivalent_result(LabelData *a, LabelData *b);

/**
 * Checks for internal consistency of component labeling.
 */
bool valid_result(LabelData *l);

#endif /* end of include guard: LABELDATA_H */
