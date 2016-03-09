#ifndef LABELDATA_H
#define LABELDATA_H

#include <set>
#include <vector>
#include <map>
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
   * Steal data.
   */
  LabelData(LabelData &&rhs);
  LabelData &operator=(LabelData &&rhs) noexcept;

  /**
   * Just allocate.
   */
  LabelData(size_t width, size_t height);

  /**
   * Do nothing.
   */
  LabelData();

  /**
   * Copies the label data to an images data with a function deciding the
   * resulting values. Assumes equally sized data portions, in elements.
   */
  void copy_to_image(unsigned char *img_data,
                     RGBA (*img_fun)(LABELTYPE in)) const;

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

void mark_explore(size_t x, size_t y, LabelData *l, LABELTYPE from,
                  LABELTYPE to);

/**
 * Returns whether the components of the two are at the same location.
 * If both pass valid_result, their data is equivalent aside from label numbers
 * if it passes this.
 */
bool equivalent_result(LabelData *a, LabelData *b);

/**
 * Checks for internal consistency of component labeling.
 */
bool valid_result(LabelData *l);

#endif /* end of include guard: LABELDATA_H */
