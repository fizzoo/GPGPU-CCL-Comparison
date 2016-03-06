#ifndef STRATEGY_H
#define STRATEGY_H

#include <CL/cl.hpp>
#include "LabelData.h"

/**
 * Shouldn't need to allocate anything.
 * As we want to store a few objects, constructor/destructor shouldn't do much.
 */
class Strategy {
private:
  Strategy(Strategy const &rhs) = delete;
  Strategy(Strategy &&rhs) = delete;
  Strategy &operator=(Strategy const &rhs) noexcept = delete;
  Strategy &operator=(Strategy &&rhs) noexcept = delete;

public:
  /**
   * Name for identification
   */
  std::string name;

  /**
   * Memory transfer, things we shouldn't time.
   * CPU algorithms can just return.
   */
  virtual void prepare_gpu(cl::Context *context,
                           std::vector<cl::Device> *devices, LabelData *data);

  /**
   * The algorithm, compute labels for the binary image and put labels in the
   * same buffer.
   */
  virtual void execute(LabelData *data);

  /**
   * Clean up programs/memory objects from gpu.
   * (Things that don't need to be done if the gpu later would process another
   * image similarly)
   */
  virtual void clean_gpu();
};

#endif /* end of include guard: STRATEGY_H */
