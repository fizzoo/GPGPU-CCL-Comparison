#ifndef STRATEGY_H
#define STRATEGY_H

#include <CL/cl.hpp>
#include "LabelData.h"
#include <queue>

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
  virtual std::string name() = 0;

  /**
   * Memory transfer, things we shouldn't time.
   * CPU algorithms can just return.
   */
  virtual void prepare_gpu(cl::Context *, cl::Device *, cl::Program *,
                           LabelData *) {}

  /**
   * The algorithm, compute labels for the binary image and put labels in the
   * same buffer.
   */
  virtual void execute(LabelData *l) = 0;

  /**
   * Clean up programs/memory objects from gpu.
   * (Things that don't need to be done if the gpu later would process another
   * image similarly)
   */
  virtual void clean_gpu() {}

  Strategy(){};
  virtual ~Strategy() {}
};

/**
 * Strategy that doesn't modify the data.
 * For testing thresholding.
 */
class IdStrategy : public Strategy {
public:
  IdStrategy(){};
  virtual std::string name() { return "Identity Strat"; }
  virtual void execute(LabelData *) {}
};

struct XY {
  unsigned int x, y;
  XY(unsigned int x, unsigned int y) : x(x), y(y) {}
};

class CPUOnePass : public Strategy {
private:
  void explore_component(unsigned int x, unsigned int y, LabelData *l,
                         unsigned int nr);

public:
  virtual std::string name() { return "CPU one-pass"; }
  virtual void execute(LabelData *l);
};

class GPUBase : public Strategy {
protected:
  cl::Buffer *buf = nullptr;
  cl::Context *context = nullptr;
  cl::Device *device = nullptr;
  cl::Program *program = nullptr;
  cl::CommandQueue *queue = nullptr;

public:
  virtual void clean_gpu();
  virtual void prepare_gpu(cl::Context *c, cl::Device *d, cl::Program *p,
                           LabelData *l);
};

class GPUNeighbourPropagation : public GPUBase {
public:
  virtual std::string name() { return "GPU neighbour propagation"; }
  virtual void execute(LabelData *l);
};

#endif /* end of include guard: STRATEGY_H */
