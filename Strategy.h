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
  virtual std::string name() = 0;

  /**
   * Memory transfer which we shouldn't time.
   * CPU algorithms aswell should make a copy in order to not edit our input.
   */
  virtual void copy_to(const LabelData *, cl::Context *, cl::Program *,
                       cl::CommandQueue *) = 0;

  /**
   * The algorithm, compute labels for the binary image and save the result
   * somewhere.
   */
  virtual void execute() = 0;

  /**
   * Clean up memory objects and return the results.
   */
  virtual LabelData copy_from() = 0;

  Strategy(){};
  virtual ~Strategy() {}
};

/**
 * ABC for cpu algorithms that simply keep a local LabelData.
 */
class CPUBase : public Strategy {
protected:
  LabelData l;

public:
  virtual void copy_to(const LabelData *, cl::Context *, cl::Program *,
                       cl::CommandQueue *);
  virtual LabelData copy_from();
};

/**
 * Strategy that doesn't modify the data.
 * For testing thresholding.
 */
class IdStrategy : public CPUBase {
public:
  IdStrategy(){};
  virtual std::string name() { return "Identity Strat"; }
  virtual void execute() {}
};

/**
 * One-pass algorithm, explores entire components at a time.
 */
class CPUOnePass : public CPUBase {
public:
  virtual std::string name() { return "CPU one-pass"; }
  virtual void execute();
};

/**
 * ABC for GPU algorithms that keep a cl::Buffer.
 */
class GPUBase : public Strategy {
protected:
  /**
   * Data corresponding to a LabelData, but in gpu.
   */
  cl::Buffer *buf = nullptr;
  size_t width;
  size_t height;

  /**
   * Necessary to create kernel, buffer and queue work.
   */
  cl::Context *context = nullptr;
  cl::Program *program = nullptr;
  cl::CommandQueue *queue = nullptr;

public:
  virtual void copy_to(const LabelData *, cl::Context *, cl::Program *,
                       cl::CommandQueue *);
  virtual LabelData copy_from();
};

class GPUNeighbourPropagation : public GPUBase {
public:
  virtual std::string name() { return "GPU neighbour propagation"; }
  virtual void execute();
};

#endif /* end of include guard: STRATEGY_H */
