#ifndef STRATEGY_H
#define STRATEGY_H

#include <CL/cl.hpp>
#include "LabelData.h"

/**
 * ABC representing a strategy for solving CCL.
 * Shouldn't need to allocate anything.
 * As we want to store a few objects without any side effects,
 * constructor/destructor shouldn't do much.
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
   * CPU algorithms should also make a copy in order to not edit our input.
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
  virtual ~CPUBase() {}
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
 * Uses a queue to record pixels to be explored, and propagates thusly.
 */
class CPUOnePass : public CPUBase {
public:
  virtual std::string name() { return "CPU one-pass"; }
  virtual void execute();
};

/**
 * Two-pass algorithm, using union-find for the equivalence.
 * Equivalences are recorded inside the labeling themselves, similar to the gpu
 * version.
 */
class CPUUnionFind : public CPUBase {
public:
  virtual std::string name() { return "CPU union-find"; }
  virtual void execute();
  int find_set(int location);
};

/**
 * Two-pass algorithm as proposed by Lifeng He, Yuyan Chao and
 * Kenju Suzuki.
 *
 * Initial scan will (if needed) assign a new label or try to assign
 * current pixel a existing label. To keep track of label relations
 * three vectors are used. These vectors are the relation label table,
 * the next table and the tail table.
 * If two or more labels are part of the same component then by the
 * end of the first scan they will both have the same value in the
 * RL table. This smalest label will have a tail in the tail label,
 * which is the last label in the component.
 * The next label simply points to the next label in the component,
 * starting at the smalest and ending at the tail
 *
 * Since all the label relations are taken care of in the first scan
 * the 2nd scan simply assigns the correct label to all the pixels
 */
class CPULinearTwoScan : public CPUBase {
private:
  std::vector<unsigned int> rl_table;
  // Next
  std::vector<unsigned int> n_label;
  // Tail
  std::vector<unsigned int> t_label;

public:
  virtual std::string name() { return "CPU linear two-scan"; }
  virtual void execute();
  virtual void copy_to(const LabelData *, cl::Context *, cl::Program *,
                       cl::CommandQueue *);
};

/**
 * Multipass algorithm as proposed by Kenji Suzuki, Isao Horiba and Noboru
 * Sugie.
 *
 * The initial scan is used to, in the forward raster direction, assign
 * initial labels and resolve some simple label eqvivalnces along the way.
 * After that multiple scans in the backwards and forwards raster direction
 * are done until no more change in result is seen. Theses subsequent scans
 * are done in pairs of one backwards and one forwards scan.
 */
class CPUFrontBack : public CPUBase {
private:
  std::vector<int> labelConnT;

public:
  virtual std::string name() { return "CPU front back scan"; }
  virtual void execute();
  virtual void copy_to(const LabelData *, cl::Context *, cl::Program *,
                       cl::CommandQueue *);
};

/**
 * ABC for GPU algorithms that keep a cl::Buffer and queues work.
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
  virtual ~GPUBase() {}
};

/**
 * Neighbour propagation using only the closest connected pixels.
 */
class GPUNeighbourPropagation : public GPUBase {
public:
  virtual std::string name() { return "GPU neighbour propagation"; }
  virtual void execute();
};

/**
 * Neighbour propagation that solves the problem locally between iterations.
 */
class GPUNeighbourPropagation_Localer : public GPUBase {
public:
  virtual std::string name() { return "GPU neighbour propagation +local"; }
  virtual void execute();
};

/**
 * Like neighbour propagation, but looks at all the pixels in a line towards
 * up/down/left/right while inside a component.
 */
class GPUPlusPropagation : public GPUBase {
public:
  virtual std::string name() { return "GPU plus propagation"; }
  virtual void execute();
};

/**
 * Union-Find, checks which trees its neighbours belong to and follows it to
 * the root, whereupon it picks the smallest root and modifies relevant nearby
 * pixels to match that.  Essentially we then have a full root-compression
 * scheme.  The tree structure itself is embedded in the label data, through
 * the labels corresponding to the pixel at index label-2.  A root of a tree
 * then has the label of its own index+2.
 */
class GPUUnionFind : public GPUBase {
public:
  virtual std::string name() { return "GPU union-find"; }
  virtual void execute();
};

/**
 * Union-find as above, augmented with a local solving inbetween iterations.
 */
class GPUUnionFind_Localer : public GPUBase {
public:
  virtual std::string name() { return "GPU union-find +local"; }
  virtual void execute();
};

/**
 * Traverses row/column forward/backwards and edits at the same time.
 */
class GPULineEditing : public GPUBase {
public:
  virtual std::string name() { return "GPU line editing"; }
  virtual void execute();
};

/**
 * Traverses row/column and reads the entire connected part before writing.
 */
class GPULines : public GPUBase {
public:
  virtual std::string name() { return "GPU lines"; }
  virtual void execute();
};

/**
 * Uses a stack, similar to the one-pass of the cpu, except that due to
 * limitations in stack size we will need to iterate until convergence.  Every
 * workgroup has its own stack, and picks one of the local labels to work on in
 * case there is one that can be spread.  The workgroup then cooperates with
 * working off the stack, propagating the results and then adding to the stack.
 */
class GPUStackOnePass : public GPUBase {
public:
  virtual std::string name() { return "GPU stack-based"; }
  virtual void execute();
};

#endif /* end of include guard: STRATEGY_H */
