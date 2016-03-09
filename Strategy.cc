#include "Strategy.h"

void CPUBase::copy_to(const LabelData *in, cl::Context *, cl::Program *,
                      cl::CommandQueue *) {
  l = *in;
}

LabelData CPUBase::copy_from() { return std::move(l); }

void CPUOnePass::execute() {
  size_t nr = 2;
  for (size_t y = 0; y < l.height; ++y) {
    for (size_t x = 0; x < l.width; ++x) {
      if (l.data[l.width * y + x] == 1) {
        mark_explore(x, y, &l, 1, nr);
        ++nr;
      }
    }
  }
}

void GPUBase::copy_to(const LabelData *l, cl::Context *c, cl::Program *p,
                      cl::CommandQueue *q) {
  queue = q;
  program = p;
  width = l->width;
  height = l->height;

  cl_int err;
  auto size = width * height * sizeof(LABELTYPE);
  buf = new cl::Buffer(*c, CL_MEM_READ_WRITE, size, nullptr, &err);
  CHECKERR

  err = queue->enqueueWriteBuffer(*buf, CL_TRUE, 0, size, l->data);
}

void GPUNeighbourPropagation::execute() {
  cl_int err;

  cl::Kernel kernel(*program, "label_with_id", &err);
  CHECKERR

  err = kernel.setArg(0, *buf);
  CHECKERR
  err = kernel.setArg(1, (cl_int)width);
  CHECKERR

  cl::Event event;
  err = queue->enqueueNDRangeKernel(kernel, cl::NullRange,
                                    cl::NDRange(width, height),
                                    cl::NDRange(1, 1), NULL, &event);
  CHECKERR

  event.wait();
}

LabelData GPUBase::copy_from() {
  LabelData ret(width, height);

  auto size = width * height * sizeof(LABELTYPE);
  queue->enqueueReadBuffer(*buf, CL_TRUE, 0, size, ret.data);

  delete buf;

  return ret;
}
