#include "Strategy.h"

void CPUOnePass::execute(LabelData *l) {
  unsigned int nr = 2;
  for (unsigned int y = 0; y < l->height; ++y) {
    for (unsigned int x = 0; x < l->width; ++x) {
      if (l->data[l->width * y + x] == 1) {
        mark_explore(x, y, l, 1, nr);
        ++nr;
      }
    }
  }
}

void GPUBase::clean_gpu() { delete buf; }

void GPUBase::prepare_gpu(cl::Context *c, cl::Device *d, cl::Program *p,
                          LabelData *l) {
  context = c;
  device = d;
  program = p;

  cl_int err;
  auto size = l->width * l->height * sizeof(LABELTYPE);
  buf = new cl::Buffer(*c, CL_MEM_READ_WRITE, size, nullptr, &err);
  CHECKERR

  queue = new cl::CommandQueue(*c, *d, 0, &err);
  CHECKERR

  cl::Event event;
  err = queue->enqueueWriteBuffer(*buf, CL_FALSE, 0, size, l->data, 0, &event);
  event.wait();
}

void GPUNeighbourPropagation::execute(LabelData *l) {
  cl_int err;

  cl::Kernel kernel(*program, "label_with_id", &err);
  CHECKERR

  err = kernel.setArg(0, *buf);
  CHECKERR
  err = kernel.setArg(1, (cl_int)l->width);
  CHECKERR

  cl::Event event;
  err = queue->enqueueNDRangeKernel(kernel, cl::NullRange,
                                    cl::NDRange(l->width, l->height),
                                    cl::NDRange(1, 1), NULL, &event);
  CHECKERR

  event.wait();
}
