#include "Strategy.h"

void mark_explore(unsigned int xinit, unsigned int yinit, LabelData *l,
                  LabelData::label_type from, LabelData::label_type to) {
  auto w = l->width;
  auto h = l->height;
  auto d = l->data;

  if (d[w * yinit + xinit] != from) {
    return;
  }
  d[w * yinit + xinit] = to;

  std::vector<XY> xys;
  xys.emplace_back(xinit, yinit);
  while (!xys.empty()) {
    auto xy = xys.back();
    xys.pop_back();
    auto x = xy.x;
    auto y = xy.y;

    if (x + 1 < w && d[w * y + (x + 1)] == from) {
      d[w * y + (x + 1)] = to;
      xys.emplace_back(x + 1, y);
    }
    if (x - 1 < w && d[w * y + (x - 1)] == from) {
      d[w * y + (x - 1)] = to;
      xys.emplace_back(x - 1, y);
    }
    if (y + 1 < h && d[w * (y + 1) + x] == from) {
      d[w * (y + 1) + x] = to;
      xys.emplace_back(x, y + 1);
    }
    if (y - 1 < h && d[w * (y - 1) + x] == from) {
      d[w * (y - 1) + x] = to;
      xys.emplace_back(x, y - 1);
    }
  }
}

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

#ifndef NDEBUG
#define CHECKERR                                                               \
  if (err) {                                                                   \
    std::cerr << "UNEXPECTED ERROR (" << err << ") on " << __FILE__ << ":"     \
              << __LINE__ << std::endl;                                        \
  }
#else
#define CHECKERR
#endif /* NDEBUG */

void GPUBase::clean_gpu() {
  delete buf;
  delete queue;
}

void GPUBase::prepare_gpu(cl::Context *c, cl::Device *d, cl::Program *p,
                          LabelData *l) {
  context = c;
  device = d;
  program = p;

  cl_int err;
  auto size = l->width * l->height * sizeof(LabelData::label_type);
  buf = new cl::Buffer(*c, CL_MEM_READ_WRITE, size, nullptr, &err);
  CHECKERR

  queue = new cl::CommandQueue(*context, *device, 0, &err);
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
