#include "Strategy.h"
using namespace boost;

int round_to_nearest(int x, int mod) {
  if (x % mod) {
    x = x + mod - (x % mod);
  }
  return x;
}

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

void CPUUnionFind::execute() {
  size_t nr = 2;
  auto w = l.width;
  auto h = l.height;
  auto d = l.data;

  std::map<size_t, int> rank;
  std::map<size_t, size_t> p;
  disjoint_sets<associative_property_map<std::map<size_t, int>>,
                associative_property_map<std::map<size_t, size_t>>> dset(rank,
                                                                         p);
  for (size_t y = 0; y < h; ++y) {
    for (size_t x = 0; x < w; ++x) {
      if (d[w * y + x] == 1) {

        if (x > 0 && y > 0 && d[w * (y - 1) + (x)] && d[w * (y) + (x - 1)]) {
          // Both foreground
          d[w * y + x] = d[w * (y) + (x - 1)];

          if (d[w * (y - 1) + (x)] != d[w * (y) + (x - 1)]) {
            dset.union_set(d[w * (y - 1) + (x)], d[w * (y) + (x - 1)]);
          }
        } else if (x > 0 && d[w * (y) + (x - 1)]) {
          d[w * y + x] = d[w * (y) + (x - 1)];
        } else if (y > 0 && d[w * (y - 1) + (x)]) {
          d[w * y + x] = d[w * (y - 1) + (x)];
        } else {
          dset.make_set(nr);
          d[w * y + x] = nr;
          ++nr;
        }
      }
    }
  }

  for (size_t y = 0; y < h; ++y) {
    for (size_t x = 0; x < w; ++x) {
      if (d[w * y + x]) {
        d[w * y + x] = dset.find_set(d[w * y + x]);
      }
    }
  }
}

int CPUUnionFindReusing::findset(int loc) {
  // All loc of found elements should be in range.
  // Also assuming there are no cycles in the links.
  while (loc != l.data[loc] - 2) {
    loc = l.data[loc] - 2;
  }

  // +2 is the correct LABEL of root at LOCATION loc
  return loc + 2;
}

void CPUUnionFindReusing::execute() {
  auto w = l.width;
  auto h = l.height;
  auto d = l.data;

  for (size_t y = 0; y < h; ++y) {
    for (size_t x = 0; x < w; ++x) {
      int locCur = w * y + x, locN = w * (y - 1) + (x),
          locW = w * (y) + (x - 1);

      if (d[locCur] == 1) {
        if (x > 0 && y > 0 && d[locN] && d[locW]) {
          // Both foreground
          int N = findset(locN);
          int W = findset(locW);

          if (N < W) { // Less is more
            d[locCur] = N;
            d[W - 2] = N;
          } else {
            d[locCur] = W;
            d[N - 2] = W;
          }
        } else if (x > 0 && d[locW]) {
          d[locCur] = d[locW];
        } else if (y > 0 && d[locN]) {
          d[locCur] = d[locN];
        } else {
          d[locCur] = w * y + x + 2;
        }
      }
    }
  }

  for (size_t y = 0; y < h; ++y) {
    for (size_t x = 0; x < w; ++x) {
      if (d[w * y + x]) {
        d[w * y + x] = findset(w * y + x);
      }
    }
  }
}

void CPULinearTwoScan::execute() {
  auto w = l.width;
  auto h = l.height;
  auto d = l.data;

  std::vector<unsigned int> rl_table(w * h, 0);
  // next label
  std::vector<unsigned int> n_label(w * h);
  // tail label
  std::vector<unsigned int> t_label(w * h);

  int m = 2;

  // first scan, pretty much everything is done here
  for (size_t y = 0; y < h; ++y) {
    for (size_t x = 0; x < w; ++x) {
      // position of b(x, y);
      int bXY = y * w + x;

      if (d[bXY]) {
        // left pixel
        int lP = 0;
        if (x) { // OOR check
          lP = d[y * w + (x - 1)];
        }

        // upper pixel
        int uP = 0;
        if (y) { // OOR check
          uP = d[(y - 1) * w + x];
        }

        // Need new label
        if (!lP && !uP) {
          d[bXY] = m;
          rl_table[m] = m;
          n_label[m] = -1;
          t_label[m] = m;
          ++m;
        } else if (lP) {
          d[bXY] = lP;
        } else {
          d[bXY] = uP;
        }

        unsigned int u = rl_table[lP];
        unsigned int v = rl_table[uP];
        // this part resolves potential label equvalence
        if (u > 1 && v > 1 && u != v) {

          // can uncomment this section if we prefer lower labels
          /*if (v < u) {
              std::swap(u, v);
          }*/

          // this part is coded exactly as shown with pseudo code in the paper
          int i = v;
          while (i != -1) {
            rl_table[i] = u;
            i = n_label[i];
          }
          n_label[t_label[u]] = v;
          t_label[u] = t_label[v];
        }
      }
    }
  }

  // 2nd scan, only asigns correct values
  for (size_t y = 0; y < h; ++y) {
    for (size_t x = 0; x < w; ++x) {
      if (d[y * w + x] != 0) {
        d[y * w + x] = rl_table[d[y * w + x]];
      }
    }
  }
}

void CPUFrontBack::execute() {
  auto w = l.width;
  auto h = l.height;
  auto d = l.data;

  // label connection table
  std::vector<int> labelConnT(w * h, 0);
  labelConnT[1] = 1;

  int m = 2;
  bool change = true;

  for (size_t y = 0; y < h; ++y) {
    for (size_t x = 0; x < w; ++x) {
      // position of b(x, y);
      int bXY = y * w + x;
      if (d[bXY]) {
        // left pixel
        int lP = 0;
        if (x) { // OOR check
          lP = d[y * w + (x - 1)];
        }

        // upper pixel
        int uP = 0;
        if (y) { // OOR check
          uP = d[(y - 1) * w + x];
        }

        if (!lP && !uP) {
          d[bXY] = m;
          labelConnT[m] = m;
          ++m;
        } else {
          int min; // = d[bXY];
          if (lP && !uP) {
            min = labelConnT[lP];
          } else if (!lP && uP) {
            min = labelConnT[uP];
          } else {
            min = (labelConnT[lP] < labelConnT[uP]) ? labelConnT[lP]
                                                    : labelConnT[uP];
          }
          d[bXY] = min;
          if (uP) {
            labelConnT[uP] = min;
          }
          if (lP) {
            labelConnT[lP] = min;
          }
        }
      }
    }
  }

  while (change) {
    change = false;

    // backwards scan
    for (int y = h - 1; y >= 0; --y) {
      for (int x = w - 1; x >= 0; --x) {
        int bXY = y * w + x;
        if (d[bXY]) {
          // right pixel
          int rP = 0;
          if (x != (int)w - 1) { // OOR check
            rP = d[y * w + (x + 1)];
          }

          // south pixel
          int sP = 0;
          if (y != (int)h - 1) { // OOR check
            sP = d[(y + 1) * w + x];
          }

          int min = -1;
          int tMin = labelConnT[d[bXY]];

          if (rP && sP) {
            min = (labelConnT[rP] < labelConnT[sP]) ? labelConnT[rP]
                                                    : labelConnT[sP];
          } else if (rP && !sP) {
            min = labelConnT[rP];
          } else if (!rP && sP) {
            min = labelConnT[sP];
          }

          if (tMin < min || min == -1) {
            min = tMin;
          }
          d[bXY] = min;

          if (rP && labelConnT[rP] != min) {
            labelConnT[rP] = min;
            change = true;
          }
          if (sP && labelConnT[sP] != min) {
            labelConnT[sP] = min;
            change = true;
          }
          if (labelConnT[d[bXY]] != min) {
            labelConnT[d[bXY]] = min;
            change = true;
          }
        }
      }
    }
    for (size_t y = 0; y < h; ++y) {
      for (size_t x = 0; x < w; ++x) {
        int bXY = y * w + x;
        if (d[bXY]) {
          // left pixel
          int lP = 0;
          if (x) { // OOR check
            lP = d[y * w + (x - 1)];
          }

          // upper pixel
          int uP = 0;
          if (y) { // OOR check
            uP = d[(y - 1) * w + x];
          }

          int min = -1;
          int tMin = labelConnT[d[bXY]];

          if (lP && uP) {
            min = (labelConnT[lP] < labelConnT[uP]) ? labelConnT[lP]
                                                    : labelConnT[uP];
          } else if (lP && !uP) {
            min = labelConnT[lP];
          } else if (!lP && uP) {
            min = labelConnT[uP];
          }

          if (tMin < min || min == -1) {
            min = tMin;
          }
          d[bXY] = min;

          if (lP && labelConnT[lP] != min) {
            labelConnT[lP] = min;
            change = true;
          }
          if (uP && labelConnT[uP] != min) {
            labelConnT[uP] = min;
            change = true;
          }
          if (labelConnT[d[bXY]] != min) {
            labelConnT[d[bXY]] = min;
            change = true;
          }
        }
      }
    }
  }
}

void GPUBase::copy_to(const LabelData *l, cl::Context *c, cl::Program *p,
                      cl::CommandQueue *q) {
  context = c;
  queue = q;
  program = p;
  width = l->width;
  height = l->height;

  cl_int err;
  auto size = width * height * sizeof(LABELTYPE);
  buf = new cl::Buffer(*c, CL_MEM_READ_WRITE, size, nullptr, &err);
  CHECKERR;

  err = queue->enqueueWriteBuffer(*buf, CL_TRUE, 0, size, l->data);
}

LabelData GPUBase::copy_from() {
  LabelData ret(width, height);

  auto size = width * height * sizeof(LABELTYPE);
  queue->enqueueReadBuffer(*buf, CL_TRUE, 0, size, ret.data);

  delete buf;

  return ret;
}

void GPUNeighbourPropagation::execute() {
  cl_int err;

  const int wgw = 32;
  const int wgh = 8;
  const int wsize = round_to_nearest(width, wgw);
  const int hsize = round_to_nearest(height, wgh);

  cl::Kernel startlabel(*program, "label_with_id", &err);
  CHECKERR;
  cl::Kernel propagate(*program, "neighbour_propagate", &err);
  CHECKERR;

  err = startlabel.setArg(0, *buf);
  CHECKERR;
  err = startlabel.setArg(1, (cl_int)width);
  CHECKERR;
  err = startlabel.setArg(2, (cl_int)height);
  CHECKERR;

  char changed = 1;
  cl::Buffer chan(*context, CL_MEM_READ_WRITE, (size_t)1, nullptr, &err);
  queue->enqueueWriteBuffer(chan, CL_FALSE, 0, 1, &changed);

  err = propagate.setArg(0, *buf);
  CHECKERR;
  err = propagate.setArg(1, (cl_int)width);
  CHECKERR;
  err = propagate.setArg(2, (cl_int)height);
  CHECKERR;
  err = propagate.setArg(3, chan);
  CHECKERR;

  err = queue->enqueueNDRangeKernel(startlabel, cl::NullRange,
                                    cl::NDRange(wsize, hsize),
                                    cl::NDRange(wgw, wgh));
  CHECKERR;

  while (true) {
    // CPU-GPU sync, sadly
    queue->enqueueReadBuffer(chan, CL_TRUE, 0, 1, &changed);
    if (changed == false) {
      break;
    }
    changed = false;
    queue->enqueueWriteBuffer(chan, CL_FALSE, 0, 1, &changed);

    queue->enqueueNDRangeKernel(propagate, cl::NullRange,
                                cl::NDRange(wsize, hsize),
                                cl::NDRange(wgw, wgh));
  }
}

void GPUNeighbourPropagation_Localer::execute() {
  cl_int err;

  const int wgw = 32;
  const int wgh = 8;
  const int wsize = round_to_nearest(width, wgw);
  const int hsize = round_to_nearest(height, wgh);

  cl::Kernel startlabel(*program, "label_with_id", &err);
  CHECKERR;
  cl::Kernel localer(*program, "plus_once_locally", &err);
  CHECKERR;
  cl::Kernel propagate(*program, "neighbour_propagate", &err);
  CHECKERR;

  err = startlabel.setArg(0, *buf);
  CHECKERR;
  err = startlabel.setArg(1, (cl_int)width);
  CHECKERR;
  err = startlabel.setArg(2, (cl_int)height);
  CHECKERR;

  err = localer.setArg(0, *buf);
  CHECKERR;
  err = localer.setArg(1, (cl_int)width);
  CHECKERR;
  err = localer.setArg(2, (cl_int)height);
  CHECKERR;

  char changed = 1;
  cl::Buffer chan(*context, CL_MEM_READ_WRITE, (size_t)1, nullptr, &err);
  queue->enqueueWriteBuffer(chan, CL_FALSE, 0, 1, &changed);

  err = propagate.setArg(0, *buf);
  CHECKERR;
  err = propagate.setArg(1, (cl_int)width);
  CHECKERR;
  err = propagate.setArg(2, (cl_int)height);
  CHECKERR;
  err = propagate.setArg(3, chan);
  CHECKERR;

  err = queue->enqueueNDRangeKernel(startlabel, cl::NullRange,
                                    cl::NDRange(wsize, hsize),
                                    cl::NDRange(wgw, wgh));
  CHECKERR;

  while (true) {
    // CPU-GPU sync, sadly
    queue->enqueueReadBuffer(chan, CL_TRUE, 0, 1, &changed);
    if (changed == false) {
      break;
    }
    changed = false;
    queue->enqueueWriteBuffer(chan, CL_FALSE, 0, 1, &changed);
    queue->enqueueNDRangeKernel(localer, cl::NullRange,
                                cl::NDRange(wsize, hsize),
                                cl::NDRange(wgw, wgh));
    queue->enqueueNDRangeKernel(propagate, cl::NullRange,
                                cl::NDRange(wsize, hsize),
                                cl::NDRange(wgw, wgh));
  }
}

void GPUPlusPropagation::execute() {
  cl_int err;

  const int wgw = 16;
  const int wgh = 8;
  const int wsize = round_to_nearest(width, wgw);
  const int hsize = round_to_nearest(height, wgh);

  cl::Kernel startlabel(*program, "label_with_id", &err);
  CHECKERR;
  cl::Kernel propagate(*program, "plus_propagate", &err);
  CHECKERR;

  err = startlabel.setArg(0, *buf);
  CHECKERR;
  err = startlabel.setArg(1, (cl_int)width);
  CHECKERR;
  err = startlabel.setArg(2, (cl_int)height);
  CHECKERR;

  char changed = 1;
  cl::Buffer chan(*context, CL_MEM_READ_WRITE, (size_t)1, nullptr, &err);
  queue->enqueueWriteBuffer(chan, CL_FALSE, 0, 1, &changed);

  err = propagate.setArg(0, *buf);
  CHECKERR;
  err = propagate.setArg(1, (cl_int)width);
  CHECKERR;
  err = propagate.setArg(2, (cl_int)height);
  CHECKERR;
  err = propagate.setArg(3, chan);
  CHECKERR;

  err = queue->enqueueNDRangeKernel(startlabel, cl::NullRange,
                                    cl::NDRange(wsize, hsize),
                                    cl::NDRange(wgw, wgh));
  CHECKERR;

  while (true) {
    // CPU-GPU sync, sadly
    queue->enqueueReadBuffer(chan, CL_TRUE, 0, 1, &changed);
    if (changed == false) {
      break;
    }
    changed = false;
    queue->enqueueWriteBuffer(chan, CL_FALSE, 0, 1, &changed);
    queue->enqueueNDRangeKernel(propagate, cl::NullRange,
                                cl::NDRange(wsize, hsize),
                                cl::NDRange(wgw, wgh));
  }
}

void GPUUnionFind::execute() {
  cl_int err;

  const int wgw = 32;
  const int wgh = 8;
  const int wsize = round_to_nearest(width, wgw);
  const int hsize = round_to_nearest(height, wgh);

  cl::Kernel startlabel(*program, "label_with_id", &err);
  CHECKERR;
  cl::Kernel propagate(*program, "union_find", &err);
  CHECKERR;

  err = startlabel.setArg(0, *buf);
  CHECKERR;
  err = startlabel.setArg(1, (cl_int)width);
  CHECKERR;
  err = startlabel.setArg(2, (cl_int)height);
  CHECKERR;

  char changed = 1;
  cl::Buffer chan(*context, CL_MEM_READ_WRITE, (size_t)1, nullptr, &err);
  queue->enqueueWriteBuffer(chan, CL_FALSE, 0, 1, &changed);

  err = propagate.setArg(0, *buf);
  CHECKERR;
  err = propagate.setArg(1, (cl_int)width);
  CHECKERR;
  err = propagate.setArg(2, (cl_int)height);
  CHECKERR;
  err = propagate.setArg(3, chan);
  CHECKERR;

  err = queue->enqueueNDRangeKernel(startlabel, cl::NullRange,
                                    cl::NDRange(wsize, hsize),
                                    cl::NDRange(wgw, wgh));
  CHECKERR;

  while (true) {
    // CPU-GPU sync, sadly
    queue->enqueueReadBuffer(chan, CL_TRUE, 0, 1, &changed);
    if (changed == false) {
      break;
    }
    changed = false;
    queue->enqueueWriteBuffer(chan, CL_FALSE, 0, 1, &changed);
    queue->enqueueNDRangeKernel(propagate, cl::NullRange,
                                cl::NDRange(wsize, hsize),
                                cl::NDRange(wgw, wgh));
  }
}

void GPUUnionFind_Localer::execute() {
  cl_int err;

  const int wgw = 32;
  const int wgh = 8;
  const int wsize = round_to_nearest(width, wgw);
  const int hsize = round_to_nearest(height, wgh);

  cl::Kernel startlabel(*program, "label_with_id", &err);
  CHECKERR;
  cl::Kernel localer(*program, "solve_locally_plus", &err);
  CHECKERR;
  cl::Kernel propagate(*program, "union_find", &err);
  CHECKERR;

  err = startlabel.setArg(0, *buf);
  CHECKERR;
  err = startlabel.setArg(1, (cl_int)width);
  CHECKERR;
  err = startlabel.setArg(2, (cl_int)height);
  CHECKERR;

  err = localer.setArg(0, *buf);
  CHECKERR;
  err = localer.setArg(1, (cl_int)width);
  CHECKERR;
  err = localer.setArg(2, (cl_int)height);
  CHECKERR;

  char changed = 1;
  cl::Buffer chan(*context, CL_MEM_READ_WRITE, (size_t)1, nullptr, &err);
  queue->enqueueWriteBuffer(chan, CL_FALSE, 0, 1, &changed);

  err = propagate.setArg(0, *buf);
  CHECKERR;
  err = propagate.setArg(1, (cl_int)width);
  CHECKERR;
  err = propagate.setArg(2, (cl_int)height);
  CHECKERR;
  err = propagate.setArg(3, chan);
  CHECKERR;

  err = queue->enqueueNDRangeKernel(startlabel, cl::NullRange,
                                    cl::NDRange(wsize, hsize),
                                    cl::NDRange(wgw, wgh));
  CHECKERR;

  while (true) {
    // CPU-GPU sync, sadly
    queue->enqueueReadBuffer(chan, CL_TRUE, 0, 1, &changed);
    if (changed == false) {
      break;
    }
    changed = false;
    queue->enqueueWriteBuffer(chan, CL_FALSE, 0, 1, &changed);
    queue->enqueueNDRangeKernel(localer, cl::NullRange,
                                cl::NDRange(wsize, hsize),
                                cl::NDRange(wgw, wgh));
    queue->enqueueNDRangeKernel(propagate, cl::NullRange,
                                cl::NDRange(wsize, hsize),
                                cl::NDRange(wgw, wgh));
  }
}

void GPUUnionFind_Oneshot::execute() {
  cl_int err;

  const int wgw = 32;
  const int wgh = 8;
  const int wsize = round_to_nearest(width, wgw);
  const int hsize = round_to_nearest(height, wgh);

  cl::Kernel startlabel(*program, "label_with_id", &err);
  CHECKERR;
  cl::Kernel localer(*program, "plus_once_locally", &err);
  CHECKERR;
  cl::Kernel propagate(*program, "union_find", &err);
  CHECKERR;

  err = startlabel.setArg(0, *buf);
  CHECKERR;
  err = startlabel.setArg(1, (cl_int)width);
  CHECKERR;
  err = startlabel.setArg(2, (cl_int)height);
  CHECKERR;

  err = localer.setArg(0, *buf);
  CHECKERR;
  err = localer.setArg(1, (cl_int)width);
  CHECKERR;
  err = localer.setArg(2, (cl_int)height);
  CHECKERR;

  char changed = 1;
  cl::Buffer chan(*context, CL_MEM_READ_WRITE, (size_t)1, nullptr, &err);
  queue->enqueueWriteBuffer(chan, CL_FALSE, 0, 1, &changed);

  err = propagate.setArg(0, *buf);
  CHECKERR;
  err = propagate.setArg(1, (cl_int)width);
  CHECKERR;
  err = propagate.setArg(2, (cl_int)height);
  CHECKERR;
  err = propagate.setArg(3, chan);
  CHECKERR;

  err = queue->enqueueNDRangeKernel(startlabel, cl::NullRange,
                                    cl::NDRange(wsize, hsize),
                                    cl::NDRange(wgw, wgh));
  CHECKERR;

  while (true) {
    // CPU-GPU sync, sadly
    queue->enqueueReadBuffer(chan, CL_TRUE, 0, 1, &changed);
    if (changed == false) {
      break;
    }
    changed = false;
    queue->enqueueWriteBuffer(chan, CL_FALSE, 0, 1, &changed);
    queue->enqueueNDRangeKernel(localer, cl::NullRange,
                                cl::NDRange(wsize, hsize),
                                cl::NDRange(wgw, wgh));
    queue->enqueueNDRangeKernel(propagate, cl::NullRange,
                                cl::NDRange(wsize, hsize),
                                cl::NDRange(wgw, wgh));
  }
}

void GPULineEditing::execute() {
  cl_int err;

  const int wgs = 32;
  const int wsize = round_to_nearest(width, wgs);
  const int hsize = round_to_nearest(height, wgs);

  cl::Kernel startlabel(*program, "label_with_id", &err);
  CHECKERR;
  cl::Kernel up(*program, "lineedit_up", &err);
  CHECKERR;
  cl::Kernel down(*program, "lineedit_down", &err);
  CHECKERR;
  cl::Kernel left(*program, "lineedit_left", &err);
  CHECKERR;
  cl::Kernel right(*program, "lineedit_right", &err);
  CHECKERR;

  err = startlabel.setArg(0, *buf);
  CHECKERR;
  err = startlabel.setArg(1, (cl_uint)width);
  CHECKERR;
  err = startlabel.setArg(2, (cl_uint)height);
  CHECKERR;

  char changed = 1;
  cl::Buffer chan(*context, CL_MEM_READ_WRITE, (size_t)1, nullptr, &err);
  queue->enqueueWriteBuffer(chan, CL_TRUE, 0, 1, &changed);

  err = up.setArg(0, *buf);
  CHECKERR;
  err = up.setArg(1, (cl_int)width);
  CHECKERR;
  err = up.setArg(2, (cl_int)height);
  CHECKERR;
  err = up.setArg(3, chan);
  CHECKERR;

  err = down.setArg(0, *buf);
  CHECKERR;
  err = down.setArg(1, (cl_int)width);
  CHECKERR;
  err = down.setArg(2, (cl_int)height);
  CHECKERR;
  err = down.setArg(3, chan);
  CHECKERR;

  err = left.setArg(0, *buf);
  CHECKERR;
  err = left.setArg(1, (cl_int)width);
  CHECKERR;
  err = left.setArg(2, (cl_int)height);
  CHECKERR;
  err = left.setArg(3, chan);
  CHECKERR;

  err = right.setArg(0, *buf);
  CHECKERR;
  err = right.setArg(1, (cl_int)width);
  CHECKERR;
  err = right.setArg(2, (cl_int)height);
  CHECKERR;
  err = right.setArg(3, chan);
  CHECKERR;

  err = queue->enqueueNDRangeKernel(
      startlabel, cl::NullRange,
      cl::NDRange(round_to_nearest(width, 16), round_to_nearest(height, 16)),
      cl::NDRange(16, 16));
  CHECKERR;

  while (true) {
    // CPU-GPU sync, sadly
    queue->enqueueReadBuffer(chan, CL_TRUE, 0, 1, &changed);
    if (changed == false) {
      break;
    }
    changed = false;
    queue->enqueueWriteBuffer(chan, CL_FALSE, 0, 1, &changed);
    queue->enqueueNDRangeKernel(right, cl::NullRange, cl::NDRange(hsize),
                                cl::NDRange(wgs));
    queue->enqueueNDRangeKernel(down, cl::NullRange, cl::NDRange(wsize),
                                cl::NDRange(wgs));
    queue->enqueueNDRangeKernel(left, cl::NullRange, cl::NDRange(hsize),
                                cl::NDRange(wgs));
    queue->enqueueNDRangeKernel(up, cl::NullRange, cl::NDRange(wsize),
                                cl::NDRange(wgs));
  }
}

void GPULines::execute() {
  cl_int err;

  const int wgs = 4;
  const int wsize = round_to_nearest(width, wgs);
  const int hsize = round_to_nearest(height, wgs);

  cl::Kernel startlabel(*program, "label_with_id", &err);
  CHECKERR;
  cl::Kernel right(*program, "lines_right", &err);
  CHECKERR;
  cl::Kernel up(*program, "lines_up", &err);
  CHECKERR;

  err = startlabel.setArg(0, *buf);
  CHECKERR;
  err = startlabel.setArg(1, (cl_uint)width);
  CHECKERR;
  err = startlabel.setArg(2, (cl_uint)height);
  CHECKERR;

  char changed = 1;
  cl::Buffer chan(*context, CL_MEM_READ_WRITE, (size_t)1, nullptr, &err);
  queue->enqueueWriteBuffer(chan, CL_TRUE, 0, 1, &changed);

  err = right.setArg(0, *buf);
  CHECKERR;
  err = right.setArg(1, (cl_int)width);
  CHECKERR;
  err = right.setArg(2, (cl_int)height);
  CHECKERR;
  err = right.setArg(3, chan);
  CHECKERR;

  err = up.setArg(0, *buf);
  CHECKERR;
  err = up.setArg(1, (cl_int)width);
  CHECKERR;
  err = up.setArg(2, (cl_int)height);
  CHECKERR;
  err = up.setArg(3, chan);
  CHECKERR;

  err = queue->enqueueNDRangeKernel(
      startlabel, cl::NullRange,
      cl::NDRange(round_to_nearest(width, 16), round_to_nearest(height, 16)),
      cl::NDRange(16, 16));
  CHECKERR;

  while (true) {
    // CPU-GPU sync, sadly
    queue->enqueueReadBuffer(chan, CL_TRUE, 0, 1, &changed);
    if (changed == false) {
      break;
    }
    changed = false;
    queue->enqueueWriteBuffer(chan, CL_FALSE, 0, 1, &changed);
    queue->enqueueNDRangeKernel(right, cl::NullRange, cl::NDRange(hsize),
                                cl::NDRange(wgs));
    queue->enqueueNDRangeKernel(up, cl::NullRange, cl::NDRange(wsize),
                                cl::NDRange(wgs));
  }
}

void GPURecursive::execute() {
  cl_int err;

  const int wgw = 32;
  const int wgh = 8;
  const int wsize = round_to_nearest(width, wgw);
  const int hsize = round_to_nearest(height, wgh);

  cl::Kernel startlabel(*program, "label_with_id", &err);
  CHECKERR;
  cl::Kernel propagate(*program, "recursively_win", &err);
  CHECKERR;

  err = startlabel.setArg(0, *buf);
  CHECKERR;
  err = startlabel.setArg(1, (cl_int)width);
  CHECKERR;
  err = startlabel.setArg(2, (cl_int)height);
  CHECKERR;

  char changed = 1;
  cl::Buffer chan(*context, CL_MEM_READ_WRITE, (size_t)1, nullptr, &err);
  queue->enqueueWriteBuffer(chan, CL_FALSE, 0, 1, &changed);

  err = propagate.setArg(0, *buf);
  CHECKERR;
  err = propagate.setArg(1, (cl_int)width);
  CHECKERR;
  err = propagate.setArg(2, (cl_int)height);
  CHECKERR;
  err = propagate.setArg(3, chan);
  CHECKERR;

  err = queue->enqueueNDRangeKernel(startlabel, cl::NullRange,
                                    cl::NDRange(wsize, hsize),
                                    cl::NDRange(wgw, wgh));
  CHECKERR;

  while (true) {
    // CPU-GPU sync, sadly
    queue->enqueueReadBuffer(chan, CL_TRUE, 0, 1, &changed);
    if (changed == false) {
      break;
    }
    changed = false;
    queue->enqueueWriteBuffer(chan, CL_FALSE, 0, 1, &changed);
    queue->enqueueNDRangeKernel(propagate, cl::NullRange,
                                cl::NDRange(wsize, hsize),
                                cl::NDRange(wgw, wgh));
  }
}
