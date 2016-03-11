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

  cl::Kernel startlabel(*program, "label_with_id", &err);
  CHECKERR;
  cl::Kernel propagate(*program, "neighbour_propagate", &err);
  CHECKERR;

  err = startlabel.setArg(0, *buf);
  CHECKERR;
  err = startlabel.setArg(1, (cl_uint)width);
  CHECKERR;

  char changed = 1;
  cl::Buffer chan(*context, CL_MEM_READ_WRITE, 1, nullptr, &err);
  queue->enqueueWriteBuffer(chan, CL_FALSE, 0, 1, &changed);

  err = propagate.setArg(0, *buf);
  CHECKERR;
  err = propagate.setArg(1, (cl_uint)width);
  CHECKERR;
  err = propagate.setArg(2, (cl_uint)height);
  CHECKERR;
  err = propagate.setArg(3, chan);
  CHECKERR;

  std::vector<cl::Event> events(1);
  std::vector<cl::Event> writtenevents(1);
  err = queue->enqueueNDRangeKernel(startlabel, cl::NullRange,
                                    cl::NDRange(width, height),
                                    cl::NDRange(1, 1), NULL, &events[0]);
  CHECKERR;

  while (true) {
    // CPU-GPU sync, sadly
    queue->enqueueReadBuffer(chan, CL_TRUE, 0, 1, &changed, &events, NULL);
    if (changed == false) {
      break;
    }
    changed = false;
    queue->enqueueWriteBuffer(chan, CL_FALSE, 0, 1, &changed, NULL,
                              &writtenevents[0]);
    queue->enqueueNDRangeKernel(propagate, cl::NullRange,
                                cl::NDRange(width, height), cl::NDRange(1, 1),
                                &writtenevents, &events[0]);
  }
}

void GPUPlusPropagation::execute() {
  cl_int err;

  int wsize, hsize;
  if (width % 16) {
    wsize = width + 16 - (width % 16);
  } else {
    wsize = width;
  }
  if (height % 8) {
    hsize = height + 8 - (height % 8);
  } else {
    hsize = height;
  }

  cl::Kernel startlabel(*program, "label_with_id", &err);
  CHECKERR;
  cl::Kernel propagate(*program, "plus_propagate", &err);
  CHECKERR;

  err = startlabel.setArg(0, *buf);
  CHECKERR;
  err = startlabel.setArg(1, (cl_uint)width);
  CHECKERR;

  char changed = 1;
  cl::Buffer chan(*context, CL_MEM_READ_WRITE, 1, nullptr, &err);
  queue->enqueueWriteBuffer(chan, CL_FALSE, 0, 1, &changed);

  err = propagate.setArg(0, *buf);
  CHECKERR;
  err = propagate.setArg(1, (cl_uint)width);
  CHECKERR;
  err = propagate.setArg(2, (cl_uint)height);
  CHECKERR;
  err = propagate.setArg(3, chan);
  CHECKERR;

  std::vector<cl::Event> events(1);
  std::vector<cl::Event> writtenevents(1);
  err = queue->enqueueNDRangeKernel(startlabel, cl::NullRange,
                                    cl::NDRange(wsize, hsize),
                                    cl::NDRange(16, 8), NULL, &events[0]);
  CHECKERR;

  while (true) {
    // CPU-GPU sync, sadly
    queue->enqueueReadBuffer(chan, CL_TRUE, 0, 1, &changed, &events, NULL);
    if (changed == false) {
      break;
    }
    changed = false;
    queue->enqueueWriteBuffer(chan, CL_FALSE, 0, 1, &changed, NULL,
                              &writtenevents[0]);
    queue->enqueueNDRangeKernel(propagate, cl::NullRange,
                                cl::NDRange(wsize, hsize), cl::NDRange(16, 8),
                                &writtenevents, &events[0]);
  }
}

void GPULineEditing::execute() {
  cl_int err;

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

  char changed = 1;
  cl::Buffer chan(*context, CL_MEM_READ_WRITE, 1, nullptr, &err);
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

  std::vector<cl::Event> event1(1);
  std::vector<cl::Event> event2(1);
  err = queue->enqueueNDRangeKernel(startlabel, cl::NullRange,
                                    cl::NDRange(width, height),
                                    cl::NDRange(1, 1), NULL, &event1[0]);
  CHECKERR;

  while (true) {
    // CPU-GPU sync, sadly
    queue->enqueueReadBuffer(chan, CL_TRUE, 0, 1, &changed, &event1, NULL);
    if (changed == false) {
      break;
    }
    changed = false;
    queue->enqueueWriteBuffer(chan, CL_FALSE, 0, 1, &changed, NULL, &event1[0]);
    queue->enqueueNDRangeKernel(right, cl::NullRange, cl::NDRange(height),
                                cl::NDRange(1), &event1, &event2[0]);
    queue->enqueueNDRangeKernel(down, cl::NullRange, cl::NDRange(width),
                                cl::NDRange(1), &event2, &event1[0]);
    queue->enqueueNDRangeKernel(left, cl::NullRange, cl::NDRange(height),
                                cl::NDRange(1), &event1, &event2[0]);
    queue->enqueueNDRangeKernel(up, cl::NullRange, cl::NDRange(width),
                                cl::NDRange(1), &event2, &event1[0]);
  }
}

void GPUKR::execute() {
    cl_int err;

    //preprocessing, init values and simple scan 
    cl::Kernel init(*program, "gpu_kr_init_phase", &err);
    CHECKERR;
    //preproccesing part 2, scan phase
    cl::Kernel scan(*program, "gpu_kr_scan_phase", &err);
    CHECKERR;
    //this one will itterate until wanted result is reached
    cl::Kernel propagate(*program, "gpu_label_mask_two", &err);
    CHECKERR;
    cl::Kernel link(*program, "gpu_kr_link_phase", &err);
    CHECKERR;
    cl::Kernel final_phases(*program, "gpu_kr_final_phases", &err);
    CHECKERR;

    err = init.setArg(0, *buf);
    CHECKERR;
    err = init.setArg(1, (cl_int)width);
    CHECKERR;
    err = init.setArg(2, (cl_int)height);
    CHECKERR;

    char iterate = 1;

    LABELTYPE *repLabels = new LABELTYPE[width * height];
    auto size = width * height * sizeof(LABELTYPE);
    cl::Buffer iter(*context, CL_MEM_READ_WRITE, 1, nullptr, &err);
    CHECKERR;
    cl::Buffer repLabelBuf(*context, CL_MEM_READ_WRITE, size, nullptr, &err);
    CHECKERR;

    err = queue->enqueueWriteBuffer(iter, CL_FALSE, 0, 1, &iterate);
    CHECKERR
    err = queue->enqueueWriteBuffer(repLabelBuf, CL_TRUE, 0, size, repLabels);
    CHECKERR;

    err = scan.setArg(0, *buf);
    CHECKERR;
    err = scan.setArg(1, (cl_int)width);
    CHECKERR;
    err = scan.setArg(2, (cl_int)height);
    CHECKERR;
    err = scan.setArg(3, repLabelBuf);
    CHECKERR;

    err = propagate.setArg(0, *buf);
    CHECKERR;
    err = propagate.setArg(1, (cl_int)width);
    CHECKERR;
    err = propagate.setArg(2, (cl_int)height);
    CHECKERR;
    err = propagate.setArg(3, repLabelBuf);
    CHECKERR;

    err = link.setArg(0, *buf);
    CHECKERR;
    err = link.setArg(1, (cl_int)width);
    CHECKERR;
    err = link.setArg(2, (cl_int)height);
    CHECKERR;
    err = link.setArg(3, repLabelBuf);
    CHECKERR;

    err = final_phases.setArg(0, *buf);
    CHECKERR;
    err = final_phases.setArg(1, (cl_int)width);
    CHECKERR;
    err = final_phases.setArg(2, (cl_int)height);
    CHECKERR;
    err = final_phases.setArg(3, iter);
    CHECKERR;
    err = final_phases.setArg(4, repLabelBuf);
    CHECKERR;



    std::vector<cl::Event> events(1);
    std::vector<cl::Event> writtenevents(1);

    err = queue->enqueueNDRangeKernel(init, cl::NullRange,
            cl::NDRange(width, height), 
            cl::NDRange(1, 1), NULL, &events[0]);
    CHECKERR;

    err = queue->enqueueNDRangeKernel(scan, cl::NullRange,
            cl::NDRange(width, height), 
            cl::NDRange(1, 1), NULL, &events[0]);
    CHECKERR;


    int i = 0;

    //since the kernel does't work yet just itterate a few times to compare
    //results
    while(i < 10) 
    {
        i++;
        queue->enqueueReadBuffer(iter, CL_TRUE, 0, 1, &iterate, &events, NULL);
        if (iterate == false) {
            break;
        }
        iterate = false;
        queue->enqueueWriteBuffer(iter, CL_FALSE, 0, 1, &iterate, NULL, &writtenevents[0]);
        queue->enqueueNDRangeKernel(propagate, cl::NullRange, 
                cl::NDRange(width, height), 
                cl::NDRange(1,1), 
                &writtenevents, &events[0]);

        queue->enqueueNDRangeKernel(link, cl::NullRange, 
                cl::NDRange(width, height), 
                cl::NDRange(1,1), 
                &writtenevents, &events[0]);

        queue->enqueueNDRangeKernel(final_phases, cl::NullRange, 
                cl::NDRange(width, height), 
                cl::NDRange(1,1), 
                &writtenevents, &events[0]);
    }

    delete repLabels;
}
