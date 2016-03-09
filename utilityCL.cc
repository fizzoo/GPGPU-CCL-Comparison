#include "utilityCL.h"

cl::Context load_context() {
  cl_int err;

  // Create the context. Iterates through the platforms and picks the first
  // one with a GPU, then creates a context from that.
  cl::Context context = cl::Context(CL_DEVICE_TYPE_GPU, NULL, NULL, NULL, &err);
  if (err) {
    // Fall back to CPU
    context = cl::Context(CL_DEVICE_TYPE_CPU, NULL, NULL, NULL, &err);
  }
  if (err) {
    fail("Couldn't find a platform/device.", err);
  }
  return context;
}

cl::Device load_device(cl::Context *context) {
  std::vector<cl::Device> devices = context->getInfo<CL_CONTEXT_DEVICES>();
  if (devices.size() == 0) {
    fail("Found no devices");
  }

  // Potentially compare available devices.
  return devices[0];
}

void checkBuildErr(cl_int err, cl::Device *d, cl::Program *p) {
  if (err != CL_SUCCESS) {
    std::cerr << "ERROR: Building OpenCL program failed!\n";
    std::string log;
    p->getBuildInfo(*d, (cl_program_build_info)CL_PROGRAM_BUILD_LOG, &log);
    std::cerr << log << std::endl;
    exit(err);
  }
}

cl::Program load_cl_program(cl::Context *context, cl::Device *device) {
  std::ifstream file("kernel.cl");
  if (!file) {
    fail("Kernel source file not opened correctly");
  }
  std::string source{std::istreambuf_iterator<char>(file),
                     std::istreambuf_iterator<char>()};

  cl_int err;
  cl::Program prog(*context, source, CL_TRUE, &err);
  checkBuildErr(err, device, &prog);
  return prog;
}

cl::CommandQueue load_queue(cl::Context *context, cl::Device *device) {
  cl_int err;

  cl::CommandQueue queue(*context, *device, 0, &err);
  if (err) {
    fail("Queue could not be opened correctly.", err);
  }

  return queue;
}
