#include <CL/cl.hpp>
#include <fstream>
#include <iostream>
#include <iomanip>
#include "Image.h"
#include "Strategy.h"
#include "LabelData.h"
#include <chrono>
#include <sys/stat.h>

template <class T> void fail(T message, cl_int err = 0) {
  std::cerr << message;
  if (err) {
    std::cerr << " ERR #: " << err;
  }
  std::cerr << std::endl;
  exit(-1);
}

/**
 * Attempts to find a suitable context, and loads that.
 */
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

/**
 * Loads available devices.
 */
cl::Device load_device(cl::Context *context) {
  std::vector<cl::Device> devices = context->getInfo<CL_CONTEXT_DEVICES>();
  if (devices.size() == 0) {
    fail("Found no devices");
  }

  // Potentially compare available devices.
  return devices[0];
}

/**
 * If build fails, print the compilation output and exit.
 */
void checkBuildErr(cl_int err, cl::Device *d, cl::Program *p) {
  if (err != CL_SUCCESS) {
    std::cerr << "ERROR: Building OpenCL program failed!\n";
    std::string log;
    p->getBuildInfo(*d, (cl_program_build_info)CL_PROGRAM_BUILD_LOG, &log);
    std::cerr << log << std::endl;
    exit(-1);
  }
}

/**
 * Tries to load the opencl program.
 */
cl::Program load_cl_program(cl::Context *context, cl::Device *device) {
  std::ifstream file("kernel.cl");
  if (!file) {
    std::cerr << "Kernel source file not opened correctly" << std::endl;
    exit(-1);
  }
  std::string source{std::istreambuf_iterator<char>(file),
                     std::istreambuf_iterator<char>()};

  cl_int err;
  cl::Program prog(*context, source, CL_TRUE, &err);
  checkBuildErr(err, device, &prog);
  return prog;
}

/**
 * Returns whether the labeldatas are equivalent, where labels may correspond to
 * a different number in the other labeldata.
 */
bool equivalent_result(LabelData *a, LabelData *b) { return true; } // TODO

/**
 * Checks for internal consistency of component labeling.
 */
bool valid_result(LabelData *l) {
  auto w = l->width;
  auto h = l->height;
  auto *d = l->data;
  for (unsigned int y = 0; y < h; ++y) {
    for (unsigned int x = 0; x < w; ++x) {
      auto curlabel = d[w * y + x];

      if (curlabel > 1 << 14) {
        std::cerr << "Labelnr above 1 << 14!" << std::endl;
      }

      if (curlabel == 1) {
        std::cerr << "Unlabeled pixel at x:" << x << " y:" << y << std::endl;
        return false;
      }

      if (curlabel != 0) {
        bool closefault = false;
        closefault |= x + 1 < w && d[w * (y) + (x + 1)] != 0 &&
                      d[w * (y) + (x + 1)] != curlabel;
        closefault |= x - 1 < w && d[w * (y) + (x - 1)] != 0 &&
                      d[w * (y) + (x - 1)] != curlabel;
        closefault |= y + 1 < h && d[w * (y + 1) + (x)] != 0 &&
                      d[w * (y + 1) + (x)] != curlabel;
        closefault |= y - 1 < h && d[w * (y - 1) + (x)] != 0 &&
                      d[w * (y - 1) + (x)] != curlabel;

        if (closefault) {
          std::cerr << "Connected components with different labels at x:" << x
                    << " y:" << y << std::endl;
          return false;
        }
      }
    }
  }

  // TODO: same-label disconnected components
  return true;
}

bool rgb_above_100(unsigned char r, unsigned char g, unsigned char b,
                   unsigned char) {
  if (r > 100 && g > 100 && b > 100) {
    return true;
  }
  return false;
}

RGBA max_if_nonzero(LabelData::label_type in) {
  RGBA out;
  if (in) {
    out.r = out.g = out.b = out.a = 255;
  } else {
    out.r = out.g = out.b = out.a = 0;
  }
  return out;
}

int main(int argc, const char *argv[]) {
  // Should RVO, want them as locals.
  cl::Context context = load_context();
  cl::Device device = load_device(&context);
  cl::Program program = load_cl_program(&context, &device);

  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " filename" << std::endl;
    return 0;
  }
  iml::Image rgba_image(argv[1]);
  if (!rgba_image) {
    fail("Image not loaded correctly, aborting.");
  }

  {
    int err = mkdir("out", 0777);
    if (err && errno != EEXIST) {
      fail("Couldn't creat output directory 'out'.");
    }
  }

  LabelData input(&rgba_image, rgb_above_100);
  LabelData correct(input);
  LabelData output(input);

  std::vector<Strategy *> strats;
  strats.push_back(new CPUOnePass);
  strats.push_back(new IdStrategy);
  strats.push_back(new GPUNeighbourPropagation);
  {
    auto &strat = strats[0];
    strat->prepare_gpu(&context, &device, &program, &input);
    strat->execute(&correct);
    strat->clean_gpu();
  }

  std::cerr << "(Name of file)           -- (Name of strategy)       -- (Times "
               "in microseconds)"
            << std::endl;

  for (auto *strat : strats) {
    output = input;
    strat->prepare_gpu(&context, &device, &program, &output);
    auto start = std::chrono::high_resolution_clock::now();
    strat->execute(&output);
    auto end = std::chrono::high_resolution_clock::now();
    strat->clean_gpu();
    auto ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                  .count();
    std::cout << std::left << std::setw(24) << argv[1] << " -- "
              << std::setw(24) << strat->name() << " -- " << std::setw(10) << ms
              << std::endl;
#ifndef NDEBUG
    if (!valid_result(&output)) {
      std::cerr << "Strategy returned an invalid labeling" << std::endl;
    }
    if (!equivalent_result(&correct, &output)) {
      std::cerr << "Strategy returned an unexpected labeling." << std::endl;
    }
#endif /* NDEBUG */

#ifndef NDEBUG
    // Write to file
    iml::Image out(output.width, output.height);
    output.copy_to_image(out.data, max_if_nonzero);
    std::string outname =
        "out/" + std::string(argv[1]) + " - " + strat->name() + ".png";
    iml::writepng(outname, &out);
#endif /* NDEBUG */
  }

  for (auto *strat : strats) {
    delete strat;
  }
}
