#include <CL/cl.hpp>
#include <iostream>
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

std::vector<cl::Device> load_devices(cl::Context *context) {
  std::vector<cl::Device> devices = context->getInfo<CL_CONTEXT_DEVICES>();
  if (devices.size() == 0) {
    fail("Found no devices");
  }
  return devices;
}

bool equivalent_result(LabelData *a, LabelData *b){ return true; } //TODO

bool valid_result(LabelData *l) { return true; } //TODO

bool rgb_above_100(unsigned char r, unsigned char g, unsigned char b,
                   unsigned char) {
  if (r > 100 && g > 100 && b > 100) {
    return true;
  }
  return false;
}

RGBA max_if_nonzero(LabelData::label_type in){
  RGBA out;
  if (in) {
    out.r = out.g = out.b = out.a = 255;
  } else {
    out.r = out.g = out.b = out.a = 0;
  }
  return out;
}

int main(int argc, const char *argv[]) {
  cl::Context context = load_context();
  std::vector<cl::Device> devices = load_devices(&context);

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

  std::vector<Strategy*> strats;
  strats.push_back(new IdStrategy);
  // TODO: Put a known good one in front
  {
    auto &strat = strats[0];
    strat->prepare_gpu(&context, &devices, &input);
    strat->execute(&correct);
    strat->clean_gpu();
  }

  for (auto *strat : strats) {
    strat->prepare_gpu(&context, &devices, &input);
    auto start = std::chrono::high_resolution_clock::now();
    strat->execute(&output);
    auto end = std::chrono::high_resolution_clock::now();
    strat->clean_gpu();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                  .count();
    std::cout << argv[1] << " -- " << strat->name() << ": " << ms << " ms."
              << std::endl;
    if (!valid_result(&output)) {
      std::cerr << "Strategy returned an invalid labeling" << std::endl;
    }
    if (!equivalent_result(&input, &output)) {
      std::cerr << "Strategy returned an unexpected labeling." << std::endl;
    }

    //Write to file
    iml::Image out(output.width, output.height);
    output.copy_to_image(out.data, max_if_nonzero);
    std::string outname = "out/" + strat->name() + ".png";
    iml::writepng(outname, &out);
  }
}
