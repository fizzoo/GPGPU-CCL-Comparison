#include <iostream>
#include <iomanip>
#include <chrono>
#include <sys/stat.h>

#include "Image.h"
#include "Strategy.h"
#include "LabelData.h"
#include "RGBAConversions.h"
#include "utilityCL.h"

int main(int argc, const char *argv[]) {
  // Should RVO, want them as locals.
  cl::Context context = load_context();
  cl::Device device = load_device(&context);
  cl::Program program = load_cl_program(&context, &device);
  cl::CommandQueue queue = load_queue(&context, &device);

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

  std::cerr << "(Name of file)                   -- (Name of strategy)         "
               "      -- (Times "
               "in microseconds)";
#ifndef NDEBUG
  std::cerr << " -- (Times with prep/cleanup)";
#endif /* NDEBUG */
  std::cerr << std::endl;

  // Ensures kernel and queue is ready, as they would only be created once in
  // a usual program.
  for (auto &strat : strats) {
    // Make warmup not be exactly the same as output for risk of
    // optimization, a clear image is safe for labeling.
    LabelData warmup(input);
    warmup.clear();

    strat->prepare_gpu(&context, &device, &program, &warmup);
    strat->execute(&warmup);
    strat->clean_gpu();
  }

  for (auto *strat : strats) {
    output = input;

    auto startwithprep = std::chrono::high_resolution_clock::now();
    strat->prepare_gpu(&context, &device, &program, &output);

    auto start = std::chrono::high_resolution_clock::now();
    strat->execute(&output);
    auto end = std::chrono::high_resolution_clock::now();

    strat->clean_gpu();
    auto endwithprep = std::chrono::high_resolution_clock::now();

    auto ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                  .count();
    auto mswithprep = std::chrono::duration_cast<std::chrono::microseconds>(
                          endwithprep - startwithprep)
                          .count();

    std::cout << std::left << std::setw(32) << argv[1] << " -- "
              << std::setw(32) << strat->name() << " -- " << std::setw(23)
              << ms;
#ifndef NDEBUG
    std::cout << " -- " << mswithprep;
#endif /* NDEBUG */
    std::cout << std::endl;

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
