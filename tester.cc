#include <iostream>
#include <iomanip>
#include <chrono>
#include <sys/stat.h>
#include <algorithm>

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
  #ifndef NDEBUG
  std::cerr << "Loaded input image into a LabelData" << std::endl;
  #endif /* NDEBUG */

  std::vector<Strategy *> strats;
  strats.push_back(new CPUOnePass);
  //strats.push_back(new GPUNeighbourPropagation);
  //strats.push_back(new GPUPlusPropagation);
  strats.push_back(new GPULineEditing);
  //strats.push_back(new GPUKR);

  strats[0]->copy_to(&input, &context, &program, &queue);
  strats[0]->execute();
  LabelData correct = strats[0]->copy_from();

  // Ensures kernel and queue is ready, as they would only be created once in
  // a usual program.
  for (auto &strat : strats) {
    // Make warmup not be exactly the same as input for risk of
    // optimization. A clear image is safe for labeling.
    LabelData warmup(input);
    warmup.clear();

    #ifndef NDEBUG
    std::cerr << "Warmup with: " << strat->name() << std::endl;
    #endif /* NDEBUG */
    strat->copy_to(&warmup, &context, &program, &queue);
    strat->execute();
    strat->copy_from();
  }

  std::cerr << "(Name of file)                   -- (Name of strategy)         "
               "      -- (Times "
               "in microseconds)";
#ifndef NDEBUG
  std::cerr << " -- (Times with prep/cleanup)";
#endif /* NDEBUG */
  std::cerr << std::endl;

  for (auto *strat : strats) {
    auto startwithprep = std::chrono::high_resolution_clock::now();
    strat->copy_to(&input, &context, &program, &queue);

    auto start = std::chrono::high_resolution_clock::now();
    strat->execute();
    auto end = std::chrono::high_resolution_clock::now();

    LabelData output = strat->copy_from();
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
    output.copy_to_image(out.data, mod8);
    std::string cleaninput = argv[1];
    std::replace( cleaninput.begin(), cleaninput.end(), '/', '-');
    std::string outname =
        "out/" + cleaninput + " - " + strat->name() + ".png";
    iml::writepng(outname, &out);
#endif /* NDEBUG */
  }

  for (auto *strat : strats) {
    delete strat;
  }
}
