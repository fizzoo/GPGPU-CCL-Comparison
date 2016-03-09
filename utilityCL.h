#ifndef UTILITYCL_H
#define UTILITYCL_H

#include <CL/cl.hpp>
#include <fstream>
#include "defines.h"

/**
 * Attempts to find a suitable context, and loads that.
 */
cl::Context load_context();

/**
 * Loads available devices.
 */
cl::Device load_device(cl::Context *context);

/**
 * If build fails, print the compilation output and exit.
 */
void checkBuildErr(cl_int err, cl::Device *d, cl::Program *p);

/**
 * Tries to load the opencl program.
 */
cl::Program load_cl_program(cl::Context *context, cl::Device *device);

/**
 * Creates a queue.
 */
cl::CommandQueue load_queue(cl::Context *context, cl::Device *device);

#endif /* end of include guard: UTILITYCL_H */
