/**
 * A small example with oneDNN, assuming OpenMP
 */

#include <iostream>

#include <dnnl.hpp>
#include <dnnl_debug.h>

void check_defines(void);

int main(int argc, char **argv)
{
    std::cout << "small_example.cpp" << std::endl;

    std::cout << "checking defines..." << std::endl;
    check_defines();
}

/**
 * Check oneDNN definitions
 */
void check_defines(void)
{
    std::cout << "DNNL_GPU_RUNTIME is " <<
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    "DNNL_RUNTIME_OCL"
#elif DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
    "DNNL_RUNTIME_SYCL"
#else
    "<unknown>"
#endif
    << std::endl;

    std::cout << "DNNL_CPU_THREADING_RUNTIME is " <<
#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_OMP
    "DNNL_RUNTIME_OMP"
#else
    "<unknown>"
#endif
    << std::endl;

    std::cout << "Using DNNL_WITH_SYCL? " <<
#ifdef DNNL_WITH_SYCL
    "Yes! " <<
#  if DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
    "... and it is also the CPU runtime!"
#  else
    "But it is not the CPU runtime."
#  endif
#else
    "no... :("
#endif
    << std::endl;

    std::cout << "DNNL_CPU_RUNTIME=" << DNNL_CPU_RUNTIME << std::endl;
}
