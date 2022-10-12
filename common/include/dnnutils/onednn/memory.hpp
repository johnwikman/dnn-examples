/**
 * memory specific utilities for onednn
 */

#ifndef DNNUTILS_ONEDNN_MEMORY_HPP
#define DNNUTILS_ONEDNN_MEMORY_HPP

#include <cstdint>
#include <numeric>
#include <vector>

#include <dnnl.h>
#include <oneapi/dnnl/dnnl.hpp>

namespace dnnutils {
namespace onednn {

template<typename T>
inline constexpr T product(const std::vector<T> &vec)
{
    return std::accumulate(vec.cbegin(), vec.cend(), (T) 1, std::multiplies<T>());
}

template<typename T>
inline constexpr T sum(const std::vector<T> &vec)
{
    return std::accumulate(vec.cbegin(), vec.cend(), (T) 0, std::plus<T>());
}

void cpu_to_memory(const void *handle, dnnl::memory &mem, std::size_t size);
void cpu_to_memory(const void *handle, dnnl::memory &mem);
void memory_to_cpu(void *handle, dnnl::memory &mem, std::size_t size);
void memory_to_cpu(void *handle, dnnl::memory &mem);

} // namespace onednn
} // namespace dnnutils

#endif // DNNUTILS_ONEDNN_MEMORY_HPP
