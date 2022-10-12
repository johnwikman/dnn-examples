/**
 * operations.hpp
 */

#ifndef DNNUTILS_OPERATIONS_HPP
#define DNNUTILS_OPERATIONS_HPP

#include <cstdint>
#include <functional>
#include <numeric>
#include <vector>

#include <dnnl.hpp>
#include <oneapi/dnnl/dnnl.hpp>

namespace dnnutils {

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

inline void cpu_to_dnnl_memory(const void *handle, dnnl::memory &mem, std::size_t size)
{
    const std::uint8_t *src = static_cast<const std::uint8_t *>(handle);
    std::uint8_t *dst = static_cast<std::uint8_t *>(mem.get_data_handle());
    if (dst == nullptr)
        throw std::runtime_error("mem.get_data_handle() is null");
    for (std::size_t i = 0; i < size; ++i)
        dst[i] = src[i];
}

inline void cpu_to_dnnl_memory(const void *handle, dnnl::memory &mem)
{
    cpu_to_dnnl_memory(handle, mem, mem.get_desc().get_size());
}

inline void cpu_from_dnnl_memory(void *handle, dnnl::memory &mem, std::size_t size)
{
    const std::uint8_t *src = static_cast<const std::uint8_t *>(mem.get_data_handle());
    std::uint8_t *dst = static_cast<std::uint8_t *>(handle);
    if (src == nullptr)
        throw std::runtime_error("mem.get_data_handle() is null");
    for (std::size_t i = 0; i < size; ++i)
        dst[i] = src[i];
}

inline void cpu_from_dnnl_memory(void *handle, dnnl::memory &mem)
{
    cpu_from_dnnl_memory(handle, mem, mem.get_desc().get_size());
}

} // namespace dnnutils

#endif // DNNUTILS_OPERATIONS_HPP

