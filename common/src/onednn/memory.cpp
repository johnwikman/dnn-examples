/**
 * memory.hpp OneDNN memory related utilities.
 */

#include <cstdint>
#include <vector>

#include <dnnl.h>
#include <oneapi/dnnl/dnnl.hpp>

#include <dnnutils/onednn/memory.hpp>

namespace dnnutils {
namespace onednn {

// TODO: This assumes we are using openmp

void cpu_to_memory(const void *handle, dnnl::memory &mem, std::size_t size)
{
    const std::uint8_t *src = static_cast<const std::uint8_t *>(handle);
    std::uint8_t *dst = static_cast<std::uint8_t *>(mem.get_data_handle());
    if (dst == nullptr)
        throw std::runtime_error("mem.get_data_handle() is null");
    for (std::size_t i = 0; i < size; ++i)
        dst[i] = src[i];
}

void cpu_to_memory(const void *handle, dnnl::memory &mem)
{
    cpu_to_memory(handle, mem, mem.get_desc().get_size());
}

void memory_to_cpu(void *handle, dnnl::memory &mem, std::size_t size)
{
    const std::uint8_t *src = static_cast<const std::uint8_t *>(mem.get_data_handle());
    std::uint8_t *dst = static_cast<std::uint8_t *>(handle);
    if (src == nullptr)
        throw std::runtime_error("mem.get_data_handle() is null");
    for (std::size_t i = 0; i < size; ++i)
        dst[i] = src[i];
}

void memory_to_cpu(void *handle, dnnl::memory &mem)
{
    memory_to_cpu(handle, mem, mem.get_desc().get_size());
}

} // namespace onednn
} // namespace dnnutils
