/**
 * decode.hpp - utilities for decoding data
 */

#ifndef DNNUTILS_TOOLS_DECODE_HPP
#define DNNUTILS_TOOLS_DECODE_HPP

#include <cstdint>

namespace dnnutils {
namespace tools {

/**
 * Decode the bytes at the provided address as 32-bit BE integer.
 */
static inline std::uint32_t frombe32(const std::byte* src)
{
    std::uint32_t v = (((uint8_t) src[0]) << 24)
                    | (((uint8_t) src[1]) << 16)
                    | (((uint8_t) src[2]) << 8)
                    | ((uint8_t) src[3]);
    return v;
}

}
} // namespace dnnutils


#endif //DNNUTILS_TOOLS_DECODE_HPP
