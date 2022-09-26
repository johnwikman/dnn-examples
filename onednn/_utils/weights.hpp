/**
 * weights.hpp
 * 
 * Initialize weights of a layer
 */

#ifndef DNNUTILS_WEIGHTS_HPP
#define DNNUTILS_WEIGHTS_HPP

#include <vector>

namespace dnnutils {

void he_initialize(std::vector<float> &v);
void he_initialize(float *weights, std::size_t len);

} // namespace dnnutils

#endif // DNNUTILS_WEIGHTS_HPP
