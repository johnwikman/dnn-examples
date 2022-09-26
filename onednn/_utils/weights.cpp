/**
 * implements weight initializations
 */

#include <random>
#include <vector>

#include "weights.hpp"

namespace dnnutils {

void he_initialize(std::vector<float> &v)
{
    he_initialize(v.data(), v.size());
}

void he_initialize(float *weights, std::size_t len)
{
    std::random_device rd;
    std::mt19937 gen(rd());

    std::normal_distribution<float> d(0.0f, 0.001f);

    for (std::size_t i = 0; i < len; ++i)
        weights[i] = d(gen);
}

} // namespace dnnutils
