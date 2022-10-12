/**
 * Verifying gradients of the using fixed-size tests.
 */

#include <cmath>
#include <iostream>
#include <vector>

#include <dnnl.h>
#include <oneapi/dnnl/dnnl.hpp>

#include <dnnutils.hpp>

#include "test_softmax.hpp"

#define DEBUG(msgs...) dnnutils::log::debug_fl(__FILE__, __LINE__, msgs)
#define INFO(msgs...)  dnnutils::log::info_fl(__FILE__, __LINE__, msgs)
#define ERROR(msgs...) dnnutils::log::error_fl(__FILE__, __LINE__, msgs)


static void compute_stats(const std::vector<float>& agrad, const std::vector<float>& ngrad);


int main(int argc, char *argv[])
{
    dnnutils::log::add_output_stream(std::cout);
    dnnutils::log::enable_bash_color();

    auto dataset = dnnutils::dataset::mnist<float>::load_training("/mnt/_data");

    dnnl::engine eng(dnnl::engine::kind::cpu, 0);
    dnnl::stream engine_stream(eng);

    std::vector<float> onednn_values;
    std::vector<float> onednn_gradients;
    std::vector<float> numeric_values;
    std::vector<float> numeric_gradients;

    test_softmax_analytical(eng, dataset, onednn_values, onednn_gradients);
    test_softmax_numeric(dataset, numeric_values, numeric_gradients);

    INFO("----------- COMPARING VALUES -----------");
    compute_stats(onednn_values, numeric_values);

    INFO("--------- COMPARING GRADIENTS ----------");
    compute_stats(onednn_gradients, numeric_gradients);
}

/**
 * Computes stats between analytically computes gradients and the numerically
 * computed gradients.
 */
static void compute_stats(const std::vector<float>& agrad, const std::vector<float>& ngrad)
{
    INFO("Computing statistics");
    std::vector<float> ip_error(agrad.size());
    float value_min = 1e22;
    float value_max = -1e22;
    float value_bp_absavg = 0.0f;
    float value_fs_absavg = 0.0f;
    float error_avg = 0.0;
    float error_std = 0.0;
    float error_min = 1e22;
    float error_max = -1e22;
    for (size_t i = 0; i < ip_error.size(); ++i) {
        ip_error[i] = std::abs(ngrad[i] - agrad[i]);
        error_avg += ip_error[i];
        error_min = std::min(error_min, ip_error[i]);
        error_max = std::max(error_max, ip_error[i]);
        value_bp_absavg += std::abs(agrad[i]);
        value_fs_absavg += std::abs(ngrad[i]);
        value_min = std::min(value_min, agrad[i]);
        value_min = std::min(value_min, ngrad[i]);
        value_max = std::max(value_max, agrad[i]);
        value_max = std::max(value_max, ngrad[i]);
    }
    value_bp_absavg /= (float) agrad.size();
    value_fs_absavg /= (float) ngrad.size();
    error_avg /= (float) ip_error.size();
    for (size_t i = 0; i < ip_error.size(); ++i) {
        float diff = ip_error[i] - error_avg;
        error_std += diff*diff;
    }
    error_std /= ((float) ip_error.size() - 1.0f);
    error_std = std::sqrt(error_std);
    INFO("  error_avg: ", error_avg);
    INFO("  error_std: ", error_std);
    INFO("  error_min: ", error_min);
    INFO("  error_max: ", error_max);
    INFO("  value_bp_absavg: ", value_bp_absavg);
    INFO("  value_fs_absavg: ", value_fs_absavg);
    INFO("  value_min: ", value_min);
    INFO("  value_max: ", value_max);
}


