/**
 * test_softmax.cpp
 *
 * This takes a single image, produces a softmax, and the derivative for that
 * softmax. More specifically, differentiate w.r.t. -ln(SoftMax(Image)[0])
 * 
 * But we return the value for SoftMax(Image) for the entire batch
 */

#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <dnnl.h>
#include <oneapi/dnnl/dnnl.hpp>

#include <dnnutils.hpp>
#include <dnnutils/onednn.hpp>

#include "test_softmax.hpp"

#define DEBUG(msgs...) dnnutils::log::debug_fl(__FILE__, __LINE__, msgs)
#define INFO(msgs...)  dnnutils::log::info_fl(__FILE__, __LINE__, msgs)
#define ERROR(msgs...) dnnutils::log::error_fl(__FILE__, __LINE__, msgs)

/* Comment out this line to disable debug prints
#undef DEBUG
#define DEBUG(...)
// */

void test_softmax_analytical(dnnl::engine &engine,
                             const dnnutils::labelled_imgset<float> &dataset,
                             std::vector<float>& value_output,
                             std::vector<float>& diff_output)
{
    using tag = dnnl::memory::format_tag;
    using dt = dnnl::memory::data_type;

    const int N = 128;
    const int H = dataset.H();
    const int W = dataset.W();
    const int C = dataset.C();

    dnnl::stream stream(engine);

    DEBUG("Setting up forward and backward passes");
    std::vector<dnnl::primitive> net_fwd;
    std::vector<std::unordered_map<int, dnnl::memory>> net_fwd_args;
    std::vector<dnnl::primitive> net_bwd;
    std::vector<std::unordered_map<int, dnnl::memory>> net_bwd_args;

    DEBUG("Setting up memory descriptors");
    auto md_input = dnnl::memory::desc({N, C*H*W}, dt::f32, tag::nc);

    DEBUG("Setting up memory");
    auto mem_input = dnnl::memory(md_input, engine);

    DEBUG("SoftMax (in-place)");
    auto softmax_d = dnnl::softmax_forward::desc(
        dnnl::prop_kind::forward_training,
        md_input, 1
    );
    auto softmax_pd = dnnl::softmax_forward::primitive_desc(softmax_d, engine);
    net_fwd.push_back(dnnl::softmax_forward(softmax_pd));
    net_fwd_args.push_back({
        {DNNL_ARG_SRC, mem_input},
        {DNNL_ARG_DST, mem_input}
    });

    DEBUG("Loss Memory");
    auto mem_loss = dnnl::memory(md_input, engine);

    DEBUG("Backward SoftMax (in-place)");
    auto mem_diff_input = mem_loss;

    auto bwd_softmax_d = dnnl::softmax_backward::desc(
        md_input, md_input, 1
    );
    auto bwd_softmax_pd = dnnl::softmax_backward::primitive_desc(
        bwd_softmax_d, engine, softmax_pd
    );
    net_bwd.push_back(dnnl::softmax_backward(bwd_softmax_pd));
    net_bwd_args.push_back({
        {DNNL_ARG_DST,      mem_input},
        {DNNL_ARG_DIFF_DST, mem_diff_input},
        {DNNL_ARG_DIFF_SRC, mem_diff_input}
    });

    DEBUG("Loading MNIST data");
    std::vector<float> data_input(N*C*H*W);
    std::vector<float> data_loss(N*C*H*W);
    value_output = std::vector<float>(N*C*H*W);
    diff_output = std::vector<float>(N*C*H*W);
    for (int i = 0; i < N; ++i) {
        const float* data = dataset.data(i);
        int offset = i*C*H*W;
        for (int j = 0; j < C*H*W; ++j)
            data_input[offset + j] = data[j];
    }

    DEBUG("Copying over memory");
    dnnutils::onednn::cpu_to_memory(data_input.data(), mem_input);

    DEBUG("Running forward pass");
    for (std::size_t i = 0; i < net_fwd.size(); ++i)
        net_fwd.at(i).execute(stream, net_fwd_args.at(i));

    stream.wait();

    DEBUG("Computing loss");
    dnnutils::onednn::memory_to_cpu(value_output.data(), mem_input);
    // Cross entropy loss
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < (size_t) (C*H*W); ++j) {
            // batch: i, prediction: j
            if (j == 0)
                data_loss[i*(C*H*W) + j] = -std::log(value_output[i*(C*H*W) + j]) / ((float) N);
            else
                data_loss[i*(C*H*W) + j] = 0.0f;
        }
    }

    dnnutils::onednn::cpu_to_memory(data_loss.data(), mem_loss);

    DEBUG("Running backward pass");
    for (std::size_t i = 0; i < net_bwd.size(); ++i)
        net_bwd.at(i).execute(stream, net_bwd_args.at(i));

    stream.wait();

    DEBUG("Retrieving gradient data");
    dnnutils::onednn::memory_to_cpu(diff_output.data(), mem_diff_input);
}

static float calc_mean_softmax(const std::vector<float> &batch,
                               int N, int dim,
                               std::vector<float>& softmax_buf)
{
    float total_sum = 0.0;
    for (int n = 0; n < N; ++n) {
        float softmax_sum = 0.0;
        for (int i = 0; i < dim; ++i) {
            softmax_buf[dim*n + i] = std::exp(batch[dim*n + i]);
            softmax_sum += softmax_buf[dim*n + i];
        }
        total_sum += -std::log(softmax_buf[0] / softmax_sum);
    }
    float avg_sum = total_sum / ((float) N);
    return avg_sum;
}

static void calc_softmax(const std::vector<float> &batch,
                         int N, int dim,
                         std::vector<float>& softmax_buf)
{
    for (int n = 0; n < N; ++n) {
        float softmax_sum = 0.0;
        for (int i = 0; i < dim; ++i) {
            softmax_buf[dim*n + i] = std::exp(batch[dim*n + i]);
            softmax_sum += softmax_buf[dim*n + i];
        }
        for (int i = 0; i < dim; ++i)
            softmax_buf[dim*n + i] /= softmax_sum;
    }
}

void test_softmax_numeric(const dnnutils::labelled_imgset<float> &dataset,
                          std::vector<float>& value_output,
                          std::vector<float>& diff_output)
{
    const float h = 1e-5;
    DEBUG("using h = ", h);

    const int N = 128;
    const int dim = dataset.item_size();

    DEBUG("Allocating buffers");
    std::vector<float> input_buf(N*dim);
    value_output = std::vector<float>(N*dim);
    diff_output = std::vector<float>(N*dim);

    std::vector<float> softmax_buf(N*dim);

    DEBUG("Loading MNIST data");
    for (int i = 0; i < N; ++i) {
        const float* data = dataset.data(i);
        int offset = i*dim;
        for (int j = 0; j < dim; ++j)
            input_buf[offset + j] = data[j];
    }

    calc_softmax(input_buf, N, dim, value_output);

    float reference = calc_mean_softmax(input_buf, N, dim, softmax_buf);

    for (std::size_t i = 0; i < input_buf.size(); ++i) {
        if (i % 2048 == 0)
            INFO("(i: ", i, "/", input_buf.size(), ")");

        input_buf[i] += h;

        float value = calc_mean_softmax(input_buf, N, dim, softmax_buf);

        diff_output[i] = (value - reference) / h;

        // Restore h
        input_buf[i] -= h;
    }
}

