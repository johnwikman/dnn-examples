/**
 * test_inner_product.cpp
 */

#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <dnnl.h>
#include <oneapi/dnnl/dnnl.hpp>

#include <dnnutils.hpp>
#include <dnnutils/onednn.hpp>

#include "test_inner_product.hpp"

#define DEBUG(msgs...) dnnutils::log::debug_fl(__FILE__, __LINE__, msgs)
#define INFO(msgs...)  dnnutils::log::info_fl(__FILE__, __LINE__, msgs)
#define ERROR(msgs...) dnnutils::log::error_fl(__FILE__, __LINE__, msgs)

/* Comment out this line to disable debug prints
#undef DEBUG
#define DEBUG(...)
// */

std::vector<float> test_inner_product_backprop(dnnl::engine &engine, const dnnutils::labelled_imgset<float> &dataset)
{
    using tag = dnnl::memory::format_tag;
    using dt = dnnl::memory::data_type;

    const int N = 256;
    const int H = dataset.H();
    const int W = dataset.W();
    const int C = dataset.C();

    const int O = 10;

    dnnl::stream stream(engine);

    DEBUG("Setting up forward and backward passes");
    std::vector<dnnl::primitive> net_fwd;
    std::vector<std::unordered_map<int, dnnl::memory>> net_fwd_args;
    std::vector<dnnl::primitive> net_bwd;
    std::vector<std::unordered_map<int, dnnl::memory>> net_bwd_args;

    DEBUG("Setting up memory descriptors");
    //auto md_input = dnnl::memory::desc({N, C, H, W}, dt::f32, tag::nhwc);
    //auto md_fc1_w_user = dnnl::memory::desc({O, C, H, W}, dt::f32, tag::oihw);
    //auto md_fc1_w = dnnl::memory::desc({O, C, H, W}, dt::f32, tag::any);
    auto md_input = dnnl::memory::desc({N, C*H*W}, dt::f32, tag::nc);
    auto md_fc1_w_user = dnnl::memory::desc({O, C*H*W}, dt::f32, tag::ab);
    auto md_fc1_w = dnnl::memory::desc({O, C*H*W}, dt::f32, tag::any);
    auto md_fc1_b = dnnl::memory::desc({O}, dt::f32, tag::a);
    auto md_fc1_out = dnnl::memory::desc({N, O}, dt::f32, tag::nc);

    DEBUG("Initializing weights");
    std::vector<float> fc1_w(dnnutils::onednn::product(md_fc1_w_user.dims()));
    std::vector<float> fc1_b(dnnutils::onednn::product(md_fc1_b.dims()));
    for (std::size_t i = 0; i < fc1_w.size(); ++i)
        fc1_w[i] = std::cos((float) i);
    for (std::size_t i = 0; i < fc1_b.size(); ++i)
        fc1_b[i] = std::cos((float) i);

    DEBUG("FC1");
    auto mem_input = dnnl::memory(md_input, engine);
    auto mem_fc1_w_user = dnnl::memory(md_fc1_w_user, engine);
    auto mem_fc1_b = dnnl::memory(md_fc1_b, engine);
    auto mem_fc1_out = dnnl::memory(md_fc1_out, engine);

    auto fc1_d = dnnl::inner_product_forward::desc(
        dnnl::prop_kind::forward_training,
        md_input, md_fc1_w, md_fc1_b, md_fc1_out
    );
    //dnnl::post_ops fc1_post_ops;
    //fc1_post_ops.append_eltwise(1.0f, dnnl::algorithm::eltwise_relu, 0.0f, 0.0f);
    //dnnl::primitive_attr fc1_attr;
    //fc1_attr.set_post_ops(fc1_post_ops);
    auto fc1_pd = dnnl::inner_product_forward::primitive_desc(
        fc1_d, engine
    );

    auto mem_fc1_w = mem_fc1_w_user;
    if (fc1_pd.weights_desc() != mem_fc1_w_user.get_desc()) {
        mem_fc1_w = dnnl::memory(fc1_pd.weights_desc(), engine);
        net_fwd.push_back(dnnl::reorder(mem_fc1_w_user, mem_fc1_w));
        net_fwd_args.push_back({
            {DNNL_ARG_FROM, mem_fc1_w_user},
            {DNNL_ARG_TO,   mem_fc1_w}
        });
    }

    net_fwd.push_back(dnnl::inner_product_forward(fc1_pd));
    net_fwd_args.push_back({
        {DNNL_ARG_SRC,     mem_input},
        {DNNL_ARG_WEIGHTS, mem_fc1_w},
        {DNNL_ARG_BIAS,    mem_fc1_b},
        {DNNL_ARG_DST,     mem_fc1_out}
    });

    DEBUG("SoftMax (in-place)");
    auto softmax_d = dnnl::softmax_forward::desc(
        dnnl::prop_kind::forward_training,
        md_fc1_out, 1
    );
    auto softmax_pd = dnnl::softmax_forward::primitive_desc(softmax_d, engine);
    net_fwd.push_back(dnnl::softmax_forward(softmax_pd));
    net_fwd_args.push_back({
        {DNNL_ARG_SRC, mem_fc1_out},
        {DNNL_ARG_DST, mem_fc1_out}
    });

    DEBUG("Loss Memory");
    auto mem_loss = dnnl::memory(md_fc1_out, engine);

    DEBUG("Backward SoftMax (in-place)");
    auto mem_diff_fc1_out = mem_loss; //dnnl::memory(md_fc1_out, engine);

    auto bwd_softmax_d = dnnl::softmax_backward::desc(
        md_fc1_out, md_fc1_out, 1
    );
    auto bwd_softmax_pd = dnnl::softmax_backward::primitive_desc(
        bwd_softmax_d, engine, softmax_pd
    );
    net_bwd.push_back(dnnl::softmax_backward(bwd_softmax_pd));
    net_bwd_args.push_back({
        {DNNL_ARG_DST,      mem_fc1_out},
        {DNNL_ARG_DIFF_DST, mem_diff_fc1_out},
        {DNNL_ARG_DIFF_SRC, mem_diff_fc1_out}
    });

    DEBUG("Backward FC1 (weights)");
    auto mem_diff_fc1_w_user = dnnl::memory(md_fc1_w_user, engine);
    auto mem_diff_fc1_b = dnnl::memory(md_fc1_b, engine);

    auto bwd_fc1_d = dnnl::inner_product_backward_weights::desc(
        md_input, md_fc1_w, md_fc1_b, md_fc1_out
    );
    auto bwd_fc1_pd = dnnl::inner_product_backward_weights::primitive_desc(
        bwd_fc1_d, engine, fc1_pd // <-- forward primitive_desc
    );

    if (bwd_fc1_pd.diff_bias_desc() != mem_diff_fc1_b.get_desc())
        DEBUG("BIAS ERROR");
    if (bwd_fc1_pd.diff_dst_desc() != mem_diff_fc1_out.get_desc())
        DEBUG("DEST ERROR");

    auto mem_bwd_input = mem_input;
    if (bwd_fc1_pd.src_desc() != mem_input.get_desc()) {
        DEBUG("???");
        mem_bwd_input = dnnl::memory(bwd_fc1_pd.src_desc(), engine);
        net_bwd.push_back(dnnl::reorder(mem_input, mem_bwd_input));
        net_bwd_args.push_back({
            {DNNL_ARG_FROM, mem_input},
            {DNNL_ARG_TO,   mem_bwd_input}
        });
    }

    net_bwd.push_back(dnnl::inner_product_backward_weights(bwd_fc1_pd));
    net_bwd_args.push_back({
        {DNNL_ARG_SRC,          mem_bwd_input},
        //{DNNL_ARG_DIFF_WEIGHTS, ???}, // wait with this
        {DNNL_ARG_DIFF_BIAS,    mem_diff_fc1_b},
        {DNNL_ARG_DIFF_DST,     mem_diff_fc1_out}
    });

    auto mem_diff_fc1_w = mem_diff_fc1_w_user;
    if (bwd_fc1_pd.diff_weights_desc() != mem_diff_fc1_w_user.get_desc()) {
        mem_diff_fc1_w = dnnl::memory(bwd_fc1_pd.diff_weights_desc(), engine);

        net_bwd_args.back().insert({DNNL_ARG_DIFF_WEIGHTS, mem_diff_fc1_w});

        net_bwd.push_back(dnnl::reorder(mem_diff_fc1_w, mem_diff_fc1_w_user));
        net_bwd_args.push_back({
            {DNNL_ARG_FROM, mem_diff_fc1_w},
            {DNNL_ARG_TO,   mem_diff_fc1_w_user}
        });
    } else {
        net_bwd_args.back().insert({DNNL_ARG_DIFF_WEIGHTS, mem_diff_fc1_w});
    }

    DEBUG("Loading MNIST data");
    std::vector<float> data_input(N*H*W*C);
    std::vector<float> data_output(N*O);
    std::vector<float> data_loss(N*O);
    std::vector<uint8_t> data_labels(N);
    for (int i = 0; i < N; ++i) {
        const float* data = dataset.data(i);
        int offset = i*C*H*W;
        for (int j = 0; j < C*H*W; ++j)
            data_input[offset + j] = data[j];

        data_labels[i] = dataset.label(i);
    }

    DEBUG("Copying over memory");
    dnnutils::onednn::cpu_to_memory(data_input.data(), mem_input);
    dnnutils::onednn::cpu_to_memory(fc1_w.data(), mem_fc1_w_user);
    dnnutils::onednn::cpu_to_memory(fc1_b.data(), mem_fc1_b);

    DEBUG("Running forward pass");
    for (std::size_t i = 0; i < net_fwd.size(); ++i)
        net_fwd.at(i).execute(stream, net_fwd_args.at(i));

    stream.wait();

    DEBUG("Computing loss");
    dnnutils::onednn::memory_to_cpu(data_output.data(), mem_fc1_out);
    // Cross entropy loss
    for (size_t i = 0; i < N; ++i) {
        for (uint8_t j = 0; j < (uint8_t) O; ++j) {
            // batch: i, prediction: j
            if (data_labels[i] == j)
                data_loss[i*O + j] = (-1.0f / data_output[i*O + j]) / ((float) N);
            else
                data_loss[i*O + j] = 0.0f;
        }
    }
    /* // Mean loss
    for (size_t i = 0; i < N; ++i) {
        for (uint8_t j = 0; j < (uint8_t) O; ++j) {
            // batch: i, prediction: j
            if (data_labels[i] == j)
                data_loss[i*O + j] = data_output[i*O + j] / ((float) N);
            else
                data_loss[i*O + j] = 0.0f;
        }
    } // */

    dnnutils::onednn::cpu_to_memory(data_loss.data(), mem_loss);

    DEBUG("Running backward pass");
    for (std::size_t i = 0; i < net_bwd.size(); ++i)
        net_bwd.at(i).execute(stream, net_bwd_args.at(i));

    stream.wait();

    DEBUG("Retrieving fc1_w gradient data");
    std::vector<float> fc1_w_grad(dnnutils::onednn::product(md_fc1_w_user.dims()));
    dnnutils::onednn::memory_to_cpu(fc1_w_grad.data(), mem_diff_fc1_w_user);

    return fc1_w_grad;
}


// FIXED STEP GRADIENTS!

// computes y = ReLU(Wx+b) over the batch size N
static void calc_relu_inner_product(std::vector<float> &y,
                                    const std::vector<float> &w,
                                    const std::vector<float> &x,
                                    const std::vector<float> &b,
                                    int N, int indim, int odim)
{
    for (int n = 0; n < N; ++n) {
        for (int i = 0; i < odim; ++i) {
            float yval = b[i];
            for (int j = 0; j < indim; ++j) {
                yval += w[i*indim + j] *
                        x[n*indim + j];
            }
            // ReLU
            //if (yval < 0.0f)
            //    yval = 0.0f;

            y[n*odim + i] = yval;
        }
    }
}

// computes y = mean(-log(SoftMax(x) * 1-hot(labels)))
static float calc_mean_softmax_cross_entropy_loss(const std::vector<float> &x,
                                                  const std::vector<uint8_t> &labels,
                                                  int N, int odim)
{
    std::vector<float> softmax_buf(odim);
    float total_sum = 0.0;
    for (int n = 0; n < N; ++n) {
        float softmax_sum = 0.0;
        for (int i = 0; i < odim; ++i) {
            softmax_buf[i] = std::exp(x[n*odim + i]);
            softmax_sum += softmax_buf[i];
        }
        total_sum += -std::log(softmax_buf[labels[n]] / softmax_sum);
    }
    float avg_sum = total_sum / ((float) N);
    return avg_sum;
}

// computes y = mean(x[labels])
//static float calc_mean(const std::vector<float> &x,
//                       const std::vector<uint8_t> &labels,
//                       int N)
//{
//    float total_sum = 0.0;
//    for (int n = 0; n < N; ++n) {
//        total_sum += x[labels[n]];
//    }
//    float avg_sum = total_sum / ((float) N);
//    return avg_sum;
//}

std::vector<float> test_inner_product_fixedstep(const dnnutils::labelled_imgset<float> &dataset)
{
    const float h = 1e-5;
    DEBUG("using h = ", h);

    const int N = 256;
    const int C = dataset.C();
    const int H = dataset.H();
    const int W = dataset.W();

    const int O = 10;

    DEBUG("Allocating buffers");
    std::vector<float> input_buf(N*C*H*W);
    std::vector<float> output_buf(N*O);
    std::vector<uint8_t> labels_buf(N);

    DEBUG("Loading MNIST data");
    for (int i = 0; i < N; ++i) {
        const float* data = dataset.data(i);
        int offset = i*C*H*W;
        for (int j = 0; j < C*H*W; ++j)
            input_buf[offset + j] = data[j];

        labels_buf[i] = dataset.label(i);
        if (labels_buf[i] > 9)
            DEBUG("Assert failed on index ", i);
    }

    DEBUG("Initializing weights");
    std::vector<float> fc1_w(O*C*H*W);
    std::vector<float> fc1_b(O);
    for (std::size_t i = 0; i < fc1_w.size(); ++i)
        fc1_w[i] = std::cos((float) i);
    for (std::size_t i = 0; i < fc1_b.size(); ++i)
        fc1_b[i] = std::cos((float) i);

    DEBUG("Setting up gradient vector");
    std::vector<float> fc1_w_grad(O*H*W*C);

    calc_relu_inner_product(output_buf, fc1_w, input_buf, fc1_b, N, H*W*C, O);
    float reference = calc_mean_softmax_cross_entropy_loss(output_buf, labels_buf, N, O);
    //float reference = calc_mean(output_buf, labels_buf, N);

    for (std::size_t i = 0; i < fc1_w.size(); ++i) {
        if (i % 256 == 0)
            INFO("(i: ", i, "/", fc1_w.size(), ")");

        fc1_w[i] += h;

        calc_relu_inner_product(output_buf, fc1_w, input_buf, fc1_b, N, H*W*C, O);
        float value = calc_mean_softmax_cross_entropy_loss(output_buf, labels_buf, N, O);
        //float value = calc_mean(output_buf, labels_buf, N);

        fc1_w_grad[i] = (value - reference) / h;

        // Restore h
        fc1_w[i] -= h;
    }

    return fc1_w_grad;
}
