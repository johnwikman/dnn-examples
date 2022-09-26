/**
 * A small example with OpenMP. Should be generalizable to other platforms, but
 * assuming the platform we are executing on is OpenMP for simplicity's sake.
 */

#include <iostream>
#include <numeric>
#include <unordered_map>
#include <vector>

#include <dnnl.h>
#include <oneapi/dnnl/dnnl.hpp>

#include "../_utils/utils.hpp"

#define LOG(msgs...) dnnutils::stdout_log(msgs)
#define ERRLOG(msgs...) dnnutils::stderr_log(msgs)



int main(int argc, char **argv)
{
    if (DNNL_CPU_RUNTIME != DNNL_RUNTIME_OMP) {
        ERRLOG("Expected DNNL_CPU_RUNTIME to be OpenMP");
        return 1;
    }
    dnnl::engine::kind engine_kind = dnnl::engine::kind::cpu;

    LOG("Loading dataset from /mnt/_data");

    dnnutils::mnist mnist("/mnt/_data");

    LOG("training length: ", mnist.train_length());
    LOG("t10k length: ", mnist.t10k_length());

    // Load 256 images
    auto N = 256;
    auto H = mnist.H();
    auto W = mnist.W();
    auto C = mnist.C();
    auto O = 10;

    LOG("Start oneDNN code");
    using tag = dnnl::memory::format_tag;
    using dt = dnnl::memory::data_type;

    dnnl::memory::dims net_src_dims = {N, C, H, W};
    dnnl::memory::dims fc_w_dims = {O, C, H, W};
    dnnl::memory::dims fc_b_dims = {O};
    dnnl::memory::dims net_dst_dims = {N, O};
    LOG("net_src_dims: ", net_src_dims, " ({N, C, H, W})");
    LOG("net_dst_dims: ", net_dst_dims, " ({N, O})");
    LOG("dnnutils::product(net_src_dims): ", dnnutils::product(net_src_dims));
    LOG("dnnutils::product(net_dst_dims): ", dnnutils::product(net_dst_dims));
    LOG("Allocating input and output buffers");

    std::vector<float> net_src(dnnutils::product(net_src_dims));
    std::vector<float> net_dst(dnnutils::product(net_dst_dims));

    LOG("Allocating 1-hot label buffer");
    std::vector<float> net_labels(N * O);

    LOG("Allocating buffers for weights and biases");
    std::vector<float> fc_w(dnnutils::product(fc_w_dims));
    std::vector<float> fc_b(dnnutils::product(fc_b_dims));

    LOG("Loading batch...");
    mnist.train_copydata(0, N, net_src.data());
    LOG("Loading labels...");
    mnist.train_copylabels_1hot(0, N, net_labels.data());

    LOG("Initializing inner_product weights");
    dnnutils::he_initialize(fc_w);
    dnnutils::he_initialize(fc_b);

    LOG("Creating engine and the engine stream");
    dnnl::engine eng(engine_kind, 0);
    dnnl::stream engine_stream(eng);

    LOG("Creating memory descriptors");
    auto net_src_md = dnnl::memory::desc(net_src_dims, dt::f32, tag::nchw);
    auto fc_w_md = dnnl::memory::desc(fc_w_dims, dt::f32, tag::any); // <-- using tag any to allow oneDNN optimizer to choose the best layout
    auto fc_b_md = dnnl::memory::desc(fc_b_dims, dt::f32, tag::a);
    auto net_dst_md = dnnl::memory::desc(net_dst_dims, dt::f32, tag::nc);

    LOG("Creating the memory (except for the weights...)");
    auto net_src_mem = dnnl::memory(net_src_md, eng);
    auto fc_b_mem = dnnl::memory(fc_b_md, eng);
    auto net_dst_mem = dnnl::memory(net_dst_md, eng);

    LOG("Creating user weights memory");
    auto user_fc_w_mem = dnnl::memory({fc_w_dims, dt::f32, tag::oihw}, eng);

    LOG("Writing data to the DNNL memory");
    dnnutils::cpu_to_dnnl_memory(net_src.data(), net_src_mem);
    dnnutils::cpu_to_dnnl_memory(fc_w.data(), user_fc_w_mem);
    dnnutils::cpu_to_dnnl_memory(fc_b.data(), fc_b_mem);

    LOG("Creating Inner Product (FullyConnected) operation descriptor");
    auto fc_d = dnnl::inner_product_forward::desc(
        dnnl::prop_kind::forward_training,
        net_src_md, fc_w_md, fc_b_md, net_dst_md
    );

    LOG("Setting up ReLU activation post op");
    const float scale = 1.0f;
    const float alpha = 0.0f;
    const float beta = 0.0f;
    dnnl::post_ops fc_ops;
    fc_ops.append_eltwise(scale, dnnl::algorithm::eltwise_relu, alpha, beta);
    dnnl::primitive_attr fc_attr;
    fc_attr.set_post_ops(fc_ops);

    LOG("Creating FC primitive descriptor");
    auto fc_pd = dnnl::inner_product_forward::primitive_desc(fc_d, fc_attr, eng);

    LOG("Setup FC internal memory (initially assume it is the same format as the user memory)");
    auto fc_w_mem = user_fc_w_mem;
    if (fc_pd.weights_desc() != user_fc_w_mem.get_desc()) {
        LOG("  Layout mismatch! Setting up dedicated internal memory and reordering");
        fc_w_mem = dnnl::memory(fc_pd.weights_desc(), eng);
        dnnl::reorder(user_fc_w_mem, fc_w_mem).execute(engine_stream, user_fc_w_mem, fc_w_mem);
    } else {
        LOG("  They had the same layout!");
    }

    LOG("Creating the FC primitive (might take a while...)");
    auto fc_prim = dnnl::inner_product_forward(fc_pd);

    LOG("Setting up the arguments to the FC primitive");
    std::unordered_map<int, dnnl::memory> fc_args;
    fc_args.insert({DNNL_ARG_SRC, net_src_mem});
    fc_args.insert({DNNL_ARG_WEIGHTS, fc_w_mem});
    fc_args.insert({DNNL_ARG_BIAS, fc_b_mem});
    fc_args.insert({DNNL_ARG_DST, net_dst_mem});

    LOG("Setting up softmax");
    auto softmax_d = dnnl::softmax_forward::desc(
        dnnl::prop_kind::forward_training, net_dst_md, 1 // I assume axis=1 is correct?
    );
    auto softmax_pd = dnnl::softmax_forward::primitive_desc(softmax_d, eng);
    auto softmax_prim = dnnl::softmax_forward(softmax_pd);

    LOG("Setting up Softmax arguments (in-place operation: net_dst_mem -> net_dst_mem)");
    std::unordered_map<int, dnnl::memory> softmax_args;
    softmax_args.insert({DNNL_ARG_SRC, net_dst_mem});
    softmax_args.insert({DNNL_ARG_DST, net_dst_mem});

    LOG("Executing the primitives");
    fc_prim.execute(engine_stream, fc_args);
    softmax_prim.execute(engine_stream, softmax_args);

    LOG("Waiting for the stream to finalize");
    engine_stream.wait();

    LOG("Read from the destination memory");
    dnnutils::cpu_from_dnnl_memory(net_dst.data(), net_dst_mem);


    LOG("------------ FORWARD PASS DONE! ------------");


    ERRLOG("Computing Cross Entropy Loss..."); // for practice, compute the cross-entropy locally.
    // This could be done by using binary operations:
    //  o = p - y
    //  o2 = o * o
    //  L = sum(o2) / len(o2)
    std::vector<float> loss_buf(net_labels.size());
    for (std::size_t i = 0; i < net_labels.size(); ++i) {
        float diff = net_dst[i] - net_labels[i];
        loss_buf[i] = diff*diff / ((float) net_labels.size());
    }

    LOG("------------ LOSS COMPUTATION DONE! ------------");

    LOG("Starting backwards pass");
    std::vector<dnnl::primitive> net_bwd;
    std::vector<std::unordered_map<int, dnnl::memory>> net_bwd_args;

    LOG("Setting up diff buffers");
    std::vector<float> fc_w_bwd(dnnutils::product(fc_w_dims));
    std::vector<float> fc_b_bwd(dnnutils::product(fc_b_dims));

    LOG("Setting upp diff memory");
    auto net_dst_bwd_mem = dnnl::memory(net_dst_md, eng);
    dnnutils::cpu_to_dnnl_memory(loss_buf.data(), net_dst_bwd_mem);

    LOG("Setting up backward softmax");
    auto softmax_bwd_d = dnnl::softmax_backward::desc(
        net_dst_md, net_dst_md, 1
    );
    auto softmax_bwd_pd = dnnl::softmax_backward::primitive_desc(
        softmax_bwd_d, eng, softmax_pd // pass along prim_desc to the forward object
    );

    LOG("Setting up softmax backward primitive and arguments");
    net_bwd.push_back(dnnl::softmax_backward(softmax_bwd_pd));
    net_bwd_args.push_back({
        {DNNL_ARG_DST, net_dst_mem},
        {DNNL_ARG_DIFF_DST, net_dst_bwd_mem},
        {DNNL_ARG_DIFF_SRC, net_dst_bwd_mem}
    });
    // NOTE: Here is where I found information about which arguments to use
    // https://github.com/oneapi-src/oneDNN/blob/master/src/cpu/ref_softmax.cpp#L338

    LOG("Setting up backward desc and primitive_desc inner product on the weights");
    auto fc_bwd_d = dnnl::inner_product_backward_weights::desc(
        net_src_md, // src_desc
        fc_w_md,    // diff_weights_desc
        fc_b_md,    // diff_bias_desc
        net_dst_md  // diff_dst_desc
    );
    auto fc_bwd_pd = dnnl::inner_product_backward_weights::primitive_desc(
        fc_bwd_d, eng, fc_pd // <-- primitive_desc from forward pass
    );

    LOG("Setting up backward memory for inner_product");
    auto fc_b_bwd_mem = dnnl::memory(fc_b_md, eng);

    LOG("Creating user backward weights memory");
    auto user_fc_w_bwd_mem = dnnl::memory({fc_w_dims, dt::f32, tag::oihw}, eng);

    LOG("Writing local buffer data to dnnl memory, for some reason...");
    dnnutils::cpu_to_dnnl_memory(fc_w_bwd.data(), user_fc_w_bwd_mem);
    dnnutils::cpu_to_dnnl_memory(fc_b_bwd.data(), fc_b_bwd_mem);

    LOG("Setting up src memory for inner_product");
    auto net_src_bwd_mem = net_src_mem;
    if (fc_bwd_pd.src_desc() != net_src_mem.get_desc()) {
        LOG("  reordering net_src_bwd_mem");
        net_src_bwd_mem = dnnl::memory(fc_bwd_pd.src_desc(), eng);
        net_bwd.push_back(dnnl::reorder(net_src_mem, net_src_bwd_mem));
        net_bwd_args.push_back({
            {DNNL_ARG_FROM, net_src_mem},
            {DNNL_ARG_TO, net_src_bwd_mem}
        });
    }

    LOG("Setting up backward inner product primitive (delaying weights mem argument a little though...)");
    net_bwd.push_back(dnnl::inner_product_backward_weights(fc_bwd_pd));
    net_bwd_args.push_back({
        {DNNL_ARG_SRC, net_src_bwd_mem},
        //{DNNL_ARG_DIFF_WEIGHTS, ???}, (delaying this until the next if-stmt)
        {DNNL_ARG_DIFF_BIAS, fc_b_bwd_mem},
        {DNNL_ARG_DIFF_DST, net_dst_bwd_mem}
    });

    LOG("Setting up the backward weights memory (checking if we need to reorder the memory...)");
    auto fc_w_bwd_mem = user_fc_w_bwd_mem;
    if (fc_bwd_pd.diff_weights_desc() != user_fc_w_bwd_mem.get_desc()) {
        LOG("  Layout mismatch! Setting up dedicated internal memory and reordering after execution");
        fc_w_bwd_mem = dnnl::memory(fc_bwd_pd.diff_weights_desc(), eng);

        net_bwd_args.back().insert({DNNL_ARG_DIFF_WEIGHTS, fc_w_bwd_mem});

        LOG("  Setting up weights matrix reordering after backward");
        net_bwd.push_back(dnnl::reorder(fc_w_bwd_mem, user_fc_w_bwd_mem));
        net_bwd_args.push_back({
            {DNNL_ARG_FROM, fc_w_bwd_mem},
            {DNNL_ARG_TO, user_fc_w_bwd_mem}
        });
    } else {
        LOG("  They had the same layout!");
        net_bwd_args.back().insert({DNNL_ARG_DIFF_WEIGHTS, fc_w_bwd_mem});
    }

    LOG("Executing backward pass");
    for (std::size_t i = 0; i < net_bwd.size(); ++i) {
        ERRLOG("Executing layer ", i, "...");
        net_bwd.at(i).execute(engine_stream, net_bwd_args.at(i));
    }

    engine_stream.wait();

    LOG("------------ BACKWARD PASS DONE! ------------");

    return 0;
}
