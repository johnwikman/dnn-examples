/**
 * model.cpp
 * 
 * Implements the MNIST model instantiation, training, and inferrence.
 */

#include <vector>
#include <unordered_map>

#include <dnnl.h>
#include <oneapi/dnnl/dnnl.hpp>

#include "model.hpp"
#include "../_utils/utils.hpp"

#define LOG(msgs...) dnnutils::stdout_log(msgs)
#define DBGLOG(msgs...) dnnutils::stdout_log(__FILE__, ":", __LINE__, ": ", msgs)
#define ERRLOG(msgs...) dnnutils::stderr_log(msgs)

///* Comment out this line to disable debug prints
#undef DBGLOG
#define DBGLOG(...)
// */

#define SIZE_SANITY_CHECK(name, desc_size, bufsize)     \
    do {                                                \
        if ((desc_size) != ((bufsize) * sizeof(float))) { \
            ERRLOG("SIZE MISMATCH ", name, ". Got ", desc_size, ", expected ", (bufsize) * sizeof(float), "."); \
        }                                               \
    } while (0)


mnist_model::mnist_model(dnnl::engine &engine):
    m_engine(engine),
    m_stream(engine),
    // Setup default learning rate
    m_sgd_learning_rate(0.9),
    // Set batch sizes to 0 initially, to represent that no network has yet been initialized
    m_train_batch_size(0),
    m_infer_batch_size(0)
{
    const int flat_idim = idim_H * idim_W * idim_C;

    // Initialize the weights using he-initialization
    m_param_fc1_w = std::vector<float>(flat_idim * ldim_fc1);
    m_param_fc1_b = std::vector<float>(ldim_fc1);
    m_param_fc2_w = std::vector<float>(ldim_fc1 * ldim_fc2);
    m_param_fc2_b = std::vector<float>(ldim_fc2);
    m_param_fc3_w = std::vector<float>(ldim_fc2 * ldim_fc3);
    m_param_fc3_b = std::vector<float>(ldim_fc3);
    dnnutils::he_initialize(m_param_fc1_w);
    dnnutils::he_initialize(m_param_fc1_b);
    dnnutils::he_initialize(m_param_fc2_w);
    dnnutils::he_initialize(m_param_fc2_b);
    dnnutils::he_initialize(m_param_fc3_w);
    dnnutils::he_initialize(m_param_fc3_b);
}


mnist_model::~mnist_model()
{
    // Do nothing as of yet
}


/**
 * Rebuilds the training network to fit the specified batch size.
 * 
 * NOTE: If generating from DSL, might be some performance to gain by
 * statically transforming the data?
 */
void mnist_model::rebuild_train_network(size_t batch_size)
{
    using tag = dnnl::memory::format_tag;
    using dt = dnnl::memory::data_type;

    const int N = batch_size;
    const int H = idim_H;
    const int W = idim_W;
    const int C = idim_C;
    const int fc1_O = ldim_fc1;
    const int fc2_O = ldim_fc2;
    const int fc3_O = ldim_fc3;
    const int O = odim_O;

    DBGLOG("Clear up everything in the training net");
    m_net_fwd.clear();
    m_net_fwd_args.clear();
    m_net_bwd.clear();
    m_net_bwd_args.clear();
    m_net_mem.clear();

    DBGLOG("Setup all memory descriptors");
    auto md_input = dnnl::memory::desc({N, C, H, W}, dt::f32, tag::nhwc);
    auto md_fc1_w_user = dnnl::memory::desc({fc1_O, C, H, W}, dt::f32, tag::oihw);
    auto md_fc1_w = dnnl::memory::desc({fc1_O, C, H, W}, dt::f32, tag::any);
    auto md_fc1_b = dnnl::memory::desc({fc1_O}, dt::f32, tag::a);
    auto md_fc1_out = dnnl::memory::desc({N, fc1_O}, dt::f32, tag::nc);
    auto md_fc2_w_user = dnnl::memory::desc({fc2_O, fc1_O}, dt::f32, tag::ab);
    auto md_fc2_w = dnnl::memory::desc({fc2_O, fc1_O}, dt::f32, tag::any);
    auto md_fc2_b = dnnl::memory::desc({fc2_O}, dt::f32, tag::a);
    auto md_fc2_out = dnnl::memory::desc({N, fc2_O}, dt::f32, tag::nc);
    auto md_fc3_w_user = dnnl::memory::desc({fc3_O, fc2_O}, dt::f32, tag::ab);
    auto md_fc3_w = dnnl::memory::desc({fc3_O, fc2_O}, dt::f32, tag::any);
    auto md_fc3_b = dnnl::memory::desc({fc3_O}, dt::f32, tag::a);
    auto md_fc3_out = dnnl::memory::desc({N, fc3_O}, dt::f32, tag::nc);

    DBGLOG("Input mem");
    int memidx_input = m_net_mem.size();
    m_net_mem.push_back(dnnl::memory(md_input, m_engine));

    DBGLOG("Setup FC1 + relu");
    int memidx_param_fc1_w_user = m_net_mem.size();
    m_net_mem.push_back(dnnl::memory(md_fc1_w_user, m_engine));
    int memidx_param_fc1_b = m_net_mem.size();
    m_net_mem.push_back(dnnl::memory(md_fc1_b, m_engine));
    int memidx_param_fc1_out = m_net_mem.size();
    m_net_mem.push_back(dnnl::memory(md_fc1_out, m_engine));

    auto fc1_d = dnnl::inner_product_forward::desc(
        dnnl::prop_kind::forward_training,
        md_input, md_fc1_w, md_fc1_b, md_fc1_out
    );
    dnnl::post_ops fc1_post_ops;
    fc1_post_ops.append_eltwise(1.0f, dnnl::algorithm::eltwise_relu, 0.0f, 0.0f);
    dnnl::primitive_attr fc1_attr;
    fc1_attr.set_post_ops(fc1_post_ops);
    auto fc1_pd = dnnl::inner_product_forward::primitive_desc(
        fc1_d, fc1_attr, m_engine
    );

    int memidx_param_fc1_w = memidx_param_fc1_w_user;
    if (fc1_pd.weights_desc() != m_net_mem[memidx_param_fc1_w_user].get_desc()) {
        memidx_param_fc1_w = m_net_mem.size();
        m_net_mem.push_back(dnnl::memory(fc1_pd.weights_desc(), m_engine));
        m_net_fwd.push_back(dnnl::reorder(m_net_mem[memidx_param_fc1_w_user], m_net_mem[memidx_param_fc1_w]));
        m_net_fwd_args.push_back({
            {DNNL_ARG_FROM, m_net_mem[memidx_param_fc1_w_user]},
            {DNNL_ARG_TO,   m_net_mem[memidx_param_fc1_w]}
        });
    }

    m_net_fwd.push_back(dnnl::inner_product_forward(fc1_pd));
    m_net_fwd_args.push_back({
        {DNNL_ARG_SRC,     m_net_mem[memidx_input]},
        {DNNL_ARG_WEIGHTS, m_net_mem[memidx_param_fc1_w]},
        {DNNL_ARG_BIAS,    m_net_mem[memidx_param_fc1_b]},
        {DNNL_ARG_DST,     m_net_mem[memidx_param_fc1_out]}
    });

    DBGLOG("FC2 + relu");
    int memidx_param_fc2_w_user = m_net_mem.size();
    m_net_mem.push_back(dnnl::memory(md_fc2_w_user, m_engine));
    int memidx_param_fc2_b = m_net_mem.size();
    m_net_mem.push_back(dnnl::memory(md_fc2_b, m_engine));
    int memidx_param_fc2_out = m_net_mem.size();
    m_net_mem.push_back(dnnl::memory(md_fc2_out, m_engine));

    auto fc2_d = dnnl::inner_product_forward::desc(
        dnnl::prop_kind::forward_training,
        md_fc1_out, md_fc2_w, md_fc2_b, md_fc2_out
    );
    dnnl::post_ops fc2_post_ops;
    fc2_post_ops.append_eltwise(1.0f, dnnl::algorithm::eltwise_relu, 0.0f, 0.0f);
    dnnl::primitive_attr fc2_attr;
    fc2_attr.set_post_ops(fc2_post_ops);
    auto fc2_pd = dnnl::inner_product_forward::primitive_desc(
        fc2_d, fc2_attr, m_engine
    );

    int memidx_param_fc2_w = memidx_param_fc2_w_user;
    if (fc2_pd.weights_desc() != m_net_mem[memidx_param_fc2_w_user].get_desc()) {
        memidx_param_fc2_w = m_net_mem.size();
        m_net_mem.push_back(dnnl::memory(fc2_pd.weights_desc(), m_engine));
        m_net_fwd.push_back(dnnl::reorder(m_net_mem[memidx_param_fc2_w_user], m_net_mem[memidx_param_fc2_w]));
        m_net_fwd_args.push_back({
            {DNNL_ARG_FROM, m_net_mem[memidx_param_fc2_w_user]},
            {DNNL_ARG_TO,   m_net_mem[memidx_param_fc2_w]}
        });
    }

    m_net_fwd.push_back(dnnl::inner_product_forward(fc2_pd));
    m_net_fwd_args.push_back({
        {DNNL_ARG_SRC,     m_net_mem[memidx_param_fc1_out]},
        {DNNL_ARG_WEIGHTS, m_net_mem[memidx_param_fc2_w]},
        {DNNL_ARG_BIAS,    m_net_mem[memidx_param_fc2_b]},
        {DNNL_ARG_DST,     m_net_mem[memidx_param_fc2_out]}
    });

    DBGLOG("FC3 (with no ReLU, doing SoftMax after this)");
    int memidx_param_fc3_w_user = m_net_mem.size();
    m_net_mem.push_back(dnnl::memory(md_fc3_w_user, m_engine));
    int memidx_param_fc3_b = m_net_mem.size();
    m_net_mem.push_back(dnnl::memory(md_fc3_b, m_engine));
    int memidx_param_fc3_out = m_net_mem.size();
    m_net_mem.push_back(dnnl::memory(md_fc3_out, m_engine));

    auto fc3_d = dnnl::inner_product_forward::desc(
        dnnl::prop_kind::forward_training,
        md_fc2_out, md_fc3_w, md_fc3_b, md_fc3_out
    );
    dnnl::primitive_attr fc3_attr;
    auto fc3_pd = dnnl::inner_product_forward::primitive_desc(
        fc3_d, fc3_attr, m_engine
    );

    int memidx_param_fc3_w = memidx_param_fc3_w_user;
    if (fc3_pd.weights_desc() != m_net_mem[memidx_param_fc3_w_user].get_desc()) {
        memidx_param_fc3_w = m_net_mem.size();
        m_net_mem.push_back(dnnl::memory(fc3_pd.weights_desc(), m_engine));
        m_net_fwd.push_back(dnnl::reorder(m_net_mem[memidx_param_fc3_w_user], m_net_mem[memidx_param_fc3_w]));
        m_net_fwd_args.push_back({
            {DNNL_ARG_FROM, m_net_mem[memidx_param_fc3_w_user]},
            {DNNL_ARG_TO,   m_net_mem[memidx_param_fc3_w]}
        });
    }

    m_net_fwd.push_back(dnnl::inner_product_forward(fc3_pd));
    m_net_fwd_args.push_back({
        {DNNL_ARG_SRC,     m_net_mem[memidx_param_fc2_out]},
        {DNNL_ARG_WEIGHTS, m_net_mem[memidx_param_fc3_w]},
        {DNNL_ARG_BIAS,    m_net_mem[memidx_param_fc3_b]},
        {DNNL_ARG_DST,     m_net_mem[memidx_param_fc3_out]}
    });

    DBGLOG("SoftMax (inplace operation)");
    auto softmax_d = dnnl::softmax_forward::desc(
        dnnl::prop_kind::forward_training,
        md_fc3_out, 1 // axis=1
    );
    auto softmax_pd = dnnl::softmax_forward::primitive_desc(softmax_d, m_engine);

    m_net_fwd.push_back(dnnl::softmax_forward(softmax_pd));
    m_net_fwd_args.push_back({
        {DNNL_ARG_SRC, m_net_mem[memidx_param_fc3_out]},
        {DNNL_ARG_DST, m_net_mem[memidx_param_fc3_out]}
    });

    DBGLOG("Loss Memory");
    int memidx_bwd_diff_fc3_out = m_net_mem.size();
    m_net_mem.push_back(dnnl::memory(md_fc3_out, m_engine));

    DBGLOG("Backward SoftMax (inplace operation)");
    auto bwd_softmax_d = dnnl::softmax_backward::desc(
        md_fc3_out, md_fc3_out, 1
    );
    auto bwd_softmax_pd = dnnl::softmax_backward::primitive_desc(
        bwd_softmax_d, m_engine, softmax_pd
    );
    m_net_bwd.push_back(dnnl::softmax_backward(bwd_softmax_pd));
    m_net_bwd_args.push_back({
        {DNNL_ARG_DST,      m_net_mem[memidx_param_fc3_out]},
        {DNNL_ARG_DIFF_DST, m_net_mem[memidx_bwd_diff_fc3_out]},
        {DNNL_ARG_DIFF_SRC, m_net_mem[memidx_bwd_diff_fc3_out]}
    });

    DBGLOG("Backward FC3 (weights)");
    int memidx_bwd_diff_fc3_w_user = m_net_mem.size();
    m_net_mem.push_back(dnnl::memory(md_fc3_w_user, m_engine));
    int memidx_bwd_diff_fc3_b = m_net_mem.size();
    m_net_mem.push_back(dnnl::memory(md_fc3_b, m_engine));

    auto bwd_fc3_d = dnnl::inner_product_backward_weights::desc(
        md_fc2_out,
        md_fc3_w,
        md_fc3_b,
        md_fc3_out
    );
    auto bwd_fc3_pd = dnnl::inner_product_backward_weights::primitive_desc(
        bwd_fc3_d, m_engine, fc3_pd // <-- forward primitive_desc
    );

    int memidx_bwd_param_fc2_out = memidx_param_fc2_out;
    if (bwd_fc3_pd.src_desc() != m_net_mem[memidx_bwd_param_fc2_out].get_desc()) {
        memidx_bwd_param_fc2_out = m_net_mem.size();
        m_net_mem.push_back(dnnl::memory(bwd_fc3_pd.weights_desc(), m_engine));
        m_net_bwd.push_back(dnnl::reorder(m_net_mem[memidx_param_fc2_out], m_net_mem[memidx_bwd_param_fc2_out]));
        m_net_bwd_args.push_back({
            {DNNL_ARG_FROM, m_net_mem[memidx_param_fc2_out]},
            {DNNL_ARG_TO,   m_net_mem[memidx_bwd_param_fc2_out]}
        });
    }

    m_net_bwd.push_back(dnnl::inner_product_backward_weights(bwd_fc3_pd));
    m_net_bwd_args.push_back({
        {DNNL_ARG_SRC,            m_net_mem[memidx_bwd_param_fc2_out]},
        //{DNNL_ARG_DIFF_WEIGHTS, ???}, (delaying this until the next if-stmt)
        {DNNL_ARG_DIFF_BIAS,      m_net_mem[memidx_bwd_diff_fc3_b]},
        {DNNL_ARG_DIFF_DST,       m_net_mem[memidx_bwd_diff_fc3_out]}
    });

    int memidx_bwd_diff_fc3_w = memidx_bwd_diff_fc3_w_user;
    if (bwd_fc3_pd.diff_weights_desc() != m_net_mem[memidx_bwd_diff_fc3_w_user].get_desc()) {
        memidx_bwd_diff_fc3_w = m_net_mem.size();
        m_net_mem.push_back(dnnl::memory(bwd_fc3_pd.diff_weights_desc(), m_engine));

        m_net_bwd_args.back().insert({DNNL_ARG_DIFF_WEIGHTS, m_net_mem[memidx_bwd_diff_fc3_w]});

        m_net_bwd.push_back(dnnl::reorder(m_net_mem[memidx_bwd_diff_fc3_w], m_net_mem[memidx_bwd_diff_fc3_w_user]));
        m_net_bwd_args.push_back({
            {DNNL_ARG_FROM, m_net_mem[memidx_bwd_diff_fc3_w]},
            {DNNL_ARG_TO,   m_net_mem[memidx_bwd_diff_fc3_w_user]}
        });
    } else {
        m_net_bwd_args.back().insert({DNNL_ARG_DIFF_WEIGHTS, m_net_mem[memidx_bwd_diff_fc3_w]});
    }

    DBGLOG("Backward FC3 (data)");
    int memidx_bwd_diff_fc2_out = m_net_mem.size();
    m_net_mem.push_back(dnnl::memory(md_fc2_out, m_engine));

    auto bwd_fc3data_d = dnnl::inner_product_backward_data::desc(
        md_fc2_out,
        md_fc3_w,
        md_fc3_out
    );
    auto bwd_fc3data_pd = dnnl::inner_product_backward_data::primitive_desc(
        bwd_fc3data_d, m_engine, fc3_pd // <-- forward primitive_desc
    );

    m_net_bwd.push_back(dnnl::inner_product_backward_data(bwd_fc3data_pd));
    m_net_bwd_args.push_back({
        {DNNL_ARG_DIFF_SRC, m_net_mem[memidx_bwd_diff_fc2_out]},
        {DNNL_ARG_WEIGHTS,  m_net_mem[memidx_param_fc3_w]},
        {DNNL_ARG_DIFF_DST, m_net_mem[memidx_bwd_diff_fc3_out]}
    });

    DBGLOG("Backward FC2 + ReLU (weights)");
    int memidx_bwd_diff_fc2_w_user = m_net_mem.size();
    m_net_mem.push_back(dnnl::memory(md_fc2_w_user, m_engine));
    int memidx_bwd_diff_fc2_b = m_net_mem.size();
    m_net_mem.push_back(dnnl::memory(md_fc2_b, m_engine));

    auto bwd_fc2_d = dnnl::inner_product_backward_weights::desc(
        md_fc1_out,
        md_fc2_w,
        md_fc2_b,
        md_fc2_out
    );
    auto bwd_fc2_pd = dnnl::inner_product_backward_weights::primitive_desc(
        bwd_fc2_d, m_engine, fc2_pd // <-- forward primitive_desc
    );

    int memidx_bwd_param_fc1_out = memidx_param_fc1_out;
    if (bwd_fc2_pd.src_desc() != m_net_mem[memidx_bwd_param_fc1_out].get_desc()) {
        memidx_bwd_param_fc1_out = m_net_mem.size();
        m_net_mem.push_back(dnnl::memory(bwd_fc2_pd.weights_desc(), m_engine));
        m_net_bwd.push_back(dnnl::reorder(m_net_mem[memidx_param_fc1_out], m_net_mem[memidx_bwd_param_fc1_out]));
        m_net_bwd_args.push_back({
            {DNNL_ARG_FROM, m_net_mem[memidx_param_fc1_out]},
            {DNNL_ARG_TO,   m_net_mem[memidx_bwd_param_fc1_out]}
        });
    }

    m_net_bwd.push_back(dnnl::inner_product_backward_weights(bwd_fc2_pd));
    m_net_bwd_args.push_back({
        {DNNL_ARG_SRC,            m_net_mem[memidx_bwd_param_fc1_out]},
        //{DNNL_ARG_DIFF_WEIGHTS, ???}, (delaying this until the next if-stmt)
        {DNNL_ARG_DIFF_BIAS,      m_net_mem[memidx_bwd_diff_fc2_b]},
        {DNNL_ARG_DIFF_DST,       m_net_mem[memidx_bwd_diff_fc2_out]}
    });

    int memidx_bwd_diff_fc2_w = memidx_bwd_diff_fc2_w_user;
    if (bwd_fc2_pd.diff_weights_desc() != m_net_mem[memidx_bwd_diff_fc2_w_user].get_desc()) {
        memidx_bwd_diff_fc2_w = m_net_mem.size();
        m_net_mem.push_back(dnnl::memory(bwd_fc2_pd.diff_weights_desc(), m_engine));

        m_net_bwd_args.back().insert({DNNL_ARG_DIFF_WEIGHTS, m_net_mem[memidx_bwd_diff_fc2_w]});

        m_net_bwd.push_back(dnnl::reorder(m_net_mem[memidx_bwd_diff_fc2_w], m_net_mem[memidx_bwd_diff_fc2_w_user]));
        m_net_bwd_args.push_back({
            {DNNL_ARG_FROM, m_net_mem[memidx_bwd_diff_fc2_w]},
            {DNNL_ARG_TO,   m_net_mem[memidx_bwd_diff_fc2_w_user]}
        });
    } else {
        m_net_bwd_args.back().insert({DNNL_ARG_DIFF_WEIGHTS, m_net_mem[memidx_bwd_diff_fc2_w]});
    }

    DBGLOG("Backward FC2 + ReLU (data)");
    int memidx_bwd_diff_fc1_out = m_net_mem.size();
    m_net_mem.push_back(dnnl::memory(md_fc1_out, m_engine));

    auto bwd_fc2data_d = dnnl::inner_product_backward_data::desc(
        md_fc1_out,
        md_fc2_w,
        md_fc2_out
    );
    auto bwd_fc2data_pd = dnnl::inner_product_backward_data::primitive_desc(
        bwd_fc2data_d, m_engine, fc2_pd // <-- forward primitive_desc
    );

    m_net_bwd.push_back(dnnl::inner_product_backward_data(bwd_fc2data_pd));
    m_net_bwd_args.push_back({
        {DNNL_ARG_DIFF_SRC, m_net_mem[memidx_bwd_diff_fc1_out]},
        {DNNL_ARG_WEIGHTS,  m_net_mem[memidx_param_fc2_w]},
        {DNNL_ARG_DIFF_DST, m_net_mem[memidx_bwd_diff_fc2_out]}
    });

    DBGLOG("Backward FC1 + ReLU (weights)");
    int memidx_bwd_diff_fc1_w_user = m_net_mem.size();
    m_net_mem.push_back(dnnl::memory(md_fc1_w_user, m_engine));
    int memidx_bwd_diff_fc1_b = m_net_mem.size();
    m_net_mem.push_back(dnnl::memory(md_fc1_b, m_engine));

    auto bwd_fc1_d = dnnl::inner_product_backward_weights::desc(
        md_input,
        md_fc1_w,
        md_fc1_b,
        md_fc1_out
    );
    auto bwd_fc1_pd = dnnl::inner_product_backward_weights::primitive_desc(
        bwd_fc1_d, m_engine, fc1_pd // <-- forward primitive_desc
    );

    int memidx_bwd_input = memidx_input;
    if (bwd_fc1_pd.src_desc() != m_net_mem[memidx_bwd_input].get_desc()) {
        memidx_bwd_input = m_net_mem.size();
        m_net_mem.push_back(dnnl::memory(bwd_fc1_pd.weights_desc(), m_engine));
        m_net_bwd.push_back(dnnl::reorder(m_net_mem[memidx_input], m_net_mem[memidx_bwd_input]));
        m_net_bwd_args.push_back({
            {DNNL_ARG_FROM, m_net_mem[memidx_input]},
            {DNNL_ARG_TO,   m_net_mem[memidx_bwd_input]}
        });
    }

    m_net_bwd.push_back(dnnl::inner_product_backward_weights(bwd_fc1_pd));
    m_net_bwd_args.push_back({
        {DNNL_ARG_SRC,            m_net_mem[memidx_bwd_input]},
        //{DNNL_ARG_DIFF_WEIGHTS, ???}, (delaying this until the next if-stmt)
        {DNNL_ARG_DIFF_BIAS,      m_net_mem[memidx_bwd_diff_fc1_b]},
        {DNNL_ARG_DIFF_DST,       m_net_mem[memidx_bwd_diff_fc1_out]}
    });

    int memidx_bwd_diff_fc1_w = memidx_bwd_diff_fc1_w_user;
    if (bwd_fc1_pd.diff_weights_desc() != m_net_mem[memidx_bwd_diff_fc1_w_user].get_desc()) {
        memidx_bwd_diff_fc1_w = m_net_mem.size();
        m_net_mem.push_back(dnnl::memory(bwd_fc1_pd.diff_weights_desc(), m_engine));

        m_net_bwd_args.back().insert({DNNL_ARG_DIFF_WEIGHTS, m_net_mem[memidx_bwd_diff_fc1_w]});

        m_net_bwd.push_back(dnnl::reorder(m_net_mem[memidx_bwd_diff_fc1_w], m_net_mem[memidx_bwd_diff_fc1_w_user]));
        m_net_bwd_args.push_back({
            {DNNL_ARG_FROM, m_net_mem[memidx_bwd_diff_fc1_w]},
            {DNNL_ARG_TO,   m_net_mem[memidx_bwd_diff_fc1_w_user]}
        });
    } else {
        m_net_bwd_args.back().insert({DNNL_ARG_DIFF_WEIGHTS, m_net_mem[memidx_bwd_diff_fc1_w]});
    }

    DBGLOG("(No need to compute the derivative in terms of the input)");

    DBGLOG("Set up necessary memory indices in the member variables");
    m_train_memidx_input = memidx_input;
    m_train_memidx_output = memidx_param_fc3_out;
    m_train_memidx_loss = memidx_bwd_diff_fc3_out;
    m_train_memidx_param_fc1_w = memidx_param_fc1_w_user;
    m_train_memidx_param_fc1_b = memidx_param_fc1_b;
    m_train_memidx_diff_fc1_w = memidx_bwd_diff_fc1_w_user;
    m_train_memidx_diff_fc1_b = memidx_bwd_diff_fc1_b;
    m_train_memidx_param_fc2_w = memidx_param_fc2_w_user;
    m_train_memidx_param_fc2_b = memidx_param_fc2_b;
    m_train_memidx_diff_fc2_w = memidx_bwd_diff_fc2_w_user;
    m_train_memidx_diff_fc2_b = memidx_bwd_diff_fc2_b;
    m_train_memidx_param_fc3_w = memidx_param_fc3_w_user;
    m_train_memidx_param_fc3_b = memidx_param_fc3_b;
    m_train_memidx_diff_fc3_w = memidx_bwd_diff_fc3_w_user;
    m_train_memidx_diff_fc3_b = memidx_bwd_diff_fc3_b;

    DBGLOG("Setting up training buffers");
    m_trainbuf_output = std::vector<float>(dnnutils::product(md_fc3_out.dims()));
    m_trainbuf_loss = std::vector<float>(dnnutils::product(md_fc3_out.dims()));
    m_trainbuf_diff_fc1_w = std::vector<float>(dnnutils::product(md_fc1_w.dims()));
    m_trainbuf_diff_fc1_b = std::vector<float>(dnnutils::product(md_fc1_b.dims()));
    m_trainbuf_diff_fc2_w = std::vector<float>(dnnutils::product(md_fc2_w.dims()));
    m_trainbuf_diff_fc2_b = std::vector<float>(dnnutils::product(md_fc2_b.dims()));
    m_trainbuf_diff_fc3_w = std::vector<float>(dnnutils::product(md_fc3_w.dims()));
    m_trainbuf_diff_fc3_b = std::vector<float>(dnnutils::product(md_fc3_b.dims()));

    DBGLOG("Record the new batch size");
    m_train_batch_size = N;

    DBGLOG("Final sanity checks");
    SIZE_SANITY_CHECK("fc1_w", m_net_mem[m_train_memidx_param_fc1_w].get_desc().get_size(), m_param_fc1_w.size());
    SIZE_SANITY_CHECK("fc1_b", m_net_mem[m_train_memidx_param_fc1_b].get_desc().get_size(), m_param_fc1_b.size());
    SIZE_SANITY_CHECK("diff_fc1_w", m_net_mem[m_train_memidx_diff_fc1_w].get_desc().get_size(), m_trainbuf_diff_fc1_w.size());
    SIZE_SANITY_CHECK("diff_fc1_b", m_net_mem[m_train_memidx_diff_fc1_b].get_desc().get_size(), m_trainbuf_diff_fc1_b.size());
    SIZE_SANITY_CHECK("fc2_w", m_net_mem[m_train_memidx_param_fc2_w].get_desc().get_size(), m_param_fc2_w.size());
    SIZE_SANITY_CHECK("fc2_b", m_net_mem[m_train_memidx_param_fc2_b].get_desc().get_size(), m_param_fc2_b.size());
    SIZE_SANITY_CHECK("diff_fc2_w", m_net_mem[m_train_memidx_diff_fc2_w].get_desc().get_size(), m_trainbuf_diff_fc2_w.size());
    SIZE_SANITY_CHECK("diff_fc2_b", m_net_mem[m_train_memidx_diff_fc2_b].get_desc().get_size(), m_trainbuf_diff_fc2_b.size());
    SIZE_SANITY_CHECK("fc3_w", m_net_mem[m_train_memidx_param_fc3_w].get_desc().get_size(), m_param_fc3_w.size());
    SIZE_SANITY_CHECK("fc3_b", m_net_mem[m_train_memidx_param_fc3_b].get_desc().get_size(), m_param_fc3_b.size());
    SIZE_SANITY_CHECK("diff_fc3_w", m_net_mem[m_train_memidx_diff_fc3_w].get_desc().get_size(), m_trainbuf_diff_fc3_w.size());
    SIZE_SANITY_CHECK("diff_fc3_b", m_net_mem[m_train_memidx_diff_fc3_b].get_desc().get_size(), m_trainbuf_diff_fc3_b.size());
}

/**
 * Rebuilds the inference network to fit the specified batch size.
 */
void mnist_model::rebuild_infer_network(size_t batch_size)
{
    using tag = dnnl::memory::format_tag;
    using dt = dnnl::memory::data_type;

    const int N = batch_size;
    const int H = idim_H;
    const int W = idim_W;
    const int C = idim_C;
    const int fc1_O = ldim_fc1;
    const int fc2_O = ldim_fc2;
    const int fc3_O = ldim_fc3;
    const int O = odim_O;

    DBGLOG("Clear up everything in the training net");
    m_infer_net.clear();
    m_infer_net_args.clear();
    m_infer_mem.clear();

    DBGLOG("Setup all memory descriptors");
    auto md_input = dnnl::memory::desc({N, C, H, W}, dt::f32, tag::nhwc);
    auto md_fc1_w_user = dnnl::memory::desc({fc1_O, C, H, W}, dt::f32, tag::oihw);
    auto md_fc1_w = dnnl::memory::desc({fc1_O, C, H, W}, dt::f32, tag::any);
    auto md_fc1_b = dnnl::memory::desc({fc1_O}, dt::f32, tag::a);
    auto md_fc1_out = dnnl::memory::desc({N, fc1_O}, dt::f32, tag::nc);
    auto md_fc2_w_user = dnnl::memory::desc({fc2_O, fc1_O}, dt::f32, tag::ab);
    auto md_fc2_w = dnnl::memory::desc({fc2_O, fc1_O}, dt::f32, tag::any);
    auto md_fc2_b = dnnl::memory::desc({fc2_O}, dt::f32, tag::a);
    auto md_fc2_out = dnnl::memory::desc({N, fc2_O}, dt::f32, tag::nc);
    auto md_fc3_w_user = dnnl::memory::desc({fc3_O, fc2_O}, dt::f32, tag::ab);
    auto md_fc3_w = dnnl::memory::desc({fc3_O, fc2_O}, dt::f32, tag::any);
    auto md_fc3_b = dnnl::memory::desc({fc3_O}, dt::f32, tag::a);
    auto md_fc3_out = dnnl::memory::desc({N, fc3_O}, dt::f32, tag::nc);

    DBGLOG("Input mem");
    int memidx_input = m_infer_mem.size();
    m_infer_mem.push_back(dnnl::memory(md_input, m_engine));

    DBGLOG("Setup FC1 + relu");
    int memidx_param_fc1_w_user = m_infer_mem.size();
    m_infer_mem.push_back(dnnl::memory(md_fc1_w_user, m_engine));
    int memidx_param_fc1_b = m_infer_mem.size();
    m_infer_mem.push_back(dnnl::memory(md_fc1_b, m_engine));
    int memidx_param_fc1_out = m_infer_mem.size();
    m_infer_mem.push_back(dnnl::memory(md_fc1_out, m_engine));

    auto fc1_d = dnnl::inner_product_forward::desc(
        dnnl::prop_kind::forward_inference, // difference towards the training network!
        md_input, md_fc1_w, md_fc1_b, md_fc1_out
    );
    dnnl::post_ops fc1_post_ops;
    fc1_post_ops.append_eltwise(1.0f, dnnl::algorithm::eltwise_relu, 0.0f, 0.0f);
    dnnl::primitive_attr fc1_attr;
    fc1_attr.set_post_ops(fc1_post_ops);
    auto fc1_pd = dnnl::inner_product_forward::primitive_desc(
        fc1_d, fc1_attr, m_engine
    );

    int memidx_param_fc1_w = memidx_param_fc1_w_user;
    if (fc1_pd.weights_desc() != m_infer_mem[memidx_param_fc1_w_user].get_desc()) {
        memidx_param_fc1_w = m_infer_mem.size();
        m_infer_mem.push_back(dnnl::memory(fc1_pd.weights_desc(), m_engine));
        m_infer_net.push_back(dnnl::reorder(m_infer_mem[memidx_param_fc1_w_user], m_infer_mem[memidx_param_fc1_w]));
        m_infer_net_args.push_back({
            {DNNL_ARG_FROM, m_infer_mem[memidx_param_fc1_w_user]},
            {DNNL_ARG_TO,   m_infer_mem[memidx_param_fc1_w]}
        });
    }

    m_infer_net.push_back(dnnl::inner_product_forward(fc1_pd));
    m_infer_net_args.push_back({
        {DNNL_ARG_SRC,     m_infer_mem[memidx_input]},
        {DNNL_ARG_WEIGHTS, m_infer_mem[memidx_param_fc1_w]},
        {DNNL_ARG_BIAS,    m_infer_mem[memidx_param_fc1_b]},
        {DNNL_ARG_DST,     m_infer_mem[memidx_param_fc1_out]}
    });

    DBGLOG("Setup FC2 + relu");
    int memidx_param_fc2_w_user = m_infer_mem.size();
    m_infer_mem.push_back(dnnl::memory(md_fc2_w_user, m_engine));
    int memidx_param_fc2_b = m_infer_mem.size();
    m_infer_mem.push_back(dnnl::memory(md_fc2_b, m_engine));
    int memidx_param_fc2_out = m_infer_mem.size();
    m_infer_mem.push_back(dnnl::memory(md_fc2_out, m_engine));

    auto fc2_d = dnnl::inner_product_forward::desc(
        dnnl::prop_kind::forward_inference,
        md_fc1_out, md_fc2_w, md_fc2_b, md_fc2_out
    );
    dnnl::post_ops fc2_post_ops;
    fc2_post_ops.append_eltwise(1.0f, dnnl::algorithm::eltwise_relu, 0.0f, 0.0f);
    dnnl::primitive_attr fc2_attr;
    fc2_attr.set_post_ops(fc2_post_ops);
    auto fc2_pd = dnnl::inner_product_forward::primitive_desc(
        fc2_d, fc2_attr, m_engine
    );

    int memidx_param_fc2_w = memidx_param_fc2_w_user;
    if (fc2_pd.weights_desc() != m_infer_mem[memidx_param_fc2_w_user].get_desc()) {
        memidx_param_fc2_w = m_infer_mem.size();
        m_infer_mem.push_back(dnnl::memory(fc2_pd.weights_desc(), m_engine));
        m_infer_net.push_back(dnnl::reorder(m_infer_mem[memidx_param_fc2_w_user], m_infer_mem[memidx_param_fc2_w]));
        m_infer_net_args.push_back({
            {DNNL_ARG_FROM, m_infer_mem[memidx_param_fc2_w_user]},
            {DNNL_ARG_TO,   m_infer_mem[memidx_param_fc2_w]}
        });
    }

    m_infer_net.push_back(dnnl::inner_product_forward(fc2_pd));
    m_infer_net_args.push_back({
        {DNNL_ARG_SRC,     m_infer_mem[memidx_param_fc1_out]},
        {DNNL_ARG_WEIGHTS, m_infer_mem[memidx_param_fc2_w]},
        {DNNL_ARG_BIAS,    m_infer_mem[memidx_param_fc2_b]},
        {DNNL_ARG_DST,     m_infer_mem[memidx_param_fc2_out]}
    });

    DBGLOG("Setup FC3");
    int memidx_param_fc3_w_user = m_infer_mem.size();
    m_infer_mem.push_back(dnnl::memory(md_fc3_w_user, m_engine));
    int memidx_param_fc3_b = m_infer_mem.size();
    m_infer_mem.push_back(dnnl::memory(md_fc3_b, m_engine));
    int memidx_param_fc3_out = m_infer_mem.size();
    m_infer_mem.push_back(dnnl::memory(md_fc3_out, m_engine));

    auto fc3_d = dnnl::inner_product_forward::desc(
        dnnl::prop_kind::forward_inference,
        md_fc2_out, md_fc3_w, md_fc3_b, md_fc3_out
    );
    dnnl::primitive_attr fc3_attr;
    auto fc3_pd = dnnl::inner_product_forward::primitive_desc(
        fc3_d, fc3_attr, m_engine
    );

    int memidx_param_fc3_w = memidx_param_fc3_w_user;
    if (fc3_pd.weights_desc() != m_infer_mem[memidx_param_fc3_w_user].get_desc()) {
        memidx_param_fc3_w = m_infer_mem.size();
        m_infer_mem.push_back(dnnl::memory(fc3_pd.weights_desc(), m_engine));
        m_infer_net.push_back(dnnl::reorder(m_infer_mem[memidx_param_fc3_w_user], m_infer_mem[memidx_param_fc3_w]));
        m_infer_net_args.push_back({
            {DNNL_ARG_FROM, m_infer_mem[memidx_param_fc3_w_user]},
            {DNNL_ARG_TO,   m_infer_mem[memidx_param_fc3_w]}
        });
    }

    m_infer_net.push_back(dnnl::inner_product_forward(fc3_pd));
    m_infer_net_args.push_back({
        {DNNL_ARG_SRC,     m_infer_mem[memidx_param_fc2_out]},
        {DNNL_ARG_WEIGHTS, m_infer_mem[memidx_param_fc3_w]},
        {DNNL_ARG_BIAS,    m_infer_mem[memidx_param_fc3_b]},
        {DNNL_ARG_DST,     m_infer_mem[memidx_param_fc3_out]}
    });

    DBGLOG("Setup SoftMax (inplace operation)");
    auto softmax_d = dnnl::softmax_forward::desc(
        dnnl::prop_kind::forward_inference,
        md_fc3_out, 1 // axis=1
    );
    auto softmax_pd = dnnl::softmax_forward::primitive_desc(softmax_d, m_engine);

    m_infer_net.push_back(dnnl::softmax_forward(softmax_pd));
    m_infer_net_args.push_back({
        {DNNL_ARG_SRC, m_infer_mem[memidx_param_fc3_out]},
        {DNNL_ARG_DST, m_infer_mem[memidx_param_fc3_out]}
    });

    DBGLOG("Set up necessary memory indices in the member variables");
    m_infer_memidx_input = memidx_input;
    m_infer_memidx_param_fc1_w = memidx_param_fc1_w_user;
    m_infer_memidx_param_fc1_b = memidx_param_fc1_b;
    m_infer_memidx_param_fc2_w = memidx_param_fc2_w_user;
    m_infer_memidx_param_fc2_b = memidx_param_fc2_b;
    m_infer_memidx_param_fc3_w = memidx_param_fc3_w_user;
    m_infer_memidx_param_fc3_b = memidx_param_fc3_b;
    m_infer_memidx_output = memidx_param_fc3_out;

    DBGLOG("Setting up inference buffers");
    m_inferbuf_output = std::vector<float>(dnnutils::product(md_fc3_out.dims()));

    DBGLOG("Record the new batch size");
    m_infer_batch_size = batch_size;

    DBGLOG("Performing final sanity checks");
    SIZE_SANITY_CHECK("fc1_w", m_infer_mem[m_infer_memidx_param_fc1_w].get_desc().get_size(), m_param_fc1_w.size());
    SIZE_SANITY_CHECK("fc1_b", m_infer_mem[m_infer_memidx_param_fc1_b].get_desc().get_size(), m_param_fc1_b.size());
    SIZE_SANITY_CHECK("fc2_w", m_infer_mem[m_infer_memidx_param_fc2_w].get_desc().get_size(), m_param_fc2_w.size());
    SIZE_SANITY_CHECK("fc2_b", m_infer_mem[m_infer_memidx_param_fc2_b].get_desc().get_size(), m_param_fc2_b.size());
    SIZE_SANITY_CHECK("fc3_w", m_infer_mem[m_infer_memidx_param_fc3_w].get_desc().get_size(), m_param_fc3_w.size());
    SIZE_SANITY_CHECK("fc3_b", m_infer_mem[m_infer_memidx_param_fc3_b].get_desc().get_size(), m_param_fc3_b.size());
}

/**
 * Trains the network on the provided data, doing one gradient descent update step.
 */
void mnist_model::train(const float *data, const float *labels, size_t batch_size)
{
    DBGLOG("Check if we need to scale up the network");
    if (batch_size > m_train_batch_size)
        rebuild_train_network(batch_size);

    DBGLOG("Load FC weights");
    dnnutils::cpu_to_dnnl_memory(m_param_fc1_w.data(), m_net_mem[m_train_memidx_param_fc1_w]);
    dnnutils::cpu_to_dnnl_memory(m_param_fc1_b.data(), m_net_mem[m_train_memidx_param_fc1_b]);
    dnnutils::cpu_to_dnnl_memory(m_param_fc2_w.data(), m_net_mem[m_train_memidx_param_fc2_w]);
    dnnutils::cpu_to_dnnl_memory(m_param_fc2_b.data(), m_net_mem[m_train_memidx_param_fc2_b]);
    dnnutils::cpu_to_dnnl_memory(m_param_fc3_w.data(), m_net_mem[m_train_memidx_param_fc3_w]);
    dnnutils::cpu_to_dnnl_memory(m_param_fc3_b.data(), m_net_mem[m_train_memidx_param_fc3_b]);

    DBGLOG("Load input data");
    dnnutils::cpu_to_dnnl_memory(data, m_net_mem[m_train_memidx_input]);

    DBGLOG("Running forward pass");
    for (std::size_t i = 0; i < m_net_fwd.size(); ++i)
        m_net_fwd.at(i).execute(m_stream, m_net_fwd_args.at(i));

    m_stream.wait();

    DBGLOG("Computing loss");
    dnnutils::cpu_from_dnnl_memory(m_trainbuf_output.data(), m_net_mem[m_train_memidx_output]);
    for (std::size_t i = 0; i < (batch_size * odim_O); ++i) {
        float diff = m_trainbuf_output[i] - labels[i];
        m_trainbuf_loss[i] = diff * diff / ((float) batch_size);
    }
    DBGLOG("(Set everything above the batch_size to have loss 0, i.e. ignore these computations)");
    for (std::size_t i = (batch_size * odim_O); i < m_trainbuf_loss.size(); ++i)
        m_trainbuf_loss[i] = 0.0;

    dnnutils::cpu_to_dnnl_memory(m_trainbuf_loss.data(), m_net_mem[m_train_memidx_loss]);

    DBGLOG("Running backward pass");
    for (std::size_t i = 0; i < m_net_bwd.size(); ++i)
        m_net_bwd.at(i).execute(m_stream, m_net_bwd_args.at(i));

    m_stream.wait();

    DBGLOG("Retrieving backward data");
    dnnutils::cpu_from_dnnl_memory(m_trainbuf_diff_fc1_w.data(), m_net_mem[m_train_memidx_diff_fc1_w]);
    dnnutils::cpu_from_dnnl_memory(m_trainbuf_diff_fc1_b.data(), m_net_mem[m_train_memidx_diff_fc1_b]);
    dnnutils::cpu_from_dnnl_memory(m_trainbuf_diff_fc2_w.data(), m_net_mem[m_train_memidx_diff_fc2_w]);
    dnnutils::cpu_from_dnnl_memory(m_trainbuf_diff_fc2_b.data(), m_net_mem[m_train_memidx_diff_fc2_b]);
    dnnutils::cpu_from_dnnl_memory(m_trainbuf_diff_fc3_w.data(), m_net_mem[m_train_memidx_diff_fc3_w]);
    dnnutils::cpu_from_dnnl_memory(m_trainbuf_diff_fc3_b.data(), m_net_mem[m_train_memidx_diff_fc3_b]);

    DBGLOG("Performing gradient step");
    apply_gradient(m_param_fc1_w, m_trainbuf_diff_fc1_w);
    apply_gradient(m_param_fc1_b, m_trainbuf_diff_fc1_b);
    apply_gradient(m_param_fc2_w, m_trainbuf_diff_fc2_w);
    apply_gradient(m_param_fc2_b, m_trainbuf_diff_fc2_b);
    apply_gradient(m_param_fc3_w, m_trainbuf_diff_fc3_w);
    apply_gradient(m_param_fc3_b, m_trainbuf_diff_fc3_b);
}

/**
 * Infers categories from the provided data.
 */
void mnist_model::infer(const float *data, uint8_t *category, size_t batch_size)
{
    DBGLOG("Check if we need to scale up the network");
    if (batch_size > m_infer_batch_size)
        rebuild_infer_network(batch_size);

    // Note, might be good with a flag here to check whether we need to reload the weights or not.
    DBGLOG("Load FC weights");
    dnnutils::cpu_to_dnnl_memory(m_param_fc1_w.data(), m_infer_mem[m_infer_memidx_param_fc1_w]);
    dnnutils::cpu_to_dnnl_memory(m_param_fc1_b.data(), m_infer_mem[m_infer_memidx_param_fc1_b]);
    dnnutils::cpu_to_dnnl_memory(m_param_fc2_w.data(), m_infer_mem[m_infer_memidx_param_fc2_w]);
    dnnutils::cpu_to_dnnl_memory(m_param_fc2_b.data(), m_infer_mem[m_infer_memidx_param_fc2_b]);
    dnnutils::cpu_to_dnnl_memory(m_param_fc3_w.data(), m_infer_mem[m_infer_memidx_param_fc3_w]);
    dnnutils::cpu_to_dnnl_memory(m_param_fc3_b.data(), m_infer_mem[m_infer_memidx_param_fc3_b]);

    DBGLOG("Load input data");
    dnnutils::cpu_to_dnnl_memory(data, m_infer_mem[m_infer_memidx_input]);

    DBGLOG("Running forward pass");
    for (std::size_t i = 0; i < m_infer_net.size(); ++i) {
        m_infer_net.at(i).execute(m_stream, m_infer_net_args.at(i));
    }

    m_stream.wait();

    DBGLOG("Retrieving output data");
    dnnutils::cpu_from_dnnl_memory(m_inferbuf_output.data(), m_infer_mem[m_infer_memidx_output]);

    DBGLOG("Computing argmax");
    for (std::size_t i = 0; i < batch_size; ++i) {
        uint8_t amax = 0;
        float val = m_inferbuf_output[i*10 + 0];
        for (uint8_t j = 1; j < 10; ++j) {
            float cand = m_inferbuf_output[i*10 + j];
            if (cand > val) {
                amax = j;
                val = cand;
            }
        }
        category[i] = amax;
    }
}

/**
 * Decays the learning rate, scaling it by the provided scalar.
 */
void mnist_model::decay_learning_rate(float scale)
{
    m_sgd_learning_rate *= scale;
}
