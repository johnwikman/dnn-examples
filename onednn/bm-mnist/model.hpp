/**
 * model.hpp
 * 
 * Model class for the MNIST network
 */

#ifndef BM_MNIST_MODEL_HPP
#define BM_MNIST_MODEL_HPP

#include <vector>
#include <unordered_map>

#include <dnnl.h>
#include <oneapi/dnnl/dnnl.hpp>

/**
 * A 2-layer MNIST network, equivalent to that found in miking-ml.
 * 
 * Input -> FC(in, 784) -> ReLU -> FC(784, 128) -> ReLU -> FC(128, 10) -> SoftMax
 * 
 * Trained with cross-entropy loss
 */
class mnist_model {
private:
    // Input dimensions (for a single batch item)
    static const int idim_H = 28;
    static const int idim_W = 28;
    static const int idim_C = 1;

    // FC Layer dimensions
    static const int ldim_fc1 = 784;
    static const int ldim_fc2 = 128;
    static const int ldim_fc3 = 10;

    // Output dimension (for a single batch item)
    static const int odim_O = 10;

    // Layer weights
    std::vector<float> m_param_fc1_w;
    std::vector<float> m_param_fc1_b;
    std::vector<float> m_param_fc2_w;
    std::vector<float> m_param_fc2_b;
    std::vector<float> m_param_fc3_w;
    std::vector<float> m_param_fc3_b;

    // oneDNN Engine
    dnnl::engine &m_engine;
    dnnl::stream m_stream;

    // Network for training
    std::vector<dnnl::primitive> m_net_fwd;
    std::vector<std::unordered_map<int, dnnl::memory>> m_net_fwd_args;
    std::vector<dnnl::primitive> m_net_bwd;
    std::vector<std::unordered_map<int, dnnl::memory>> m_net_bwd_args;

    // Training parameters
    float m_sgd_learning_rate;

    // Training memory handles
    std::vector<dnnl::memory> m_net_mem;
    int m_train_memidx_input;
    int m_train_memidx_output;
    int m_train_memidx_loss;
    int m_train_memidx_param_fc1_w;
    int m_train_memidx_param_fc1_b;
    int m_train_memidx_diff_fc1_w;
    int m_train_memidx_diff_fc1_b;
    int m_train_memidx_param_fc2_w;
    int m_train_memidx_param_fc2_b;
    int m_train_memidx_diff_fc2_w;
    int m_train_memidx_diff_fc2_b;
    int m_train_memidx_param_fc3_w;
    int m_train_memidx_param_fc3_b;
    int m_train_memidx_diff_fc3_w;
    int m_train_memidx_diff_fc3_b;

    // Training buffers
    std::vector<float> m_trainbuf_output;
    std::vector<float> m_trainbuf_loss;
    std::vector<float> m_trainbuf_diff_fc1_w;
    std::vector<float> m_trainbuf_diff_fc1_b;
    std::vector<float> m_trainbuf_diff_fc2_w;
    std::vector<float> m_trainbuf_diff_fc2_b;
    std::vector<float> m_trainbuf_diff_fc3_w;
    std::vector<float> m_trainbuf_diff_fc3_b;

    // Network for inference
    std::vector<dnnl::primitive> m_infer_net;
    std::vector<std::unordered_map<int, dnnl::memory>> m_infer_net_args;

    // Inference memory handles
    std::vector<dnnl::memory> m_infer_mem;
    int m_infer_memidx_input;
    int m_infer_memidx_output;
    int m_infer_memidx_param_fc1_w;
    int m_infer_memidx_param_fc1_b;
    int m_infer_memidx_param_fc2_w;
    int m_infer_memidx_param_fc2_b;
    int m_infer_memidx_param_fc3_w;
    int m_infer_memidx_param_fc3_b;

    // Inference buffers
    std::vector<float> m_inferbuf_output;

    size_t m_train_batch_size;
    size_t m_infer_batch_size;

    void rebuild_train_network(size_t batch_size);
    void rebuild_infer_network(size_t batch_size);

    inline void apply_gradient(std::vector<float> &buf, const std::vector<float> &diffbuf) {
        for (std::size_t i = 0; i < buf.size(); ++i)
            buf[i] -= m_sgd_learning_rate * diffbuf[i];
    }
public:
    mnist_model(dnnl::engine &engine);
    ~mnist_model();

    void train(const float *data, const float *labels, size_t batch_size);
    void infer(const float *data, uint8_t *category, size_t batch_size);

    void decay_learning_rate(float scale);
};

#endif // BM_MNIST_MODEL_HPP
