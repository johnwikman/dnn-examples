/**
 * main.cpp - Entry point to the MNIST training
 */


#include <iostream>
#include <string>

#include <dnnl.h>
#include <oneapi/dnnl/dnnl.hpp>

#include "model.hpp"
#include "../_utils/utils.hpp"

#define LOG(msgs...) dnnutils::stdout_log(msgs)
#define ERRLOG(msgs...) dnnutils::stderr_log(msgs)


int main(int argc, char *argv[])
{
    const std::string DATA_PATH = "/mnt/_data";

    if (DNNL_CPU_RUNTIME != DNNL_RUNTIME_OMP) {
        ERRLOG("Expected DNNL_CPU_RUNTIME to be OpenMP");
        return 1;
    }
    dnnl::engine engine(dnnl::engine::kind::cpu, 0);

    LOG("Loading dataset from ", DATA_PATH);
    dnnutils::mnist mnist(DATA_PATH);

    LOG("Creating model");
    mnist_model model(engine);

    auto total_epochs = 10;
    auto N = 256;
    auto H = mnist.H();
    auto W = mnist.W();
    auto C = mnist.C();
    auto O = 10;

    std::vector<float> batch_data(N * H * W * C);
    std::vector<float> batch_labels(N * O);
    std::vector<uint8_t> batch_rawlabels(N);
    std::vector<uint8_t> batch_predicts(N);

    LOG("Starting training...");
    for (int epoch = 1; epoch <= total_epochs; ++epoch) {
        LOG("Epoch ", epoch, "/", total_epochs);
        auto start = 0;

        LOG("  Computing accuracy training step");
        start = 0;
        auto correct = 0;
        int32_t mean_predict = 0;
        int32_t max_predict = 0;
        int32_t min_predict = 9;
        while (start < mnist.t10k_length()) {
            auto end = start + N;
            if (end > mnist.t10k_length())
                end = mnist.t10k_length();

            //LOG("Loading batch...");
            mnist.t10k_copydata(start, end, batch_data.data());
            mnist.t10k_copylabels(start, end, batch_rawlabels.data());

            //LOG("Inferring categories...");
            model.infer(batch_data.data(), batch_predicts.data(), end - start);

            auto len = end - start;
            for (auto i = 0; i < len; ++i) {
                if (batch_predicts[i] == batch_rawlabels[i])
                    correct += 1;

                mean_predict += (int32_t) batch_predicts[i];
                if ((int32_t) batch_predicts[i] > max_predict)
                    max_predict = (int32_t) batch_predicts[i];
                if ((int32_t) batch_predicts[i] < min_predict)
                    min_predict = (int32_t) batch_predicts[i];
            }

            start = end;
        }
        LOG("  Accuracy:     ", 100.0 * (double) correct / (double) mnist.t10k_length(), "%");
        LOG("  Mean predict: ", (double) mean_predict / (double) mnist.t10k_length());
        LOG("  Min predict:  ", min_predict);
        LOG("  Max predict:  ", max_predict);

        LOG("  Performing training step");
        start = 0;
        while (start < mnist.train_length()) {
            auto end = start + N;
            if (end > mnist.train_length())
                end = mnist.train_length();

            //LOG("Loading batch...");
            mnist.train_copydata(start, end, batch_data.data());
            mnist.train_copylabels_1hot(start, end, batch_labels.data());

            //LOG("Training model...");
            model.train(batch_data.data(), batch_labels.data(), end - start);

            start = end;
        }

        LOG("  Decaying learning rate");
        model.decay_learning_rate(0.9f);
    }

    LOG("Done");
}


