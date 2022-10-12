/**
 * main.cpp
 */

#include <filesystem>
#include <iostream>
#include <string>

#include <dnnutils.hpp>

namespace fs = std::filesystem;

#define DEBUG(msgs...) dnnutils::log::debug_fl(__FILE__, __LINE__, msgs)
#define INFO(msgs...)  dnnutils::log::info_fl(__FILE__, __LINE__, msgs)
#define ERROR(msgs...) dnnutils::log::error_fl(__FILE__, __LINE__, msgs)

int main(int argc, char *argv[])
{
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " [data_dir]" << std::endl;
        return 1;
    }

    dnnutils::log::add_output_stream(std::cout);
    dnnutils::log::enable_bash_color();

    INFO("Test logging! The number is ", 512);
    DEBUG("Test logging! The number is ", 512);
    ERROR("Test logging! The number is ", 512);

    {
        auto mnist = dnnutils::dataset::mnist<float>::load_training(argv[1]);

        std::cout << "[Training set]" << "  "
                  << "N: " << mnist.N() << " | "
                  << "C: " << mnist.C() << " | "
                  << "H: " << mnist.H() << " | "
                  << "W: " << mnist.W() << " | "
                  << "item_size: " << mnist.item_size() << std::endl;
    }

    {
        auto mnist = dnnutils::dataset::mnist<float>::load_validation(argv[1]);

        std::cout << "[Validation set]" << "  "
                  << "N: " << mnist.N() << " | "
                  << "C: " << mnist.C() << " | "
                  << "H: " << mnist.H() << " | "
                  << "W: " << mnist.W() << " | "
                  << "item_size: " << mnist.item_size() << std::endl;
    }
}
