/**
 * mnist.hpp
 */

#ifndef DNNUTILS_DATASETS_MNIST_HPP
#define DNNUTILS_DATASETS_MNIST_HPP

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>

#include "../labelled_imgset.hpp"
#include "../tools/decode.hpp"

namespace dnnutils {
namespace dataset {

template<typename T>
class mnist : public labelled_imgset<T> {
public:
    using size_type = typename labelled_imgset<T>::size_type;
    using label_type = typename labelled_imgset<T>::label_type;

    mnist(const std::filesystem::path& data_file,
          const std::filesystem::path& label_file)
    {
        size_type data_size = std::filesystem::file_size(data_file);
        size_type label_size = std::filesystem::file_size(label_file);

        if (data_size < 16)
            throw std::out_of_range("MNIST data_size is less than 16");
        if (data_size < 8)
            throw std::out_of_range("MNIST label_size is less than 8");

        std::unique_ptr<std::byte[]> data_buf(new std::byte[data_size]);
        std::unique_ptr<std::byte[]> label_buf(new std::byte[label_size]);

        /* Read data_file */ {
            std::ifstream ifs(data_file, std::ios::binary);
            ifs.read((char *) data_buf.get(), data_size);
        }

        /* Read label_file */ {
            std::ifstream ifs(label_file, std::ios::binary);
            ifs.read((char *) label_buf.get(), label_size);
        }

        if (dnnutils::tools::frombe32(&(data_buf.get())[0]) != 0x00000803)
            throw std::invalid_argument("invalid magic number for MNIST data");
        if (dnnutils::tools::frombe32(&(label_buf.get())[0]) != 0x00000801)
            throw std::invalid_argument("invalid magic number for MNIST labels");

        size_type data_images = dnnutils::tools::frombe32(&(data_buf.get())[4]);
        size_type data_rows = dnnutils::tools::frombe32(&(data_buf.get())[8]);
        size_type data_columns = dnnutils::tools::frombe32(&(data_buf.get())[12]);

        size_type label_items = dnnutils::tools::frombe32(&(label_buf.get())[4]);

        size_type predict_data_size = 16 + data_images*data_rows*data_columns;
        size_type predict_label_size = 8 + label_items;

        if (predict_data_size != data_size)
            throw std::invalid_argument("mismatch for MNIST data between file size and metadata");
        if (predict_label_size != label_size)
            throw std::invalid_argument("mismatch for MNIST data between file size and metadata");

        if (data_images != label_items)
            throw std::invalid_argument("mismatch between number of MNIST images and labels");

        // Instatiate the underlying dataset
        labelled_imgset<T>::initialize(data_images, 1, data_rows, data_columns);

        // Fill the dataset with images
        labelled_imgset<T>::append_rawpixel_NCHW(&(data_buf.get())[16], data_images*data_rows*data_columns);

        // Fill it with labels
        for (size_type i = 0; i < label_items; ++i)
            labelled_imgset<T>::m_labels[i] = (size_type) label_buf[8 + i];
    }

    // Load training data contained in the specified directory
    static mnist load_training(const std::filesystem::path& dataset_dir)
    {
        return mnist(dataset_dir / "train-images-idx3-ubyte",
                     dataset_dir / "train-labels-idx1-ubyte");
    }

    // Load validation data contained in the specified directory
    static mnist load_validation(const std::filesystem::path& dataset_dir)
    {
        return mnist(dataset_dir / "t10k-images-idx3-ubyte",
                     dataset_dir / "t10k-labels-idx1-ubyte");
    }
};

} // namespace dataset
} // namespace dnnutils

#endif // DNNUTILS_DATASETS_MNIST_HPP
