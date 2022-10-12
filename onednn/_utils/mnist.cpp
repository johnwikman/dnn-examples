/**
 * load MNIST data into oneDNN
 */

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "mnist.hpp"

namespace fs = std::filesystem;

namespace dnnutils {

/**
 * Loads the contents from the provided path
 */
static mnist::size_type load_file(const fs::path &p, std::byte **dst)
{
    mnist::size_type size = fs::file_size(p);
    std::ifstream ifs(p, std::ios::binary);
    *dst = new std::byte[size];
    ifs.read((char *) *dst, size);
    return size;
}

static inline std::uint32_t frombe32(const std::byte *src)
{
    std::uint32_t v = (((uint8_t) src[0]) << 24)
                    | (((uint8_t) src[1]) << 16)
                    | (((uint8_t) src[2]) << 8)
                    | ((uint8_t) src[3]);
    return v;
}

mnist::mnist(const fs::path &datadir):
    m_train_data(nullptr),
    m_train_labels(nullptr),
    m_t10k_data(nullptr),
    m_t10k_labels(nullptr)
{
    // Set up the paths
    fs::path bf_train_data = datadir / "train-images-idx3-ubyte";
    fs::path bf_train_labels = datadir / "train-labels-idx1-ubyte";
    fs::path bf_t10k_data = datadir / "t10k-images-idx3-ubyte";
    fs::path bf_t10k_labels = datadir / "t10k-labels-idx1-ubyte";

    // Load data (maybe check file existence??)
    load_file(bf_train_data, &m_train_data);
    load_file(bf_train_labels, &m_train_labels);
    load_file(bf_t10k_data, &m_t10k_data);
    load_file(bf_t10k_labels, &m_t10k_labels);

    if (frombe32(m_train_data) != 0x00000803)
        std::cerr << "mismatch on m_train_data!" << std::endl;
    if (frombe32(m_train_labels) != 0x00000801)
        std::cerr << "mismatch on m_train_labels!" << std::endl;
    if (frombe32(m_t10k_data) != 0x00000803)
        std::cerr << "mismatch on m_t10k_data!" << std::endl;
    if (frombe32(m_t10k_labels) != 0x00000801)
        std::cerr << "mismatch on m_t10k_labels!" << std::endl;

    if (frombe32(m_train_data + 8) != 28)
        std::cerr << "Row count mismatch on m_train_data" << std::endl;
    if (frombe32(m_train_data + 12) != 28)
        std::cerr << "Cols count mismatch on m_train_data" << std::endl;
}

mnist::~mnist()
{
    delete[] this->m_train_data;
    delete[] this->m_train_labels;
    delete[] this->m_t10k_data;
    delete[] this->m_t10k_labels;
}

mnist::size_type mnist::train_length() const
{
    return (mnist::size_type) frombe32(m_train_data + 4);
}

mnist::size_type mnist::t10k_length() const
{
    return (mnist::size_type) frombe32(m_t10k_data + 4);
}


mnist::size_type mnist::C() const {return 1;}
mnist::size_type mnist::H() const {return 28;}
mnist::size_type mnist::W() const {return 28;}



static inline void generic_copydata(mnist::size_type start, mnist::size_type end, float *dst, const std::byte *src)
{
    const mnist::size_type stride = 1*28*28;
    // adjust length to be in bytes instead of items
    const mnist::size_type len = (end - start) * stride;
    const mnist::size_type offset = 16 + start*stride;

    for (mnist::size_type i = 0; i < len; ++i)
        dst[i] = ((float) src[offset + i]) / 255.0f;
}

void mnist::train_copydata(mnist::size_type start, mnist::size_type end, float *dst) const
{
    generic_copydata(start, end, dst, m_train_data);
}

void mnist::t10k_copydata(mnist::size_type start, mnist::size_type end, float *dst) const
{
    generic_copydata(start, end, dst, m_t10k_data);
}

static inline void generic_copylabels(mnist::size_type start, mnist::size_type end, uint8_t *dst, const std::byte *labels)
{
    const mnist::size_type len = end - start;
    const mnist::size_type offset = 8;
    for (mnist::size_type i = 0; i < len; ++i)
        dst[i] = (uint8_t) labels[offset + start + i];
}

void mnist::train_copylabels(size_type start, size_type end, uint8_t *dst) const
{
    generic_copylabels(start, end, dst, m_train_labels);
}

void mnist::t10k_copylabels(size_type start, size_type end, uint8_t *dst) const
{
    generic_copylabels(start, end, dst, m_t10k_labels);
}

static inline void generic_copylabels_1hot(mnist::size_type start, mnist::size_type end, float *dst, const std::byte *labels)
{
    const mnist::size_type len = end - start;
    const mnist::size_type offset = 8;
    const mnist::size_type stride = 10;

    // Zero out the array first
    std::memset(dst, 0, len * stride * sizeof(float));

    //
    for (mnist::size_type i = 0; i < len; ++i)
        dst[(i * stride) + (mnist::size_type) labels[offset + start + i]] = 1.0f;
}

void mnist::train_copylabels_1hot(mnist::size_type start, mnist::size_type end, float *dst) const
{
    generic_copylabels_1hot(start, end, dst, m_train_labels);
}

void mnist::t10k_copylabels_1hot(mnist::size_type start, mnist::size_type end, float *dst) const
{
    generic_copylabels_1hot(start, end, dst, m_t10k_labels);
}


} // namespace dnnutils
