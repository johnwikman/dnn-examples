/**
 * mnist data definitions
 */

#ifndef DNNUTILS_MNIST
#define DNNUTILS_MNIST

#include <cstdint>
#include <filesystem>
#include <string>

namespace dnnutils {

class mnist {
public:
    using size_type = std::int32_t;

    mnist(const std::filesystem::path &datadir);
    ~mnist();

    size_type train_length() const;
    size_type t10k_length() const;

    size_type C() const;
    size_type H() const;
    size_type W() const;

    void train_copydata(size_type start, size_type end, float *dst) const;
    void t10k_copydata(size_type start, size_type end, float *dst) const;

    void train_copylabels_1hot(size_type start, size_type end, float *dst) const;
    void t10k_copylabels_1hot(size_type start, size_type end, float *dst) const;
private:
    std::byte *m_train_data;
    std::byte *m_train_labels;
    std::byte *m_t10k_data;
    std::byte *m_t10k_labels;
};

} // namespace dnnutils

#endif
