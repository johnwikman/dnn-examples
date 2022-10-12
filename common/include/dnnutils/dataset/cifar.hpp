/**
 * cifar.hpp - CIFAR-10 and CIFAR-100 dataset loaders
 */

#ifndef DNNUTILS_DATASETS_CIFAR_HPP
#define DNNUTILS_DATASETS_CIFAR_HPP

#include "../labelled_imgset.hpp"

namespace dnnutils {
namespace dataset {

template<typename T>
class cifar10 : public labelled_imgset<T> {
public:
    // TODO
};

} // namespace dataset
} // namespace dnnutils

#endif // DNNUTILS_DATASETS_CIFAR_HPP
