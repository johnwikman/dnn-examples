#ifndef GDVERIFY_TEST_INNER_PRODUCT_HPP
#define GDVERIFY_TEST_INNER_PRODUCT_HPP

#include <vector>

#include <dnnl.h>
#include <oneapi/dnnl/dnnl.hpp>

#include <dnnutils.hpp>

std::vector<float> test_inner_product_backprop(dnnl::engine &engine, const dnnutils::labelled_imgset<float> &dataset);
std::vector<float> test_inner_product_fixedstep(const dnnutils::labelled_imgset<float> &dataset);

#endif
