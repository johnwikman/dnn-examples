#ifndef GDVERIFY_TEST_SOFTMAX_HPP
#define GDVERIFY_TEST_SOFTMAX_HPP

#include <vector>

#include <dnnl.h>
#include <oneapi/dnnl/dnnl.hpp>

#include <dnnutils.hpp>

void test_softmax_analytical(dnnl::engine &engine,
                             const dnnutils::labelled_imgset<float> &dataset,
                             std::vector<float>& value_output,
                             std::vector<float>& diff_output);
void test_softmax_numeric(const dnnutils::labelled_imgset<float> &dataset,
                          std::vector<float>& value_output,
                          std::vector<float>& diff_output);

#endif
