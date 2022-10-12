/**
 * logging configuration
 */

#ifndef DNNUTILS_LOG_CONFIG_HPP
#define DNNUTILS_LOG_CONFIG_HPP

#include <ostream>

namespace dnnutils {
namespace log {

void add_output_stream(std::ostream& os);
void enable_bash_color(bool enable = true);

} // namespace log
} // namespace dnnutils

#endif // DNNUTILS_LOG_CONFIG_HPP
