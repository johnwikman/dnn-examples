/**
 * log.hpp - logging output callbacks
 */

#ifndef DNNUTILS_LOG_LOG_HPP
#define DNNUTILS_LOG_LOG_HPP

#include <sstream>
#include <string>

namespace dnnutils {
namespace log {

void _debug(const std::string& msg);
void _debug(const char* file, int line, const std::string& msg);
void _info(const std::string& msg);
void _info(const char* file, int line, const std::string& msg);
void _error(const std::string& msg);
void _error(const char* file, int line, const std::string& msg);

template<typename... Ts>
static void debug(const Ts& ...msgs)
{
    std::ostringstream oss;
    ([&] {oss << msgs;} (), ...);
    _debug(oss.str());
}

template<typename... Ts>
static void debug_fl(const char* file, int line, const Ts& ...msgs)
{
    std::ostringstream oss;
    ([&] {oss << msgs;} (), ...);
    _debug(file, line, oss.str());
}

template<typename... Ts>
static void info(const Ts& ...msgs)
{
    std::ostringstream oss;
    ([&] {oss << msgs;} (), ...);
    _info(oss.str());
}

template<typename... Ts>
static void info_fl(const char* file, int line, const Ts& ...msgs)
{
    std::ostringstream oss;
    ([&] {oss << msgs;} (), ...);
    _info(file, line, oss.str());
}

template<typename... Ts>
static void error(const Ts& ...msgs)
{
    std::ostringstream oss;
    ([&] {oss << msgs;} (), ...);
    _error(oss.str());
}

template<typename... Ts>
static void error_fl(const char* file, int line, const Ts& ...msgs)
{
    std::ostringstream oss;
    ([&] {oss << msgs;} (), ...);
    _error(file, line, oss.str());
}


} // namespace log
} // namespace dnnutils

#endif // DNNUTILS_LOG_LOG_HPP
