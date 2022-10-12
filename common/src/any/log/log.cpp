/**
 * All logging functionality is here, even though they are spread out in
 * different header files.
 */

#include <sstream>
#include <vector>

#include <dnnutils/log/config.hpp>
#include <dnnutils/log/log.hpp>

namespace dnnutils {
namespace log {

static std::vector<std::ostream*> _log_ostreams;
static bool _log_bash_color = false;

void add_output_stream(std::ostream& os)
{
    _log_ostreams.push_back(&os);
}

void enable_bash_color(bool enable)
{
    _log_bash_color = enable;
}

enum class loglevel {
    DEBUG, INFO, ERROR
};

const char* _loglevel_name(loglevel level)
{
    switch (level) {
    case loglevel::DEBUG:
        return "debug";
    case loglevel::INFO:
        return "info";
    case loglevel::ERROR:
        return "ERROR";
    default:
        return "???";
    }
}

const char* _loglevel_color(loglevel level)
{
    switch (level) {
    case loglevel::DEBUG:
        return "0;38";
    case loglevel::INFO:
        return "1;36";
    case loglevel::ERROR:
        return "1;31";
    default:
        return "0;37";
    }
}

static void output(loglevel level,
                   const char* file,
                   const char* function,
                   int* line,
                   const std::string& msg)
{
    bool add_space = (file != nullptr) || (function != nullptr) || (line != nullptr);

    for (std::ostream* os : _log_ostreams) {
        if (_log_bash_color) *os << "\033[" << _loglevel_color(level) << "m";
        *os << "[" << _loglevel_name(level) << "]";
        if (_log_bash_color) *os << "\033[0m";
        *os << " ";

        if (file != nullptr) {
            if (_log_bash_color) *os << "\033[0;37m";
            *os << file;
            if (_log_bash_color) *os << "\033[0m";
            *os << ":";
        }
        if (function != nullptr) {
            if (_log_bash_color) *os << "\033[0;37m";
            *os << function;
            if (_log_bash_color) *os << "\033[0m";
            *os << ":";
        }
        if (line != nullptr) {
            if (_log_bash_color) *os << "\033[1;32m";
            *os << *line;
            if (_log_bash_color) *os << "\033[0m";
            *os << ":";
        }

        if (add_space)
            *os << ' ';

        *os << msg << std::endl;
    }
}

void _debug(const std::string& msg)
{
    output(loglevel::DEBUG, nullptr, nullptr, nullptr, msg);
}

void _debug(const char* file, int line, const std::string& msg)
{
    output(loglevel::DEBUG, file, nullptr, &line, msg);
}

void _info(const std::string& msg)
{
    output(loglevel::INFO, nullptr, nullptr, nullptr, msg);
}

void _info(const char* file, int line, const std::string& msg)
{
    output(loglevel::INFO, file, nullptr, &line, msg);
}

void _error(const std::string& msg)
{
    output(loglevel::ERROR, nullptr, nullptr, nullptr, msg);
}

void _error(const char* file, int line, const std::string& msg)
{
    output(loglevel::ERROR, file, nullptr, &line, msg);
}

} // namespace log
} // namespace dnnutils

