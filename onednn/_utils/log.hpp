/**
 * log.hpp
 *
 * Logging utilities
 */

#ifndef DNNUTILS_LOG
#define DNNUTILS_LOG

#include <iostream>

namespace dnnutils {

template<typename T>
inline void log_single(std::ostream &os, const T &msg)
{
    os << msg;
}

template<typename T>
inline void log_single(std::ostream &os, const std::vector<T> &msgs)
{
    bool if_first = true;
    os << "{";
    for (const T &msg : msgs) {
        if (!if_first) {
            os << ", " << msg;
        } else {
            os << msg;
            if_first = false;
        }
    }
    os << "}";
}

template<typename TLast>
void log(std::ostream &os, const TLast &msg)
{
    log_single(os, msg);
    os << std::endl;
}

template<typename TFirst, typename TNext, typename... TRest>
void log(std::ostream &os, const TFirst &f_msg, const TNext &n_msg, const TRest &...r_msgs)
{
    log_single(os, f_msg);
    log(os, n_msg, r_msgs...);
}

template<typename... TArgs>
inline void stdout_log(const TArgs &...msgs)
{
    log(std::cout, msgs...);
}

template<typename... TArgs>
inline void stderr_log(const TArgs &...msgs)
{
    log(std::cerr, msgs...);
}

} // namespace dnnutils


#endif // DNNUTILS_LOG
