/**
 * labelled_imgset.cpp
 * 
 * type-specific imgset implementations
 */

#include <stdexcept>

#include <dnnutils/labelled_imgset.hpp>

namespace dnnutils {


template<typename T>
static constexpr T from_rawpixel(std::byte pixel);

template<>
constexpr float from_rawpixel<float>(std::byte pixel)
{
    return (float) pixel / 255.0f;
}

template<>
constexpr double from_rawpixel<double>(std::byte pixel)
{
    return (double) pixel / 255.0;
}


template<typename T>
static inline typename labelled_imgset<T>::size_type generic_append_rawpixel_NCHW(
    std::byte* buf,
    typename labelled_imgset<T>::size_type bufsize,
    T* m_data,
    typename labelled_imgset<T>::size_type m_itemsize,
    typename labelled_imgset<T>::size_type m_capacity,
    typename labelled_imgset<T>::size_type m_itemcount)
{
    typename labelled_imgset<T>::size_type contained_items = bufsize / m_itemsize;
    typename labelled_imgset<T>::size_type remaininder = bufsize % m_itemsize;

    if (remaininder != 0)
        throw std::invalid_argument("bufsize is not evenly divisible by the item size");

    if (contained_items + m_itemcount > m_capacity)
        throw std::invalid_argument("cannot the amount of items that are provided");

    typename labelled_imgset<T>::size_type offset = m_itemcount * m_itemsize;
    for (typename labelled_imgset<T>::size_type i = 0; i < bufsize; ++i) {
        m_data[offset + i] = from_rawpixel<T>(buf[i]); //(float) buf[i] / 255.0;
    }

    return contained_items;
}

template<>
void labelled_imgset<float>::append_rawpixel_NCHW(std::byte* buf, size_type bufsize)
{
    m_itemcount += generic_append_rawpixel_NCHW<float>(buf, bufsize, m_data.get(), m_itemsize, m_capacity, m_itemcount);
}

template<>
void labelled_imgset<double>::append_rawpixel_NCHW(std::byte* buf, size_type bufsize)
{
    m_itemcount += generic_append_rawpixel_NCHW<double>(buf, bufsize, m_data.get(), m_itemsize, m_capacity, m_itemcount);
}

} // namespace dnnutils
