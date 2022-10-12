/**
 * labelled_imgset.hpp
 * 
 * Specified a framework for a dataset of labelled images.
 */

#ifndef DNNUTILS_LABELLED_IMGSET_HPP
#define DNNUTILS_LABELLED_IMGSET_HPP

#include <cstdint>
#include <memory>
#include <stdexcept>

#include "./labelled_dataset.hpp"

namespace dnnutils {

/**
 * A class for handling labelled image datasets. All images are stored in CHW
 * format. I.e. grouped by channel, then rows, each row containing individual
 * column pixels.
 */
template<typename T>
class labelled_imgset : public labelled_dataset<T> {
public:
    using size_type = typename labelled_dataset<T>::size_type;
    using label_type = typename labelled_dataset<T>::label_type;
protected:
    // Member variables
    size_type m_channels;
    size_type m_height;
    size_type m_width;

    labelled_imgset(): labelled_dataset<T>() {/* initialize() must be called by subclass! */}

    void initialize(size_type capacity,
                    size_type n_channels,
                    size_type n_height,
                    size_type n_width)
    {
        // Sanity check
        if (capacity < 0) throw std::out_of_range("negative item capacity");
        if (n_channels < 1) throw std::out_of_range("non-positive channel count");
        if (n_height < 1) throw std::out_of_range("non-positive image height");
        if (n_width < 1) throw std::out_of_range("non-positive image width");

        labelled_dataset<T>::m_itemcount = 0;
        labelled_dataset<T>::m_capacity = capacity;
        m_channels = n_channels;
        m_height = n_height;
        m_width = n_width;

        labelled_dataset<T>::m_itemsize = n_channels * n_height * n_width;

        size_type data_alloc = labelled_dataset<T>::m_capacity * labelled_dataset<T>::m_itemsize;
        size_type label_alloc = labelled_dataset<T>::m_capacity;

        labelled_dataset<T>::m_data = std::unique_ptr<T[]>(new T[data_alloc]);
        labelled_dataset<T>::m_labels = std::unique_ptr<label_type[]>(new label_type[label_alloc]);
    }

public:
    // Size accessors
    size_type channels() const {return m_channels;}
    size_type height() const {return m_height;}
    size_type width() const {return m_width;}

    // Short-hand to the accessors above
    size_type C() const {return m_channels;}
    size_type H() const {return m_height;}
    size_type W() const {return m_width;}

protected:
    // Convert raw pixels where images are stored in sequence as
    // <channel 1 HxW><channel 2 HxW>...<channel C HxW>
    void append_rawpixel_NCHW(std::byte* buf, size_type bufsize);
};

// Implement the loading utilities
template<> void labelled_imgset<float>::append_rawpixel_NCHW(std::byte* buf, size_type bufsize);
template<> void labelled_imgset<double>::append_rawpixel_NCHW(std::byte* buf, size_type bufsize);

} // namespace dnnutils

#endif // DNNUTILS_LABELLED_IMGSET_HPP
