/**
 * labelled_dataset.hpp
 */

#ifndef DNNUTILS_LABELLED_DATASET_HPP
#define DNNUTILS_LABELLED_DATASET_HPP

#include <cstdint>
#include <memory>

namespace dnnutils {

/**
 * Class for handling a labelled dataset.
 *
 * This is generally intended to be subclassed, and not instantiated directly.
 * The subclasses are supposed to load the necessary data, while this acts as
 * an accessor interface.
 */
template<typename T>
class labelled_dataset {
public:
    using size_type = std::int64_t;
    using label_type = std::int32_t;

protected:
    // Member variables
    std::unique_ptr<T[]> m_data;
    std::unique_ptr<label_type[]> m_labels;
    size_type m_itemcount;
    size_type m_itemsize;

    size_type m_capacity; // how many items that are allocated for

    /** Constructor */
    labelled_dataset():
        m_data(nullptr),
        m_labels(nullptr),
        m_itemcount(0),
        m_itemsize(0),
        m_capacity(0)
    {
        // Do nothing specific here, let subclass initialize these variables
    }

public:
    /** Cannot copy or move a dataset. Pass by reference instead. */
    labelled_dataset(const labelled_dataset& other) = delete;
    labelled_dataset(labelled_dataset&& other) = delete;

    // Item accessors
    const T* data(size_type idx) const {return &m_data[m_itemsize*idx];}
    label_type label(size_type idx) const {return m_labels[idx];}

    // Size accessors
    size_type count() const {return m_itemcount;}
    size_type item_size() const {return m_itemsize;}

    // Short-hand to the accessors above
    size_type N() const {return m_itemcount;} // same as count()
};


} // namespace dnnutils

#endif // DNNUTILS_LABELLED_DATASET_HPP
