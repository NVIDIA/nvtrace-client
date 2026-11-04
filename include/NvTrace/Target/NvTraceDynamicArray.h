/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/Target/NvTraceMemory.h>

namespace nv::trace {

    template <typename TElement>
    class DynamicArray
    {
    public:
        ~DynamicArray()
        {
            Reset();
        }

        DynamicArray() = default;

        explicit DynamicArray(size_t size)
            : m_pElements(size != 0 ? NewArray<TElement>(size) : nullptr)
            , m_size(m_pElements ? size : 0)
        {
        }

        DynamicArray(const DynamicArray&) = delete;
        DynamicArray(DynamicArray&& other) noexcept
            : m_pElements(std::exchange(other.m_pElements, nullptr))
            , m_size(std::exchange(other.m_size, 0))
        {
        }

        DynamicArray& operator=(const DynamicArray&) = delete;
        DynamicArray& operator=(DynamicArray&& other) noexcept
        {
            if (this != &other)
            {
                Reset();
                m_pElements = std::exchange(other.m_pElements, nullptr);
                m_size = std::exchange(other.m_size, 0);
            }
            return *this;
        }

        void Reset() noexcept
        {
            if (m_size != 0)
            {
                DeleteArray<TElement>(m_pElements, m_size);
                m_pElements = nullptr;
                m_size = 0;
            }
        }

        // Note: this function does not preserve the existing elements, all new elements are default-constructed
        void Resize(size_t size)
        {
            Reset();
            if (size != 0)
            {
                m_pElements = NewArray<TElement>(size);
                m_size = m_pElements ? size : 0;
            }
        }

        TElement& operator[](size_t index)
        {
            return m_pElements[index];
        }
        const TElement& operator[](size_t index) const
        {
            return m_pElements[index];
        }

        TElement* begin()
        {
            return m_pElements;
        }
        const TElement* begin() const
        {
            return m_pElements;
        }
        TElement* end()
        {
            return m_pElements + m_size;
        }
        const TElement* end() const
        {
            return m_pElements + m_size;
        }

        TElement* Data()
        {
            return m_pElements;
        }
        const TElement* Data() const
        {
            return m_pElements;
        }

        size_t Size() const
        {
            return m_size;
        }

        explicit operator bool() const
        {
            return m_size != 0;
        }

    private:
        TElement* m_pElements = nullptr;
        size_t m_size = 0;
    };

} //namespace nv::trace
