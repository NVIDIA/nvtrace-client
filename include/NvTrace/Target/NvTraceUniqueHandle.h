/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace nv::trace {

    template <typename THandle, std::conditional_t<std::is_same_v<THandle, void*>, uintptr_t, THandle> TNoneValue, typename TDeleter>
    class UniqueHandle : private TDeleter
    {
    public:
        ~UniqueHandle()
        {
            reset();
        }

        explicit UniqueHandle(THandle handle = (THandle)TNoneValue, TDeleter deleter = TDeleter()) noexcept
            : TDeleter(deleter)
            , m_handle(handle)
        {
        }

        UniqueHandle(const UniqueHandle&) = delete;
        UniqueHandle(UniqueHandle&& other) noexcept
            : TDeleter(std::move(other))
            , m_handle(std::exchange(other.m_handle, (THandle)TNoneValue))
        {
        }

        UniqueHandle& operator=(const UniqueHandle&) = delete;
        UniqueHandle& operator=(UniqueHandle&& other) noexcept
        {
            if (this != &other)
            {
                reset();
                TDeleter::operator=(std::move(other));
                m_handle = std::exchange(other.m_handle, (THandle)TNoneValue);
            }

            return *this;
        }

        void reset(THandle handle = (THandle)TNoneValue) noexcept
        {
            if (m_handle != (THandle)TNoneValue)
            {
                TDeleter::operator()(m_handle);
            }

            m_handle = handle;
        }

        THandle release() noexcept
        {
            return std::exchange(m_handle, (THandle)TNoneValue);
        }

        THandle* reset_and_get_address_of() noexcept
        {
            reset();
            return &m_handle;
        }

        operator THandle() const noexcept
        {
            return m_handle;
        }

        operator bool() const noexcept
        {
            return m_handle != (THandle)TNoneValue;
        }

    private:
        THandle m_handle;
    };

} //namespace nv::trace
