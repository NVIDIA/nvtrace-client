/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/Target/NvTraceBuildMacros.h>

#include <cstddef>
#include <cstdlib>
#include <limits>
#include <new>
#include <utility>

namespace nv::trace {

    template <typename TObject, typename... TArgs>
    TObject* New(TArgs&&... args)
    {
#if NVTRACE_IS_OPERATOR_NEW_ALLOWED()
        return new (std::nothrow) TObject(std::forward<TArgs>(args)...);
#else
        TObject* pMemory = (TObject*)malloc(sizeof(TObject));
        return pMemory ? new (pMemory) TObject(std::forward<TArgs>(args)...) : nullptr;
#endif
    }

    template <typename TElement>
    TElement* NewArray(size_t size)
    {
#if NVTRACE_IS_OPERATOR_NEW_ALLOWED()
        return new (std::nothrow) TElement[size];
#else
        if (size > std::numeric_limits<size_t>::max() / sizeof(TElement))
        {
            return nullptr;
        }
        TElement* pMemory = (TElement*)malloc(sizeof(TElement) * size);
        if (!pMemory)
        {
            return nullptr;
        }
        for (size_t index = 0; index < size; ++index)
        {
            new (&pMemory[index]) TElement();
        }
        return pMemory;
#endif
    }

    template <typename TObject>
    void Delete(TObject* pObject)
    {
#if NVTRACE_IS_OPERATOR_NEW_ALLOWED()
        delete pObject;
#else
        if (pObject)
        {
            pObject->~TObject();
            free(pObject);
        }
#endif
    }

    template <typename TElement>
    void DeleteArray(TElement* pElements, size_t size)
    {
#if NVTRACE_IS_OPERATOR_NEW_ALLOWED()
        (void)size;
        delete[] pElements;
#else
        if (pElements)
        {
            for (size_t index = size; index-- > 0;)
            {
                pElements[index].~TElement();
            }
            free(pElements);
        }
#endif
    }


    template <typename TObject>
    class UniquePtr
    {
    public:
        ~UniquePtr()
        {
            Reset();
        }

        UniquePtr() = default;

        UniquePtr(std::nullptr_t)
        {
        }

        explicit UniquePtr(TObject* pObject)
            : m_pObject(pObject)
        {
        }

        UniquePtr(const UniquePtr&) = delete;
        UniquePtr(UniquePtr&& other) noexcept
            : m_pObject(std::exchange(other.m_pObject, nullptr))
        {
        }

        UniquePtr& operator=(const UniquePtr&) = delete;
        UniquePtr& operator=(UniquePtr&& other) noexcept
        {
            if (this != &other)
            {
                Reset(other.m_pObject);
                other.m_pObject = nullptr;
            }

            return *this;
        }

        UniquePtr& operator=(std::nullptr_t)
        {
            Reset();
            return *this;
        }

        void Reset(TObject* pObject = nullptr) noexcept
        {
            if (m_pObject)
            {
                Delete<TObject>(m_pObject);
            }

            m_pObject = pObject;
        }

        TObject* Release()
        {
            return std::exchange(m_pObject, nullptr);
        }

        TObject* operator->() const
        {
            return m_pObject;
        }

        operator TObject*() const
        {
            return m_pObject;
        }

        explicit operator bool() const
        {
            return m_pObject != nullptr;
        }

        friend bool operator==(const UniquePtr<TObject>& lhs, const UniquePtr<TObject>& rhs)
        {
            return lhs.m_pObject == rhs.m_pObject;
        }
        friend bool operator==(const UniquePtr<TObject>& lhs, std::nullptr_t)
        {
            return lhs.m_pObject == nullptr;
        }
        friend bool operator==(std::nullptr_t, const UniquePtr<TObject>& rhs)
        {
            return nullptr == rhs.m_pObject;
        }

        friend bool operator!=(const UniquePtr<TObject>& lhs, const UniquePtr<TObject>& rhs)
        {
            return lhs.m_pObject != rhs.m_pObject;
        }
        friend bool operator!=(const UniquePtr<TObject>& lhs, std::nullptr_t)
        {
            return lhs.m_pObject != nullptr;
        }
        friend bool operator!=(std::nullptr_t, const UniquePtr<TObject>& rhs)
        {
            return nullptr != rhs.m_pObject;
        }

        friend bool operator<(const UniquePtr<TObject>& lhs, const UniquePtr<TObject>& rhs)
        {
            return lhs.m_pObject < rhs.m_pObject;
        }
        friend bool operator<(const UniquePtr<TObject>& lhs, std::nullptr_t)
        {
            return lhs.m_pObject < nullptr;
        }
        friend bool operator<(std::nullptr_t, const UniquePtr<TObject>& rhs)
        {
            return nullptr < rhs.m_pObject;
        }

    private:
        // Smart pointers that solely have the implicit cast 'operator TObject*' would normally
        // compile just fine with the following code, which results in a double-delete error:
        //      UniquePtr<Foo> pFoo = ...;
        //      delete pFoo;                    // want a compile error here
        // Introducing a second 'operator CompileErrorOnOperatorDelete*()' triggers an 'ambiguous conversion to void*'
        // on the 'delete', which catches the usage error at compile time.
        // For full details, read here: http://stackoverflow.com/a/3312507
        struct CompileErrorOnOperatorDelete;
        operator CompileErrorOnOperatorDelete*() const;

    private:
        TObject* m_pObject = nullptr;
    };

} //namespace nv::trace
