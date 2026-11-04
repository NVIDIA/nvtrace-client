/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/DataFormats/nvtrace_common_types.h>

/// Suppress warnings in driver build from flatbuffers/allocator.h:84:11
/// error: argument 1 null where non-null expected [-Werror=nonnull]
/// It's fairly safe since we don't use default allocator
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnonnull"
#endif
#include <NvTrace/Target/NvTraceCommonSerializationTypes_generated.h>
#include <flatbuffers/flatbuffers.h>
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#include <csetjmp>
#include <new>
#include <string.h>
#include <string>

namespace nv::trace {

    /// Context for setjmp/longjmp based error handling.
    ///
    /// This provides exception-like error propagation without using C++ exceptions.
    /// IMPORTANT: When using this mechanism, be aware that:
    ///   - Destructors of local objects between setjmp and longjmp will NOT be called
    ///   - Avoid creating objects with non-trivial destructors in the lambda
    ///   - The lambda should ideally only perform FlatBuffer building operations
    class FlatBufferTryContext
    {
    public:
        ~FlatBufferTryContext() = default;
        FlatBufferTryContext() = default;

        FlatBufferTryContext(const FlatBufferTryContext&) = delete;
        FlatBufferTryContext& operator=(const FlatBufferTryContext&) = delete;


        bool Try()
        {
            return setjmp(m_jmpBuffer) == 0;
        }

        // Performs longjmp back to the setjmp point
        [[noreturn]]
        void Throw()
        {
            longjmp(m_jmpBuffer, 1);
        }


    private:
        jmp_buf m_jmpBuffer;
    };


    class FlatBufferFixedAllocator : public flatbuffers::Allocator
    {
    public:
        FlatBufferFixedAllocator(FlatBufferTryContext& tryContext, uint8_t* pMemory, size_t memorySize)
            : m_tryContext(tryContext)
            , m_pMemory(pMemory)
            , m_memorySize(memorySize)
        {
        }

        uint8_t* allocate(size_t size) override
        {
            if (size > m_memorySize)
            {
                m_tryContext.Throw();
            }

            return m_pMemory + (m_memorySize - size);
        }

        void deallocate(uint8_t* /* p */, size_t /* size */) override
        {
            // no-op
        }

        uint8_t* reallocate_downward(uint8_t* old_p, size_t old_size, size_t new_size, size_t in_use_back, size_t in_use_front) override
        {
            if (new_size > m_memorySize)
            {
                m_tryContext.Throw();
            }

            std::memmove(m_pMemory + (m_memorySize - new_size), m_pMemory + (m_memorySize - old_size), in_use_front);
            return m_pMemory + (m_memorySize - new_size);
        }

    private:
        FlatBufferTryContext& m_tryContext;
        uint8_t* const m_pMemory;
        const size_t m_memorySize;
    };

    class FlatBufferMallocAllocator : public flatbuffers::Allocator
    {
    public:
        FlatBufferMallocAllocator(FlatBufferTryContext& tryContext)
            : m_tryContext(tryContext)
        {
        }

        uint8_t* allocate(size_t size) override
        {
            uint8_t* pMemory = static_cast<uint8_t*>(malloc(size));
            if (!pMemory)
            {
                m_tryContext.Throw();
            }

            return pMemory;
        }

        void deallocate(uint8_t* p, size_t) override
        {
            free(p);
        }

        uint8_t* reallocate_downward(uint8_t* old_p, size_t old_size, size_t new_size, size_t in_use_back, size_t in_use_front) override
        {
            FLATBUFFERS_ASSERT(new_size > old_size); // vector_downward only grows

            uint8_t* new_p = static_cast<uint8_t*>(malloc(new_size));
            if (!new_p)
            {
                free(old_p);
                m_tryContext.Throw();
            }

            memcpy_downward(old_p, old_size, new_p, new_size, in_use_back, in_use_front);
            free(old_p);

            return new_p;
        }

    private:
        FlatBufferTryContext& m_tryContext;
    };

    /// Executes a FlatBuffer building operation with fixed memory allocation using longjmp-based error handling.
    ///
    /// @tparam TFN_BuildFlatBuffer Callable type (typically a lambda).
    /// @param pMemory The pointer to the memory to use for the FlatBufferBuilder.
    /// @param memorySize The size of the memory to use for the FlatBufferBuilder.
    /// @param fnBuildFlatBuffer The function to execute (should contain FlatBuffer building code).
    /// @return The size of the built FlatBuffer if the operation completed successfully, 0 if an error occurred.
    ///
    /// WARNING: The lambda should not create objects with non-trivial destructors, as longjmp
    /// will bypass destructor calls. Keep the lambda focused on FlatBuffer operations only.
    template <typename TFN_BuildFlatBuffer>
    size_t TryBuildFlatBuffer(uint8_t* pMemory, size_t memorySize, TFN_BuildFlatBuffer&& fnBuildFlatBuffer)
    {
        FlatBufferTryContext tryContext;
        if (tryContext.Try())
        {
            FlatBufferFixedAllocator allocator(tryContext, pMemory, memorySize);
            flatbuffers::FlatBufferBuilder builder(memorySize, &allocator);
            fnBuildFlatBuffer(builder);
            return builder.GetSize();
        }
        else
        {
            // Arrived here via longjmp - an error occurred
            return 0;
        }
    }

    /// Typesafe way of passing around FlatBuffer buffers.
    template <typename TFlatBuffer>
    struct FlatBufferView
    {
        const uint8_t* pData = nullptr;
        size_t size = 0;

        bool Verify() const
        {
            return flatbuffers::Verifier(pData, size).VerifyBuffer<TFlatBuffer>();
        }

        const TFlatBuffer* GetRoot() const
        {
            return flatbuffers::GetRoot<TFlatBuffer>(pData);
        }
    };

    template <>
    struct FlatBufferView<void>
    {
        constexpr FlatBufferView() = default;

        constexpr FlatBufferView(const uint8_t* pData, size_t size)
            : pData(pData)
            , size(size)
        {
        }

        template <typename TFlatBuffer>
        constexpr FlatBufferView(const FlatBufferView<TFlatBuffer>& view)
            : pData(view.pData)
            , size(view.size)
        {
        }

        const uint8_t* pData = nullptr;
        size_t size = 0;
    };

    inline NvTrace_SemVer ConvertFBSemVer(const FB::SemVer* pSemVer)
    {
        if (!pSemVer)
        {
            return NvTrace_SemVer{0, 0, 0};
        }

        return NvTrace_SemVer{pSemVer->majorVer(), pSemVer->minorVer(), pSemVer->patchVer()};
    }

} //namespace nv::trace
