/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#if defined(__GNUC__)
#define NVTRACE_COMPILER_BARRIER() asm volatile("" ::: "memory")
#else
#error "Unsupported compiler."
#endif

#include <cstddef>
#include <cstdint>


namespace nv::trace {

    // One-way barrier load and store for ARMv8-A (AArch64) architecture. See https://developer.arm.com/documentation/102336/0100/Load-Acquire-and-Store-Release-instructions
#if defined(__GNUC__) && defined(__aarch64__)

    template <typename TInteger>
    inline TInteger LoadRelaxed(const volatile TInteger* p)
    {
        return *p;
    }

    inline uint32_t LoadAcquire(const volatile uint32_t* p)
    {
        uint32_t v;
        asm volatile("ldar %w0, [%1]"
                     : "=r"(v)
                     : "r"(p)
                     : "memory");
        return v;
    }
    inline int32_t LoadAcquire(const volatile int32_t* p)
    {
        return static_cast<int32_t>(LoadAcquire(reinterpret_cast<const volatile uint32_t*>(p)));
    }

    inline uint64_t LoadAcquire(const volatile uint64_t* p)
    {
        uint64_t v;
        asm volatile("ldar %x0, [%1]"
                     : "=r"(v)
                     : "r"(p)
                     : "memory");
        return v;
    }
    inline int64_t LoadAcquire(const volatile int64_t* p)
    {
        return static_cast<int64_t>(LoadAcquire(reinterpret_cast<const volatile uint64_t*>(p)));
    }

    template <typename TInteger>
    inline void StoreRelaxed(volatile TInteger* p, TInteger v)
    {
        *p = v;
    }

    inline void StoreRelease(volatile uint32_t* p, uint32_t v)
    {
        asm volatile("stlr %w0, [%1]"
                     :
                     : "r"(v), "r"(p)
                     : "memory");
    }
    inline void StoreRelease(volatile int32_t* p, int32_t v)
    {
        StoreRelease(reinterpret_cast<volatile uint32_t*>(p), static_cast<uint32_t>(v));
    }

    inline void StoreRelease(volatile uint64_t* p, uint64_t v)
    {
        asm volatile("stlr %x0, [%1]"
                     :
                     : "r"(v), "r"(p)
                     : "memory");
    }
    inline void StoreRelease(volatile int64_t* p, int64_t v)
    {
        StoreRelease(reinterpret_cast<volatile uint64_t*>(p), static_cast<uint64_t>(v));
    }

    inline void ReadFence()
    {
        asm volatile("dmb ishld" ::: "memory");
    }

    inline void WriteFence()
    {
        asm volatile("dmb ishst" ::: "memory");
    }

#elif defined(__x86_64__) || defined(NVTRACE_INTERNAL_X64_ARCH) // x64 architecture has strong memory ordering guarantees, so no special instructions are needed for acquire/release semantics

    template <typename TInteger>
    inline TInteger LoadRelaxed(const volatile TInteger* p)
    {
        return *p;
    }

    template <typename TInteger>
    inline TInteger LoadAcquire(const volatile TInteger* p)
    {
        TInteger result = *p;
        NVTRACE_COMPILER_BARRIER();
        return result;
    }

    template <typename TInteger>
    inline void StoreRelaxed(volatile TInteger* p, TInteger v)
    {
        *p = v;
    }

    template <typename TInteger>
    inline void StoreRelease(volatile TInteger* p, TInteger v)
    {
        NVTRACE_COMPILER_BARRIER();
        *p = v;
    }

    inline void ReadFence()
    {
        NVTRACE_COMPILER_BARRIER();
    }

    inline void WriteFence()
    {
        NVTRACE_COMPILER_BARRIER();
    }

#else
#error "MemoryFence.h is not implemented for this architecture"
#endif

    // A memcpy whose loads are 64-bit wide, for copying out of memory a producer may be storing a 64-bit
    // field into in place (see the StateTableReader contract).
    //
    // Precondition: both pointers and the byteCount are 8-byte aligned
    inline void Memcpy64(void* pDest, const volatile void* pSrc, size_t byteCount)
    {
        const volatile uint64_t* const pSrcWords = reinterpret_cast<const volatile uint64_t*>(pSrc);
        uint64_t* const pDestWords = reinterpret_cast<uint64_t*>(pDest);
        const size_t wordCount = byteCount / sizeof(uint64_t);
        for (size_t wordIndex = 0; wordIndex < wordCount; ++wordIndex)
        {
            pDestWords[wordIndex] = pSrcWords[wordIndex];
        }
    }

} // namespace nv::trace
