/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/DataFormats/nvtrace_common_types.h>

#include <cstdint>
#include <numeric>


namespace nv::trace {

    // Precondition: modulus != 0
    inline uint32_t CircularIncrement(uint32_t value, uint32_t modulus)
    {
        uint32_t nextValue = value + 1;
        return nextValue == modulus ? 0 : nextValue;
    }

    // Precondition: modulus != 0
    inline uint32_t CircularDecrement(uint32_t value, uint32_t modulus)
    {
        return (value == 0 ? modulus : value) - 1;
    }

    // Precondition: value != 0
    inline uint32_t FloorLog2(uint32_t value)
    {
#if defined(__GNUC__) || defined(__clang__)
        return static_cast<uint32_t>(31 - __builtin_clz(value));
#endif
    }

    // Precondition: value != 0
    inline uint32_t FloorLog2(uint64_t value)
    {
#if defined(__GNUC__) || defined(__clang__)
        return static_cast<uint32_t>(63 - __builtin_clzll(value));
#endif
    }

    // Precondition: value != 0
    inline uint32_t CeilLog2(uint32_t value)
    {
        return value == 1 ? 0 : FloorLog2(value - 1) + 1;
    }

    // Precondition: value != 0
    inline uint32_t CeilLog2(uint64_t value)
    {
        return value == 1 ? 0 : FloorLog2(value - 1) + 1;
    }

    template <typename TValue, typename TAlign>
    inline constexpr TValue AlignDn(TValue value, TAlign align)
    {
        const TValue alignAsTValue = static_cast<TValue>(align);
        return (value / alignAsTValue) * alignAsTValue;
    }

    template <typename TValue, typename TAlign>
    inline constexpr TValue AlignUp(TValue value, TAlign align)
    {
        const TValue alignAsTValue = static_cast<TValue>(align);
        return ((value + alignAsTValue - 1) / alignAsTValue) * alignAsTValue;
    }

    // Precondition: align is a power-of-2
    template <typename TValue, typename TAlign>
    inline constexpr TValue AlignDnP2(TValue value, TAlign align)
    {
        const TValue alignAsTValue = static_cast<TValue>(align);
        return value & ~(alignAsTValue - 1);
    }

    // Precondition: align is a power-of-2
    template <typename TValue, typename TAlign>
    inline constexpr TValue AlignUpP2(TValue value, TAlign align)
    {
        const TValue alignAsTValue = static_cast<TValue>(align);
        return (value + alignAsTValue - 1) & ~(alignAsTValue - 1);
    }

    template <class TElement>
    constexpr uint64_t MinElementCountToFillWholeCachelines()
    {
        constexpr uint64_t cachelineSize = NVTRACE_CPU_CACHELINE_SIZE;
        constexpr uint64_t elementSize = sizeof(TElement);
        return std::lcm(elementSize, cachelineSize) / elementSize;
    }

    template <class TElement>
    constexpr uint64_t RoundUpElementCountToFillWholeCachelines(uint64_t elementCount)
    {
        constexpr uint64_t minElementCountToFillWholeCachelines = MinElementCountToFillWholeCachelines<TElement>();
        return AlignUp(elementCount, minElementCountToFillWholeCachelines);
    }

} //namespace nv::trace
