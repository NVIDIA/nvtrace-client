/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/DataFormats/nvtrace_common_types.h>
#include <NvTrace/Target/NvTraceCpuTimestamp.h>
#include <NvTrace/Target/NvTraceMathUtilities.h>

namespace nv::trace::Overhead {

    static constexpr int HistogramDurationPreshift = 3;

    /// Increments a histogram bucket spaced apart by powers of 2^(1/3) ~= 1.26.
    inline void IncrementHistogramBucket(NvTrace_CpuTimingEntry& timingEntry, const uint64_t durationTsc)
    {
        constexpr uint64_t maxDelta = 2642245; // = (2^64 - 1)^(1/3)
        const uint64_t delta = durationTsc >> HistogramDurationPreshift;
        // note: assume delta > 0; should be true in practice for any realistic timing range.
        if (maxDelta < delta) // durations that exceed the histogram's capacity increment the topmost bucket
        {
            timingEntry.durationHistogram[NVTRACE_TIMING_HISTOGRAM_COUNT - 1] += 1;
        }
        else if (0 < delta)
        {
            const uint64_t deltaStretched = (delta * delta * delta);
            const int log2DeltaStretched = FloorLog2(deltaStretched);
            timingEntry.durationHistogram[log2DeltaStretched] += 1;
        }
        else
        {
            timingEntry.durationHistogram[0] += 1;
        }
    }

    /// A scoped timer that updates an NvTrace_CpuTimingEntry in its destructor.
    class ScopedTimer
    {
    public:
        ~ScopedTimer()
        {
            const uint64_t endTimestampTsc = ReadCpuTscTimestamp();
            const uint64_t durationTsc = endTimestampTsc - m_startTimestampTsc;
            IncrementHistogramBucket(m_timingEntry, durationTsc);
        }
        ScopedTimer(NvTrace_CpuTimingEntry& timingEntry)
            : m_timingEntry(timingEntry)
        {
            m_startTimestampTsc = ReadCpuTscTimestamp();
        }

    private:
        uint64_t m_startTimestampTsc;
        NvTrace_CpuTimingEntry& m_timingEntry;
    };

    /// A timer that is explicitly started and stopped, updating an NvTrace_CpuTimingEntry on each Stop().
    class ExplicitTimer
    {
    public:
        ~ExplicitTimer()
        {
            if (m_startTimestampTsc)
            {
                Stop();
            }
        }
        ExplicitTimer(NvTrace_CpuTimingEntry& timingEntry)
            : m_timingEntry(timingEntry)
        {
        }

        void Start()
        {
            m_startTimestampTsc = ReadCpuTscTimestamp();
        }

        void Stop()
        {
            const uint64_t endTimestampTsc = ReadCpuTscTimestamp();
            const uint64_t durationTsc = endTimestampTsc - m_startTimestampTsc;
            IncrementHistogramBucket(m_timingEntry, durationTsc);
            m_startTimestampTsc = 0;
        }

    private:
        uint64_t m_startTimestampTsc = 0;
        NvTrace_CpuTimingEntry& m_timingEntry;
    };

    /// All the information needed to convert CPU overhead timings into a human readable format.
    struct CpuOverheadTimings
    {
        size_t codeLocationCount = 0;
        const NvTrace_CpuTimingDescriptor* pTimingDescriptors = nullptr;
        const NvTrace_CpuTimingEntry* pTimingEntries = nullptr;
    };

} // namespace nv::trace::Overhead


#define NVTRACE_CPU_OVERHEAD_ENUM(codeLocation_, name_) codeLocation_,
#define NVTRACE_CPU_OVERHEAD_NAME(codeLocation_, name_) name_,
