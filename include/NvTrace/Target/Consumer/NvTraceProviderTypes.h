/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/DataFormats/nvtrace_common_types.h>
#include <NvTrace/DataFormats/nvtrace_cpu_clock_calibration.h>
#include <NvTrace/DataFormats/nvtrace_cuda_api_trace.h>
#include <NvTrace/DataFormats/nvtrace_gpu_clock_calibration.h>
#include <NvTrace/DataFormats/nvtrace_gpu_grctx_trace.h>
#include <NvTrace/DataFormats/nvtrace_gpu_hw_trace.h>
#include <NvTrace/DataFormats/nvtrace_gpu_pm_counters.h>
#include <NvTrace/DataFormats/nvtrace_nvtx_trace.h>
#include <NvTrace/DataFormats/nvtrace_thread_event.h>
#include <NvTrace/Target/NvTraceUuidUtilities.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <iterator>

namespace nv::trace::Codec {

    // For built-in providers only
    // Third party or plug-in providers evaluate to ProviderType::Unknown and must use their UUID
    // Should not be serialized
    enum class ProviderType
    {
        Invalid,
        Unknown,
        CpuClockCalibration,
        GpuClockCalibration,
        GpuPmCounters,
        GpuHwTrace,
        GpuGrCtxTrace,
        CudaApiTrace,
        NvtxTrace,
        ThreadEvent,
        Count,
    };

    template <typename T>
    using ProviderTypeTable = std::array<T, static_cast<size_t>(ProviderType::Count)>;

} // namespace nv::trace::Codec

namespace nv::trace {

    // Indexed by Codec::ProviderType
    inline constexpr Codec::ProviderTypeTable<const char*> ProviderTypeNames = {
        "Invalid",
        "Unknown",
        "CPU Clock Calibration",
        "GPU Clock Calibration",
        "GPU PM Counters",
        "GPU HW Trace",
        "GPU GR Context Trace",
        "CUDA API Trace",
        "NVTX Trace",
        "Thread Event",
    };

    struct ProviderTypeUuidMapping
    {
        NVTRACE_UUID providerTypeUuid;
        Codec::ProviderType providerType;
    };

    // Entries are sorted into a static array at runtime, ordering can be changed freely
    // Table indices should not be serialized
    inline constexpr ProviderTypeUuidMapping ProviderTypeUuidMappingTable[] = {
        {                                    NVTRACE_NIL_UUID,             Codec::ProviderType::Invalid},
        {NvTrace_CPU_ClockCalibration_TscRtc_ProviderTypeUuid, Codec::ProviderType::CpuClockCalibration},
        {NvTrace_GPU_ClockCalibration_TscGpu_ProviderTypeUuid, Codec::ProviderType::GpuClockCalibration},
        {             NvTrace_GPU_PmCounters_ProviderTypeUuid,       Codec::ProviderType::GpuPmCounters},
        {                NvTrace_GPU_HwTrace_ProviderTypeUuid,          Codec::ProviderType::GpuHwTrace},
        {             NvTrace_GPU_GrCtxTrace_ProviderTypeUuid,       Codec::ProviderType::GpuGrCtxTrace},
        {              NvTrace_CUDA_ApiTrace_ProviderTypeUuid,        Codec::ProviderType::CudaApiTrace},
        {                  NvTrace_NvtxTrace_ProviderTypeUuid,           Codec::ProviderType::NvtxTrace},
        {                NvTrace_ThreadEvent_ProviderTypeUuid,         Codec::ProviderType::ThreadEvent},
    };

    inline constexpr size_t ProviderTypeUuidMappingCount = std::size(ProviderTypeUuidMappingTable);
    inline constexpr size_t InvalidProviderTypeUuidTableIndex = 0;

    template <typename T>
    using ProviderTypeUuidTable = std::array<T, ProviderTypeUuidMappingCount>;

    namespace Detail {

        inline const ProviderTypeUuidTable<ProviderTypeUuidMapping>& GetSortedProviderTypeUuidMappingTable()
        {
            static const ProviderTypeUuidTable<ProviderTypeUuidMapping> sortedProviderTypeUuidMappingTable = [] {
                ProviderTypeUuidTable<ProviderTypeUuidMapping> sortedTable = {};
                std::copy(
                    std::begin(ProviderTypeUuidMappingTable),
                    std::end(ProviderTypeUuidMappingTable),
                    sortedTable.begin());
                std::sort(sortedTable.begin(), sortedTable.end(), [](const ProviderTypeUuidMapping& lhs, const ProviderTypeUuidMapping& rhs) {
                    return lhs.providerTypeUuid < rhs.providerTypeUuid;
                });
                return sortedTable;
            }();
            return sortedProviderTypeUuidMappingTable;
        }

    } // namespace Detail

    inline size_t GetProviderTypeUuidTableIndex(const NVTRACE_UUID& providerTypeUuid)
    {
        const ProviderTypeUuidTable<ProviderTypeUuidMapping>& sortedTable = Detail::GetSortedProviderTypeUuidMappingTable();
        const auto itrMapping = std::lower_bound(
            sortedTable.begin(),
            sortedTable.end(),
            providerTypeUuid,
            [](const ProviderTypeUuidMapping& mapping, const NVTRACE_UUID& uuid) {
                return mapping.providerTypeUuid < uuid;
            });
        if (itrMapping != sortedTable.end() && itrMapping->providerTypeUuid == providerTypeUuid)
        {
            return static_cast<size_t>(itrMapping - sortedTable.begin());
        }
        return InvalidProviderTypeUuidTableIndex;
    }

    inline Codec::ProviderType GetProviderType(const NVTRACE_UUID& providerTypeUuid)
    {
        const size_t providerTypeUuidTableIndex = GetProviderTypeUuidTableIndex(providerTypeUuid);
        if (providerTypeUuidTableIndex == InvalidProviderTypeUuidTableIndex && providerTypeUuid != NVTRACE_NIL_UUID)
        {
            return Codec::ProviderType::Unknown;
        }
        return Detail::GetSortedProviderTypeUuidMappingTable()[providerTypeUuidTableIndex].providerType;
    }

    inline const char* GetProviderTypeName(Codec::ProviderType providerType)
    {
        const size_t providerTypeIndex = static_cast<size_t>(providerType);
        if (providerTypeIndex < ProviderTypeNames.size())
        {
            return ProviderTypeNames[providerTypeIndex];
        }
        return ProviderTypeNames[static_cast<size_t>(Codec::ProviderType::Invalid)];
    }

} // namespace nv::trace
