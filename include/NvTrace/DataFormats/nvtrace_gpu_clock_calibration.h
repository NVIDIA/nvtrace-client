/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/DataFormats/nvtrace_common_types.h>
#include <NvTrace/DataFormats/nvtrace_provider_buffer.h>


NVTRACE_SUPPRESS_WARNINGS()
#ifdef __cplusplus
extern "C" {
#endif


NVTRACE_WEAK_CONST NVTRACE_UUID NvTrace_GPU_ClockCalibration_TscGpu_ProviderTypeUuid = NVTRACE_UUID_VALUE(0xb5029de1, 0x9e56, 0x48e5, 0xbb, 0xb, 0x7b, 0x49, 0xb, 0x69, 0x5d, 0xd1);


////////////////////////////////////////////////////////////
// Provider-buffer sections

/// Provider-specific section tags used with NvTrace_ProviderBufferSectionDesc::sectionTag.
enum NvTrace_GPU_ClockCalibration_TscGpu_SectionTag
{
    NvTrace_GPU_ClockCalibration_TscGpu_SectionTag_Invalid = 0,

    /// DataBlob section containing NvTrace_GPU_ClockCalibration_TscGpu_RecordVersionInfo.
    NvTrace_GPU_ClockCalibration_TscGpu_SectionTag_RecordVersionInfo = 1,

    /// DataBlob section containing a serialized FlatBuffer (nv.trace.FB.Provider.GpuClockCalibration.ProviderConfiguration).
    NvTrace_GPU_ClockCalibration_TscGpu_SectionTag_ProviderConfiguration = 2,

    /// Gates on the payload's tscTick (recordTimestampType == Tsc).
    NvTrace_GPU_ClockCalibration_TscGpu_SectionTag_CalibrationPointCircularBuffer = 3,

    /// DataBlob section containing NvTrace_GPU_DeviceInfo.
    NvTrace_GPU_ClockCalibration_TscGpu_SectionTag_GpuDeviceInfo = 4,

    NvTrace_GPU_ClockCalibration_TscGpu_SectionTag_FORCE_INT = 0x7FFFFFFF
};

/// Provider-specific record types. Each data-blob section carries a single record of the corresponding type; the
/// calibration-point circular buffer publishes one fixed record type for every cell.
/// Append only -- record types are wire-persisted and never renumbered, so a new record type takes
/// the value after the last.
enum NvTrace_GPU_ClockCalibration_TscGpu_RecordType
{
    NvTrace_GPU_ClockCalibration_TscGpu_RecordType_Invalid = 0,

    /// One NvTrace_GPU_ClockCalibration_TscGpu_RecordVersionInfo payload (RecordVersionInfo DataBlob section).
    NvTrace_GPU_ClockCalibration_TscGpu_RecordType_RecordVersionInfo = 1,

    /// One serialized ProviderConfiguration FlatBuffer (ProviderConfiguration DataBlob section).
    NvTrace_GPU_ClockCalibration_TscGpu_RecordType_ProviderConfiguration = 2,

    /// One NvTrace_GPU_ClockCalibration_TscGpu_CalibrationPoint payload.
    NvTrace_GPU_ClockCalibration_TscGpu_RecordType_CalibrationPoint = 3,

    /// One NvTrace_GPU_DeviceInfo payload (GpuDeviceInfo DataBlob section).
    NvTrace_GPU_ClockCalibration_TscGpu_RecordType_GpuDeviceInfo = 4,

    NvTrace_GPU_ClockCalibration_TscGpu_RecordType_Count,
    NvTrace_GPU_ClockCalibration_TscGpu_RecordType_FORCE_INT = 0x7FFFFFFF
};


////////////////////////////////////////////////////////////
// DataBlob records

/// Self-describing record-format descriptor. Each record, including this one, publishes both its fixed-field size and
/// minor version so a data model builder can select a compatible decoder.
struct NvTrace_GPU_ClockCalibration_TscGpu_RecordVersionInfo
{
    uint32_t recordVersionInfoSize;
    uint32_t recordVersionInfoMinorVersion;

    uint32_t calibrationPointSize;
    uint32_t calibrationPointMinorVersion;

    uint32_t gpuDeviceInfoSize;
    uint32_t gpuDeviceInfoMinorVersion;
};
#define NvTrace_GPU_ClockCalibration_TscGpu_RecordVersionInfo_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_GPU_ClockCalibration_TscGpu_RecordVersionInfo, gpuDeviceInfoMinorVersion)


////////////////////////////////////////////////////////////
// Circular-buffer records

// Each CalibrationPoint represents a pair of timestamps from the different clock domains, and the conversion factors between them.
// Conversion factors are calculated between the previous and current CalibrationPoints.
struct NvTrace_GPU_ClockCalibration_TscGpu_CalibrationPoint
{
    // An unsigned 64-bit nanosecond value can count up to the year 2554 = 1970 + 2**64 / (60 * 60 * 24 * 365) / 1e9
    uint64_t tscTick;
    uint64_t gpuNsec;
    double tscToGpuFactor; /// caveat: only use this on durations < 0.28 years
    double gpuToTscFactor; /// caveat: only use this on durations < 0.28 years
};
#define NvTrace_GPU_ClockCalibration_TscGpu_CalibrationPoint_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_GPU_ClockCalibration_TscGpu_CalibrationPoint, gpuToTscFactor)


#ifdef __cplusplus
} // extern "C"
#endif
NVTRACE_RESTORE_WARNINGS()
