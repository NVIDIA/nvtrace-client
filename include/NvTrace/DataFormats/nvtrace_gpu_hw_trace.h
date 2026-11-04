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


NVTRACE_WEAK_CONST NVTRACE_UUID NvTrace_GPU_HwTrace_ProviderTypeUuid = NVTRACE_UUID_VALUE(0x77da5524, 0x743e, 0x40c9, 0x94, 0xca, 0x84, 0x28, 0xdc, 0xa4, 0xd, 0x1b);


/// Provider-specific section tags used with NvTrace_ProviderBufferSectionDesc::sectionTag.
enum NvTrace_GPU_HwTrace_SectionTag
{
    NvTrace_GPU_HwTrace_SectionTag_Invalid = 0,

    /// DataBlob section containing NvTrace_GPU_HwTrace_RecordVersionInfo.
    NvTrace_GPU_HwTrace_SectionTag_RecordVersionInfo = 1,

    /// DataBlob section containing a serialized FlatBuffer (nv.trace.FB.Provider.GpuHwTrace.ProviderConfiguration).
    NvTrace_GPU_HwTrace_SectionTag_ProviderConfiguration = 2,

    /// Gates on the payload's failureTimestampTsc (recordTimestampType == Tsc).
    NvTrace_GPU_HwTrace_SectionTag_ProviderDiagnosticEventCircularBuffer = 3,

    /// Gates on the payload's ptimer (recordTimestampType == GpuPtimer).
    NvTrace_GPU_HwTrace_SectionTag_HwTraceEventCircularBuffer = 4,

    /// Gates on the payload's pauseTimestampTsc / resumeTimestampTsc (recordTimestampType == Tsc).
    NvTrace_GPU_HwTrace_SectionTag_LifecycleEventCircularBuffer = 5,

    /// DataBlob section containing NvTrace_GPU_DeviceInfo.
    NvTrace_GPU_HwTrace_SectionTag_GpuDeviceInfo = 6,

    NvTrace_GPU_HwTrace_SectionTag_FORCE_INT = 0x7FFFFFFF
};

/// Provider-specific record types. The circular-buffer sections are variable-record-type: each cell's recordType
/// is one of these values, read natively by the consumer to select the payload struct that follows it. Each
/// data-blob section carries a single record of the corresponding type.
/// Append only -- record types are wire-persisted and never renumbered, so a new record type takes
/// the value after the last.
enum NvTrace_GPU_HwTrace_RecordType
{
    NvTrace_GPU_HwTrace_RecordType_Invalid = 0,

    /// One NvTrace_GPU_HwTrace_RecordVersionInfo payload (RecordVersionInfo DataBlob section).
    NvTrace_GPU_HwTrace_RecordType_RecordVersionInfo = 1,

    /// One serialized ProviderConfiguration FlatBuffer (ProviderConfiguration DataBlob section).
    NvTrace_GPU_HwTrace_RecordType_ProviderConfiguration = 2,

    /// NvTrace_GPU_HwTrace_ProviderDiagnostic_InitializationFailure payload (ProviderDiagnostic circular buffer).
    NvTrace_GPU_HwTrace_RecordType_ProviderDiagnostic_InitializationFailure = 3,

    /// NvTrace_GPU_HwTrace_ComputeGrid payload: a compute grid (QMD) launch began / completed.
    NvTrace_GPU_HwTrace_RecordType_ComputeGridStart = 4,
    NvTrace_GPU_HwTrace_RecordType_ComputeGridEnd = 5,

    /// NvTrace_GPU_HwTrace_ContextChange payload: a context was loaded / saved.
    NvTrace_GPU_HwTrace_RecordType_ContextChangeLoad = 6,
    NvTrace_GPU_HwTrace_RecordType_ContextChangeSave = 7,

    /// NvTrace_GPU_Provider_LifecycleEvent_Pause / _Resume payload (lifecycle circular buffer).
    NvTrace_GPU_HwTrace_RecordType_LifecyclePause = 8,
    NvTrace_GPU_HwTrace_RecordType_LifecycleResume = 9,

    /// NvTrace_GPU_HwTrace_ComputeGrid payload: a GridNullEnd completed.
    NvTrace_GPU_HwTrace_RecordType_ComputeGridNullEnd = 10,

    /// One NvTrace_GPU_DeviceInfo payload (GpuDeviceInfo DataBlob section).
    NvTrace_GPU_HwTrace_RecordType_GpuDeviceInfo = 11,

    NvTrace_GPU_HwTrace_RecordType_Count,
    NvTrace_GPU_HwTrace_RecordType_FORCE_INT = 0x7FFFFFFF
};

/// Self-describing record-format descriptor. Each record, including this one, publishes both its fixed-field size and
/// minor version so a data model builder can select a compatible decoder.
struct NvTrace_GPU_HwTrace_RecordVersionInfo
{
    uint32_t recordVersionInfoSize;
    uint32_t recordVersionInfoMinorVersion;

    uint32_t providerDiagnosticInitializationFailureSize;
    uint32_t providerDiagnosticInitializationFailureMinorVersion;

    uint32_t computeGridSize;
    uint32_t computeGridMinorVersion;

    uint32_t contextChangeSize;
    uint32_t contextChangeMinorVersion;

    uint32_t lifecyclePauseSize;
    uint32_t lifecyclePauseMinorVersion;

    uint32_t lifecycleResumeSize;
    uint32_t lifecycleResumeMinorVersion;

    uint32_t gpuDeviceInfoSize;
    uint32_t gpuDeviceInfoMinorVersion;
};
#define NvTrace_GPU_HwTrace_RecordVersionInfo_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_GPU_HwTrace_RecordVersionInfo, gpuDeviceInfoMinorVersion)


enum NvTrace_GPU_HwTrace_ComputeLaunchQueue
{
    NvTrace_GPU_HwTrace_ComputeLaunchQueue_Zero = 0,
    NvTrace_GPU_HwTrace_ComputeLaunchQueue_One = 1,
};

struct NvTrace_GPU_HwTrace_ComputeGrid
{
    uint64_t ptimer;
    uint32_t correlationId;
    uint8_t clq; // NvTrace_GPU_HwTrace_ComputeLaunchQueue
};
#define NvTrace_GPU_HwTrace_ComputeGrid_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_GPU_HwTrace_ComputeGrid, clq)

struct NvTrace_GPU_HwTrace_ContextChange
{
    uint64_t ptimer;
    uint32_t contextId;
};
#define NvTrace_GPU_HwTrace_ContextChange_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_GPU_HwTrace_ContextChange, contextId)

enum NvTrace_GPU_HwTrace_ProviderFailureReason
{
    NvTrace_GPU_HwTrace_ProviderFailureReason_None = 0,
    NvTrace_GPU_HwTrace_ProviderFailureReason_InitializationFailed,
    NvTrace_GPU_HwTrace_ProviderFailureReason_Disabled,
    NvTrace_GPU_HwTrace_ProviderFailureReason_UnsupportedDevice,
    NvTrace_GPU_HwTrace_ProviderFailureReason_ConfigurationInvalid,

    NvTrace_GPU_HwTrace_ProviderFailureReason_FORCE_INT = 0x7FFFFFFF
};

struct NvTrace_GPU_HwTrace_ProviderDiagnostic_InitializationFailure
{
    uint64_t failureTimestampTsc;
    NvTrace_GPU_HwTrace_ProviderFailureReason failureReason;
};
#define NvTrace_GPU_HwTrace_ProviderDiagnostic_InitializationFailure_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_GPU_HwTrace_ProviderDiagnostic_InitializationFailure, failureReason)

#ifdef __cplusplus
} // extern "C"
#endif
NVTRACE_RESTORE_WARNINGS()
