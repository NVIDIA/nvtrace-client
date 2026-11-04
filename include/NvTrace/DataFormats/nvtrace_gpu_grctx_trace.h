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


NVTRACE_WEAK_CONST NVTRACE_UUID NvTrace_GPU_GrCtxTrace_ProviderTypeUuid = NVTRACE_UUID_VALUE(0x128395fd, 0xd82a, 0x47ee, 0xab, 0x6c, 0x05, 0x98, 0x69, 0x83, 0x1f, 0xfc);


////////////////////////////////////////////////////////////
// Provider-buffer sections

/// Provider-specific section tags used with NvTrace_ProviderBufferSectionDesc::sectionTag.
enum NvTrace_GPU_GrCtxTrace_SectionTag
{
    NvTrace_GPU_GrCtxTrace_SectionTag_Invalid = 0,

    /// DataBlob section containing NvTrace_GPU_GrCtxTrace_RecordVersionInfo.
    NvTrace_GPU_GrCtxTrace_SectionTag_RecordVersionInfo = 1,

    /// DataBlob section containing a serialized FlatBuffer (nv.trace.FB.Provider.GpuGrCtxTrace.ProviderConfiguration).
    NvTrace_GPU_GrCtxTrace_SectionTag_ProviderConfiguration = 2,

    /// Gates on the payload's timestampGpuNsec (recordTimestampType == GpuPtimer).
    NvTrace_GPU_GrCtxTrace_SectionTag_GrCtxTraceEventCircularBuffer = 3,

    /// Gates on the payload's failureTimestampTsc (recordTimestampType == Tsc).
    NvTrace_GPU_GrCtxTrace_SectionTag_ProviderDiagnosticEventCircularBuffer = 4,

    /// DataBlob section containing NvTrace_GPU_DeviceInfo.
    NvTrace_GPU_GrCtxTrace_SectionTag_GpuDeviceInfo = 5,

    NvTrace_GPU_GrCtxTrace_SectionTag_FORCE_INT = 0x7FFFFFFF
};

/// Provider-specific record types. Each data-blob section carries a single record of the corresponding type; the
/// GR context trace and provider-diagnostic circular buffers carry a record type per cell.
/// Append only -- record types are wire-persisted and never renumbered, so a new record type takes
/// the value after the last.
enum NvTrace_GPU_GrCtxTrace_RecordType
{
    NvTrace_GPU_GrCtxTrace_RecordType_Invalid = 0,

    /// One NvTrace_GPU_GrCtxTrace_RecordVersionInfo payload (RecordVersionInfo DataBlob section).
    NvTrace_GPU_GrCtxTrace_RecordType_RecordVersionInfo = 1,

    /// One serialized ProviderConfiguration FlatBuffer (ProviderConfiguration DataBlob section).
    NvTrace_GPU_GrCtxTrace_RecordType_ProviderConfiguration = 2,

    /// NvTrace_GPU_GrCtxTrace_ProviderDiagnostic_InitializationFailure payload (ProviderDiagnostic circular buffer).
    NvTrace_GPU_GrCtxTrace_RecordType_ProviderDiagnostic_InitializationFailure = 3,

    /// NvTrace_GPU_GrCtxTrace_ContextChange payload: a context load / save completed.
    NvTrace_GPU_GrCtxTrace_RecordType_ContextLoadEnd = 4,
    NvTrace_GPU_GrCtxTrace_RecordType_ContextSaveEnd = 5,

    /// One NvTrace_GPU_DeviceInfo payload (GpuDeviceInfo DataBlob section).
    NvTrace_GPU_GrCtxTrace_RecordType_GpuDeviceInfo = 6,

    NvTrace_GPU_GrCtxTrace_RecordType_Count,
    NvTrace_GPU_GrCtxTrace_RecordType_FORCE_INT = 0x7FFFFFFF
};


////////////////////////////////////////////////////////////
// DataBlob records

/// Self-describing record-format descriptor. Each record, including this one, publishes both its fixed-field size and
/// minor version so a data model builder can select a compatible decoder.
struct NvTrace_GPU_GrCtxTrace_RecordVersionInfo
{
    uint32_t recordVersionInfoSize;
    uint32_t recordVersionInfoMinorVersion;

    uint32_t providerDiagnosticInitializationFailureSize;
    uint32_t providerDiagnosticInitializationFailureMinorVersion;

    uint32_t contextChangeSize;
    uint32_t contextChangeMinorVersion;

    uint32_t gpuDeviceInfoSize;
    uint32_t gpuDeviceInfoMinorVersion;
};
#define NvTrace_GPU_GrCtxTrace_RecordVersionInfo_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_GPU_GrCtxTrace_RecordVersionInfo, gpuDeviceInfoMinorVersion)


////////////////////////////////////////////////////////////
// Circular-buffer records

/// Provider diagnostic
enum NvTrace_GPU_GrCtxTrace_ProviderFailureReason
{
    NvTrace_GPU_GrCtxTrace_ProviderFailureReason_None = 0,
    NvTrace_GPU_GrCtxTrace_ProviderFailureReason_InitializationFailed,
    NvTrace_GPU_GrCtxTrace_ProviderFailureReason_InsufficientDriverVersion,
    NvTrace_GPU_GrCtxTrace_ProviderFailureReason_ConfigurationInvalid,
    NvTrace_GPU_GrCtxTrace_ProviderFailureReason_RmDeviceLookupFailed,
    NvTrace_GPU_GrCtxTrace_ProviderFailureReason_RmEventBufferInitializationFailed,

    NvTrace_GPU_GrCtxTrace_ProviderFailureReason_FORCE_INT = 0x7FFFFFFF
};

struct NvTrace_GPU_GrCtxTrace_ProviderDiagnostic_InitializationFailure
{
    uint64_t failureTimestampTsc;
    enum NvTrace_GPU_GrCtxTrace_ProviderFailureReason failureReason;
};
#define NvTrace_GPU_GrCtxTrace_ProviderDiagnostic_InitializationFailure_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_GPU_GrCtxTrace_ProviderDiagnostic_InitializationFailure, failureReason)

/// GrCtx trace event
struct NvTrace_GPU_GrCtxTrace_ContextChange
{
    // PTIMER value
    uint64_t timestampGpuNsec;
    // virtual machine id
    uint8_t vmid;
    // padding
    uint8_t rsvd0009;
    // sequence number
    uint16_t rmSequenceNumber;
    // unique value allocated by RM
    uint32_t rmCid;
    // process id
    uint64_t pid;
};
#define NvTrace_GPU_GrCtxTrace_ContextChange_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_GPU_GrCtxTrace_ContextChange, pid)


#ifdef __cplusplus
} // extern "C"
#endif
NVTRACE_RESTORE_WARNINGS()
