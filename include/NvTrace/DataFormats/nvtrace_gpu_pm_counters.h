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


NVTRACE_WEAK_CONST NVTRACE_UUID NvTrace_GPU_PmCounters_ProviderTypeUuid = NVTRACE_UUID_VALUE(0x8e671dca, 0x8d25, 0x436a, 0x9e, 0x10, 0x28, 0xee, 0x94, 0x15, 0x6f, 0x37);


////////////////////////////////////////////////////////////
// Provider-buffer sections

/// Provider-specific section tags used with NvTrace_ProviderBufferSectionDesc::sectionTag.
enum NvTrace_GPU_PmCounters_SectionTag
{
    NvTrace_GPU_PmCounters_SectionTag_Invalid = 0,

    /// DataBlob section containing NvTrace_GPU_PmCounters_RecordVersionInfo.
    NvTrace_GPU_PmCounters_SectionTag_RecordVersionInfo = 1,

    /// DataBlob section containing a serialized FlatBuffer (nv.trace.FB.Provider.GpuPmCounters.ProviderConfiguration).
    NvTrace_GPU_PmCounters_SectionTag_ProviderConfiguration = 2,

    /// DataBlob section containing NvTrace_GPU_DeviceInfo.
    NvTrace_GPU_PmCounters_SectionTag_GpuDeviceInfo = 3,

    /// Gates on the payload's first timestamp field (recordTimestampType == Tsc).
    NvTrace_GPU_PmCounters_SectionTag_ProviderDiagnosticEventCircularBuffer = 4,

    /// DataBlob section containing NvTrace_GPU_PmCounters_CounterDataInfo followed by the counter-data preamble.
    NvTrace_GPU_PmCounters_SectionTag_CounterDataInfo = 5,

    /// Gates on the payload's endTimestampPtimer (recordTimestampType == GpuPtimer).
    NvTrace_GPU_PmCounters_SectionTag_CounterSampleCircularBuffer = 6,

    /// Gates on the payload's pauseTimestampTsc / resumeTimestampTsc (recordTimestampType == Tsc).
    NvTrace_GPU_PmCounters_SectionTag_LifecycleEventCircularBuffer = 7,

    NvTrace_GPU_PmCounters_SectionTag_FORCE_INT = 0x7FFFFFFF
};

/// Provider-specific record types. Each data-blob section carries a single record of the corresponding type; the
/// counter-sample circular buffer publishes one fixed record type for every cell, and the provider-diagnostic and
/// lifecycle circular buffers carry a record type per cell.
/// Append only -- record types are wire-persisted and never renumbered, so a new record type takes
/// the value after the last.
enum NvTrace_GPU_PmCounters_RecordType
{
    NvTrace_GPU_PmCounters_RecordType_Invalid = 0,

    /// One NvTrace_GPU_PmCounters_RecordVersionInfo payload (RecordVersionInfo DataBlob section).
    NvTrace_GPU_PmCounters_RecordType_RecordVersionInfo = 1,

    /// One serialized ProviderConfiguration FlatBuffer (ProviderConfiguration DataBlob section).
    NvTrace_GPU_PmCounters_RecordType_ProviderConfiguration = 2,

    /// One NvTrace_GPU_DeviceInfo payload (GpuDeviceInfo DataBlob section).
    NvTrace_GPU_PmCounters_RecordType_GpuDeviceInfo = 3,

    /// NvTrace_GPU_PmCounters_ProviderDiagnostic_InitializationFailure payload (ProviderDiagnostic circular buffer).
    NvTrace_GPU_PmCounters_RecordType_ProviderDiagnostic_InitializationFailure = 4,

    /// One NvTrace_GPU_PmCounters_CounterDataInfo payload (CounterDataInfo DataBlob section).
    NvTrace_GPU_PmCounters_RecordType_CounterDataInfo = 5,

    /// NvTrace_GPU_PmCounters_CounterSample payload: one PM counter sample (counter-sample circular buffer).
    NvTrace_GPU_PmCounters_RecordType_CounterSample = 6,

    /// NvTrace_GPU_Provider_LifecycleEvent_Pause / _Resume payload (lifecycle circular buffer).
    NvTrace_GPU_PmCounters_RecordType_LifecyclePause = 7,
    NvTrace_GPU_PmCounters_RecordType_LifecycleResume = 8,

    /// NvTrace_GPU_PmCounters_ProviderDiagnostic_UpdateTimeExceeded payload (ProviderDiagnostic circular buffer).
    NvTrace_GPU_PmCounters_RecordType_ProviderDiagnostic_UpdateTimeExceeded = 9,

    /// NvTrace_GPU_PmCounters_ProviderDiagnostic_SampleDiscontinuity payload (ProviderDiagnostic circular buffer).
    NvTrace_GPU_PmCounters_RecordType_ProviderDiagnostic_SampleDiscontinuity = 10,

    NvTrace_GPU_PmCounters_RecordType_Count,
    NvTrace_GPU_PmCounters_RecordType_FORCE_INT = 0x7FFFFFFF
};


////////////////////////////////////////////////////////////
// DataBlob records

/// Self-describing record-format descriptor. Each record, including this one, publishes both its fixed-field size and
/// minor version so a data model builder can select a compatible decoder.
struct NvTrace_GPU_PmCounters_RecordVersionInfo
{
    uint32_t recordVersionInfoSize;
    uint32_t recordVersionInfoMinorVersion;

    uint32_t gpuDeviceInfoSize;
    uint32_t gpuDeviceInfoMinorVersion;

    uint32_t providerDiagnosticInitializationFailureSize;
    uint32_t providerDiagnosticInitializationFailureMinorVersion;

    uint32_t counterDataInfoSize; // the size of the fixed fields
    uint32_t counterDataInfoMinorVersion;

    uint32_t counterSampleSize; // the size of the fixed fields
    uint32_t counterSampleMinorVersion;

    uint32_t lifecyclePauseSize;
    uint32_t lifecyclePauseMinorVersion;

    uint32_t lifecycleResumeSize;
    uint32_t lifecycleResumeMinorVersion;

    uint32_t providerDiagnosticUpdateTimeExceededSize;
    uint32_t providerDiagnosticUpdateTimeExceededMinorVersion;

    uint32_t providerDiagnosticSampleDiscontinuitySize;
    uint32_t providerDiagnosticSampleDiscontinuityMinorVersion;
};
#define NvTrace_GPU_PmCounters_RecordVersionInfo_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_GPU_PmCounters_RecordVersionInfo, lifecycleResumeMinorVersion)
#define NvTrace_GPU_PmCounters_RecordVersionInfo_MINOR_VERSION_B NVTRACE_FIELD_END_OFFSET(struct NvTrace_GPU_PmCounters_RecordVersionInfo, providerDiagnosticSampleDiscontinuityMinorVersion)

/// Leading segment of the CounterDataInfo record; the Nsight Perf SDK CounterDataImage preamble follows it as a
/// dynamic segment of counterDataPreambleSize bytes, beginning at the 8-aligned end of this struct. sampleRangeDataSize
/// and sampleEvaluatedMetricCount size the dynamic segments of every NvTrace_GPU_PmCounters_CounterSample record, so this
/// record must be decoded before any of them.
///
///   +-------------------------------+
///   | CounterDataInfo fixed fields  |
///   +-------------------------------+
///   | 8-byte alignment padding      |
///   +-------------------------------+
///   | CounterData preamble          |
///   | counterDataPreambleSize bytes |
///   +-------------------------------+
///
struct NvTrace_GPU_PmCounters_CounterDataInfo
{
    uint32_t counterDataPreambleSize;
    uint32_t sampleRangeDataSize;
    uint32_t sampleEvaluatedMetricCount; // in units of doubles (not bytes)
};
#define NvTrace_GPU_PmCounters_CounterDataInfo_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_GPU_PmCounters_CounterDataInfo, sampleEvaluatedMetricCount)


////////////////////////////////////////////////////////////
// Circular-buffer records

/// Provider diagnostic
enum NvTrace_GPU_PmCounters_ProviderFailureReason
{
    NvTrace_GPU_PmCounters_ProviderFailureReason_None = 0,
    NvTrace_GPU_PmCounters_ProviderFailureReason_InitializationFailed,
    NvTrace_GPU_PmCounters_ProviderFailureReason_Disabled,
    NvTrace_GPU_PmCounters_ProviderFailureReason_NoMetricsConfigured,
    NvTrace_GPU_PmCounters_ProviderFailureReason_PerfWorksInitializationFailed,

    NvTrace_GPU_PmCounters_ProviderFailureReason_FORCE_INT = 0x7FFFFFFF
};

struct NvTrace_GPU_PmCounters_ProviderDiagnostic_InitializationFailure
{
    uint64_t failureTimestampTsc;
    enum NvTrace_GPU_PmCounters_ProviderFailureReason failureReason;
};
#define NvTrace_GPU_PmCounters_ProviderDiagnostic_InitializationFailure_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_GPU_PmCounters_ProviderDiagnostic_InitializationFailure, failureReason)

/// Whole provider-update duration. The first timestamp is the end of the measured update,
/// captured before publishing this diagnostic record.
struct NvTrace_GPU_PmCounters_ProviderDiagnostic_UpdateTimeExceeded
{
    uint64_t updateEndTimestampTsc;
    uint64_t updateStartTimestampTsc;
    uint64_t updateThreadCpuTimeNsec; // Calling thread's CPU-time delta; zero means unmeasured or unavailable.
};
#define NvTrace_GPU_PmCounters_ProviderDiagnostic_UpdateTimeExceeded_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_GPU_PmCounters_ProviderDiagnostic_UpdateTimeExceeded, updateThreadCpuTimeNsec)

enum NvTrace_GPU_PmCounters_SampleDiscontinuityReason
{
    NvTrace_GPU_PmCounters_SampleDiscontinuityReason_Unknown = 0,
    NvTrace_GPU_PmCounters_SampleDiscontinuityReason_HwBufferOverrun = 1,

    NvTrace_GPU_PmCounters_SampleDiscontinuityReason_FORCE_INT = 0x7FFFFFFF
};

/// Observed gap between consecutive sample endpoints. All timestamps use the CPU TSC clock domain.
/// A timestamp gap alone does not establish a hardware-buffer overrun.
struct NvTrace_GPU_PmCounters_ProviderDiagnostic_SampleDiscontinuity
{
    uint64_t detectionTimestampTsc; // CPU detection time; gates diagnostic-buffer draining.
    uint64_t startTimestampTsc;     // Previous sample's end; beginning of the observed gap.
    uint64_t endTimestampTsc;       // Current sample's end; end of the observed gap.
    enum NvTrace_GPU_PmCounters_SampleDiscontinuityReason reason;
};
#define NvTrace_GPU_PmCounters_ProviderDiagnostic_SampleDiscontinuity_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_GPU_PmCounters_ProviderDiagnostic_SampleDiscontinuity, reason)

/// Counter sample
/// Leading segment of a CounterSample record. Two dynamic segments follow, each beginning at the 8-aligned end of the
/// one before it: the sample's CounterDataImage RangeData (sampleRangeDataSize bytes), then its evaluated metric
/// values (sampleEvaluatedMetricCount doubles). Both are sized by the CounterDataInfo record.
///
///   +------------------------------------+
///   | CounterSample fixed fields         |
///   +------------------------------------+
///   | 8-byte alignment padding           |
///   +------------------------------------+
///   | RangeData                          |
///   | sampleRangeDataSize bytes          |
///   +------------------------------------+
///   | 8-byte alignment padding           |
///   +------------------------------------+
///   | Evaluated metric values            |
///   | sampleEvaluatedMetricCount doubles |
///   +------------------------------------+
///
struct NvTrace_GPU_PmCounters_CounterSample
{
    uint64_t endTimestampPtimer;
    uint64_t endTimestampTsc;
};
#define NvTrace_GPU_PmCounters_CounterSample_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_GPU_PmCounters_CounterSample, endTimestampTsc)


#ifdef __cplusplus
} // extern "C"
#endif
NVTRACE_RESTORE_WARNINGS()
