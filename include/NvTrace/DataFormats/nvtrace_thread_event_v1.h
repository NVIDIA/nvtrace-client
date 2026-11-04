/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/DataFormats/nvtrace_common_types.h>
#include <NvTrace/DataFormats/nvtrace_compiler_compatibility.h>
#include <NvTrace/DataFormats/nvtrace_provider_buffer_v1.h>


NVTRACE_SUPPRESS_WARNINGS()
#ifdef __cplusplus
extern "C" {
#endif


NVTRACE_WEAK_CONST NVTRACE_UUID NvTrace_ThreadEvent_ProviderTypeUuid_V1 = NVTRACE_UUID_VALUE(0xf23e0298, 0x3a40, 0x42fa, 0xab, 0x2f, 0xee, 0xc8, 0xb8, 0xb5, 0x2a, 0xfc);

/// Buffer layout structure for the Thread Event provider.
/// Note: this provider will allocate one giant buffer for all sub-buffers. Thus, offsets are sufficient to discover sub-buffers.
struct NvTrace_ThreadEvent_ProviderBufferLayout_V1
{
    /// This is a COFF-style scheme, where each offset is relative to the pointer of the provider buffer header struct.

    /// points at NvTrace_ThreadEvent_EventVersionInfo_V1
    uint64_t eventVersionInfoOffset;
    uint32_t eventVersionInfoSize;
    uint32_t eventVersionInfoMinorVersion;

    /// points at a flatbuffer of type nv::trace::FB::Provider::ThreadEvent::ProviderConfiguration
    uint64_t configurationOffset;
    uint32_t configurationSize;

    /// points at an array of NvTrace_ThreadEvent_ProviderIdMapping_V1
    uint32_t providerIdMappingsStaticCount; // Determines where the dynamic section of the array starts
    uint64_t providerIdMappingsOffset;
    uint32_t providerIdMappingsCount;

    /// points at an array of NvTrace_ThreadEvent_ThreadState_V1
    uint64_t threadStatesOffset;
    uint32_t threadStatesCount;

    /// points at an array of NvTrace_AdaptiveBufferState_V1
    uint64_t eventBufferStatesOffset;
    uint32_t eventBufferStatesCount;
    uint32_t eventBufferStateSize;
    uint32_t eventBufferStateMinorVersion;

    /// points at an array of NvTrace_AdaptiveBufferCell_V1, holding eventBufferStatesCount
    /// consecutive blocks of eventBufferCellsCount cells each -- one block per event buffer
    uint64_t eventBufferCellsOffset;
    uint32_t eventBufferCellsCount;
    uint32_t eventBufferCellSize;
    uint32_t eventBufferCellMinorVersion;

    /// points at a cacheline-aligned array of producer-private records, opaque to consumers
    uint64_t nvTraceTlsOffset;
    uint32_t nvTraceTlsCount;
    uint32_t nvTraceTlsSize;

    /// points at an array of chunks; placed last in the buffer
    uint64_t chunksOffset;
    uint32_t chunkCount;
    uint32_t chunkSize;
};
#define NvTrace_ThreadEvent_ProviderBufferLayout_V1_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_ThreadEvent_ProviderBufferLayout_V1, chunkSize)

struct NvTrace_ThreadEvent_EventVersionInfo_V1
{
    uint32_t threadStateSize;
    uint32_t threadStateMinorVersion;

    uint32_t providerIdMappingSize;
    uint32_t providerIdMappingMinorVersion;

    uint32_t threadLifecycleEventSize;
    uint32_t threadLifecycleEventMinorVersion;

    uint32_t eventBufferChunkHeaderSize;

    uint32_t eventBufferRecordSize;
    uint32_t eventBufferRecordMinorVersion;
};
#define NvTrace_ThreadEvent_EventVersionInfo_V1_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_ThreadEvent_EventVersionInfo_V1, eventBufferRecordMinorVersion)

enum NvTrace_ThreadEvent_EventType_V1
{
    NvTrace_ThreadEvent_EventType_V1_Invalid = 0,
    NvTrace_ThreadEvent_EventType_V1_ThreadCreated,
    NvTrace_ThreadEvent_EventType_V1_ThreadDestroyed,
    NvTrace_ThreadEvent_EventType_V1_COUNT,
    NvTrace_ThreadEvent_EventType_V1_FORCE_INT = 0x7fffffff
};

struct NvTrace_ThreadEvent_ThreadLifecycle_Event_V1
{
    uint64_t timestampTsc;
    uint64_t threadStateIndex;
    uint64_t threadStateRevision;
    // Additional fields go here as needed.
};
#define NvTrace_ThreadEvent_ThreadLifecycle_Event_V1_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_ThreadEvent_ThreadLifecycle_Event_V1, threadStateRevision)

struct NvTrace_ThreadEvent_ProviderIdMapping_V1
{
    /// The data revision; used for consistency checks.
    volatile uint64_t revision;

    NVTRACE_UUID providerTypeUuid;
    NVTRACE_LUID mappedProviderLuid;
};
#define NvTrace_ThreadEvent_ProviderIdMapping_V1_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_ThreadEvent_ProviderIdMapping_V1, mappedProviderLuid)

// A V1 static provider id is its index in the provider-id mapping table; frozen with the V1 format.
enum NvTrace_ThreadEvent_StaticProviderId_V1
{
    NvTrace_ThreadEvent_StaticProviderId_V1_INVALID = 0,

    NvTrace_ThreadEvent_StaticProviderId_V1_ThreadEvent,
    NvTrace_ThreadEvent_StaticProviderId_V1_CUDA,
    NvTrace_ThreadEvent_StaticProviderId_V1_NVTX,

    NvTrace_ThreadEvent_StaticProviderId_V1_COUNT,
    NvTrace_ThreadEvent_StaticProviderId_V1_FORCE_INT = 0x7fffffff
};

// The V1 producer claimed dynamic provider ids descending from here; frozen with the V1 format.
NVTRACE_WEAK_CONST uint32_t NvTrace_ThreadEventV1_DynamicProviderIdLast = 0xFFF;

struct NvTrace_ThreadEvent_ThreadState_V1
{
    volatile uint64_t destructionTimestampTsc; // 0 if not deleted; not guarded by revision

    /// The data revision; used for consistency checks.
    volatile uint64_t revision;

    uint64_t creationTimestampTsc;

    /// A unique ID defined by NvTrace.  Has no relation to the nativeThreadId.
    /// If uniqueThreadId is zero, this ThreadState is being used by multiple threads.
    uint64_t uniqueThreadId;
    uint32_t nativeThreadId;
};
#define NvTrace_ThreadEvent_ThreadState_V1_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_ThreadEvent_ThreadState_V1, nativeThreadId)

struct NvTrace_ThreadEvent_EventBufferChunkHeader_V1
{
    struct NvTrace_AdaptiveBufferChunkHeader_V1 adaptiveBufferChunkHeader;
    uint64_t uniqueThreadId;
};
#define NvTrace_ThreadEvent_EventBufferChunkHeader_V1_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_ThreadEvent_EventBufferChunkHeader_V1, uniqueThreadId)

struct NvTrace_ThreadEvent_EventBufferRecord_V1
{
    uint32_t payloadSize;
    // 31-20: Provider ID, 19-0: Event Type
    uint32_t recordIdentifier;
};
#define NvTrace_ThreadEvent_EventBufferRecord_V1_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_ThreadEvent_EventBufferRecord_V1, recordIdentifier)


#ifdef __cplusplus
} // extern "C"
#endif
NVTRACE_RESTORE_WARNINGS()
