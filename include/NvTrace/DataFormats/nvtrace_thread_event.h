/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/DataFormats/nvtrace_common_types.h>
#include <NvTrace/DataFormats/nvtrace_compiler_compatibility.h>
#include <NvTrace/DataFormats/nvtrace_provider_buffer.h>


NVTRACE_SUPPRESS_WARNINGS()
#ifdef __cplusplus
extern "C" {
#endif


NVTRACE_WEAK_CONST NVTRACE_UUID NvTrace_ThreadEvent_ProviderTypeUuid = NVTRACE_UUID_VALUE(0x668da17e, 0xd9c2, 0x46a4, 0x86, 0xa0, 0x08, 0xb1, 0xf4, 0xd5, 0x08, 0x38);

enum NvTrace_ThreadEvent_SectionTag
{
    NvTrace_ThreadEvent_SectionTag_INVALID = 0,

    NvTrace_ThreadEvent_SectionTag_RecordVersionInfo = 1,
    NvTrace_ThreadEvent_SectionTag_Configuration = 2,
    NvTrace_ThreadEvent_SectionTag_ProviderProperties = 3,
    NvTrace_ThreadEvent_SectionTag_ProviderStatistics = 4,

    // Circular buffer of NvTrace_ThreadEvent_SourceProviderRegistered records, sized to never wrap
    NvTrace_ThreadEvent_SectionTag_SourceProviderRegistrations = 5,

    NvTrace_ThreadEvent_SectionTag_ThreadStates = 6,

    NvTrace_ThreadEvent_SectionTag_EventChunkPool = 7,
    NvTrace_ThreadEvent_SectionTag_EventBuffer = 8,

    NvTrace_ThreadEvent_SectionTag_FORCE_INT = 0x7fffffff
};

// The 20-bit within-provider record types; the upper 12 bits of a full record type carry the index of
// the source provider that emitted it (NVTRACE_THREAD_EVENT_RECORD_IDENTIFIER_SOURCE_INDEX_FIELD).
/// Append only -- record types are wire-persisted and never renumbered, so a new record type takes
/// the value after the last.
enum NvTrace_ThreadEvent_RecordType
{
    NvTrace_ThreadEvent_RecordType_INVALID = 0,

    NvTrace_ThreadEvent_RecordType_RecordVersionInfo = 1,
    NvTrace_ThreadEvent_RecordType_Configuration = 2,
    NvTrace_ThreadEvent_RecordType_ProviderProperties = 3,
    NvTrace_ThreadEvent_RecordType_ProviderStatistics = 4,

    NvTrace_ThreadEvent_RecordType_SourceProviderRegistered = 5,

    // Opens every cell, and so re-establishes attribution for a consumer that joins at any cell
    NvTrace_ThreadEvent_RecordType_ThreadChanged = 6,

    NvTrace_ThreadEvent_RecordType_ThreadCreated = 7,
    NvTrace_ThreadEvent_RecordType_ThreadDestroyed = 8,

    NvTrace_ThreadEvent_RecordType_COUNT,
    NvTrace_ThreadEvent_RecordType_FORCE_INT = 0x7fffffff
};

struct NvTrace_ThreadEvent_RecordVersionInfo
{
    uint32_t recordVersionInfoSize;
    uint32_t recordVersionInfoMinorVersion;

    uint32_t providerPropertiesSize;
    uint32_t providerPropertiesMinorVersion;

    uint32_t providerStatisticsSize;
    uint32_t providerStatisticsMinorVersion;

    uint32_t sourceProviderRegisteredSize;
    uint32_t sourceProviderRegisteredMinorVersion;

    uint32_t threadChangedSize;
    uint32_t threadChangedMinorVersion;

    uint32_t threadCreatedSize;
    uint32_t threadCreatedMinorVersion;

    uint32_t threadDestroyedSize;
    uint32_t threadDestroyedMinorVersion;
};
#define NvTrace_ThreadEvent_RecordVersionInfo_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_ThreadEvent_RecordVersionInfo, threadDestroyedMinorVersion)

// The producer's properties, fixed at initialization
struct NvTrace_ThreadEvent_ProviderProperties
{
    // The last static index the producer knows; a consumer may interpret indices at or below it
    // against NvTrace_ThreadEvent_StaticSourceProviderIndex
    uint32_t maxStaticSourceProviderIndex;

    // Static and dynamic together: every registered index is below it
    uint32_t maxSourceProviderCount;

    // How long an event buffer writes into one cell, so a buffer holds about its cell count times this
    uint64_t cellWriteDurationNsec;
};
#define NvTrace_ThreadEvent_ProviderProperties_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_ThreadEvent_ProviderProperties, cellWriteDurationNsec)

// Independently meaningful 64-bit fields, sampled best-effort: there is no consistency protocol.
struct NvTrace_ThreadEvent_ProviderStatistics
{
    uint64_t totalThreadCount;
    uint64_t registeredSourceProviderCount;
    uint64_t droppedEventCount;
};
#define NvTrace_ThreadEvent_ProviderStatistics_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_ThreadEvent_ProviderStatistics, droppedEventCount)

// A source provider is a provider whose records ride this transport. The producer reserves an index for
// each well-known source provider type; any other type is registered at the next index after these. A
// consumer learns every index, static or dynamic, from the SourceProviderRegistered records instead.
enum NvTrace_ThreadEvent_StaticSourceProviderIndex
{
    NvTrace_ThreadEvent_StaticSourceProviderIndex_ThreadEvent = 0,
    NvTrace_ThreadEvent_StaticSourceProviderIndex_CUDA,
    NvTrace_ThreadEvent_StaticSourceProviderIndex_NVTX,

    NvTrace_ThreadEvent_StaticSourceProviderIndex_COUNT,
    NvTrace_ThreadEvent_StaticSourceProviderIndex_FORCE_INT = 0x7fffffff
};

// One record per registered source provider, published into the SourceProviderRegistrations circular
// buffer before the source provider writes its first record. A registration lasts as long as the
// ThreadEvent provider and is never undone. The source provider also writes its index into its own
// provider buffer header (NvTrace_ProviderBufferHeader::threadEventSourceProviderIndex), which is how a
// consumer binds the index to that provider.
struct NvTrace_ThreadEvent_SourceProviderRegistered
{
    uint64_t registrationTimestampTsc;

    uint32_t sourceProviderIndex;
    uint32_t rsvd000c;

    NVTRACE_UUID providerTypeUuid;
};
#define NvTrace_ThreadEvent_SourceProviderRegistered_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_ThreadEvent_SourceProviderRegistered, providerTypeUuid)

// The thread every following record in the cell belongs to. 0 = unattributed (the shared slot)
struct NvTrace_ThreadEvent_ThreadChanged
{
    uint64_t uniqueThreadId;
};
#define NvTrace_ThreadEvent_ThreadChanged_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_ThreadEvent_ThreadChanged, uniqueThreadId)

struct NvTrace_ThreadEvent_ThreadCreated
{
    uint64_t creationTimestampTsc;

    // 0 while alive
    uint64_t destructionTimestampTsc;

    // A unique ID defined by NvTrace, never 0. Has no relation to the nativeThreadId. Event records
    // attributed to uniqueThreadId 0 are associated with no ThreadCreated record.
    uint64_t uniqueThreadId;

    uint32_t nativeThreadId;
};
#define NvTrace_ThreadEvent_ThreadCreated_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_ThreadEvent_ThreadCreated, nativeThreadId)

struct NvTrace_ThreadEvent_ThreadDestroyed
{
    uint64_t destructionTimestampTsc;
};
#define NvTrace_ThreadEvent_ThreadDestroyed_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_ThreadEvent_ThreadDestroyed, destructionTimestampTsc)

// clang-format off
#define NVTRACE_THREAD_EVENT_RECORD_IDENTIFIER_TYPE_FIELD 19: 0
#define NVTRACE_THREAD_EVENT_RECORD_IDENTIFIER_SOURCE_INDEX_FIELD 31:20
// clang-format on

#define NVTRACE_THREAD_EVENT_RECORD_IDENTIFIER(sourceProviderIndex, recordType) (NVTRACE_SETFIELD32(0, NVTRACE_THREAD_EVENT_RECORD_IDENTIFIER_SOURCE_INDEX_FIELD, (sourceProviderIndex)) | NVTRACE_SETFIELD32(0, NVTRACE_THREAD_EVENT_RECORD_IDENTIFIER_TYPE_FIELD, (recordType)))
#define NVTRACE_THREAD_EVENT_RECORD_IDENTIFIER_SOURCE_INDEX(recordIdentifier) NVTRACE_GETFIELD32((recordIdentifier), NVTRACE_THREAD_EVENT_RECORD_IDENTIFIER_SOURCE_INDEX_FIELD)
#define NVTRACE_THREAD_EVENT_RECORD_IDENTIFIER_TYPE(recordIdentifier) NVTRACE_GETFIELD32((recordIdentifier), NVTRACE_THREAD_EVENT_RECORD_IDENTIFIER_TYPE_FIELD)

// A record of the ThreadEvent provider's own, at its static source provider index
#define NVTRACE_COMMON_THREAD_EVENT_RECORD_IDENTIFIER(recordType) NVTRACE_THREAD_EVENT_RECORD_IDENTIFIER(NvTrace_ThreadEvent_StaticSourceProviderIndex_ThreadEvent, (recordType))

#ifdef __cplusplus
} // extern "C"
#endif
NVTRACE_RESTORE_WARNINGS()
