/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/DataFormats/nvtrace_common_types.h>
#include <NvTrace/DataFormats/nvtrace_compiler_compatibility.h>
#include <NvTrace/DataFormats/nvtrace_nvtx_trace.h>
#include <NvTrace/DataFormats/nvtrace_provider_buffer_v1.h>


NVTRACE_SUPPRESS_WARNINGS()
#ifdef __cplusplus
extern "C" {
#endif


// The V1 NVTX Trace provider buffer, as built drivers publish it. The successor format in
// nvtrace_nvtx_trace.h is a generic provider buffer; the _V1 suffix keeps the two vocabularies apart.
// The payload shapes both wires share -- the message, color and payload type enums, EventAttributes,
// StackRef and the four thread events -- live unsuffixed in nvtrace_nvtx_trace.h.

NVTRACE_WEAK_CONST NVTRACE_UUID NvTrace_NvtxTrace_ProviderTypeUuid_V1 = NVTRACE_UUID_VALUE(0x55cb7ea9, 0x220a, 0x4660, 0xa6, 0xa6, 0x54, 0xd1, 0x25, 0x67, 0x82, 0x77);


/// Buffer layout structure for the V1 NVTX Trace provider.
/// Note: this provider will allocate one giant buffer for all sub-buffers. Thus, offsets are sufficient to discover sub-buffers.
struct NvTrace_NvtxTrace_ProviderBufferLayout_V1
{
    /// This is a COFF-style scheme, where each offset is relative to the pointer of the provider buffer header struct.

    /// points at NvTrace_NvtxTrace_EventVersionInfo_V1
    uint64_t eventVersionInfoOffset;
    uint32_t eventVersionInfoSize;
    uint32_t eventVersionInfoMinorVersion;

    /// points at a flatbuffer of type nv::trace::FB::Provider::NvtxTrace::ProviderConfiguration
    uint64_t configurationOffset;
    uint32_t configurationSize;

    /// points at NvTrace_NvtxTrace_Demand_V1
    uint64_t currentDemandOffset;
    /// points at NvTrace_NvtxTrace_Demand_V1
    uint64_t peakDemandOffset;

    /// points at string table buffer (linear byte array with null-terminated strings)
    uint64_t stringTableOffset;
    uint64_t stringTableSize;

    /// points at NvTrace_NvtxTrace_DomainState_V1
    uint64_t domainStatesOffset;
    uint32_t domainStatesCount;

    /// points at an array of state chunks, each stateChunkSize bytes, addressed by NvTrace_RevisionedRef.
    /// Unlike every other adaptive chunk producer, the V1 NVTX provider encodes a record's
    /// payloadEndOffsetDiv8 from the chunk's start, not its end.
    uint64_t stateChunksOffset;
    uint32_t stateChunksCount;
    uint32_t stateChunkSize;

    /// points at an array of NvTrace_NvtxTrace_PushPopRangeState_V1
    uint64_t pushPopRangeStatesOffset;
    uint32_t pushPopRangeStatesCount;
    /// Size of message stored inline in the trailer after each PushPopRangeState.
    uint32_t pushPopRangeInlineMessageSize;

    /// points at an array of NvTrace_NvtxTrace_StackGroupState_V1
    uint64_t stackGroupStatesOffset;
    uint32_t stackGroupStatesCount;
    /// Number of NvTrace_NvtxTrace_StackRef entries stored in the trailer after each StackGroupState.
    uint32_t stackGroupInlineStackRefCount;
};
#define NvTrace_NvtxTrace_ProviderBufferLayout_V1_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_NvtxTrace_ProviderBufferLayout_V1, stackGroupInlineStackRefCount)


struct NvTrace_NvtxTrace_EventVersionInfo_V1
{
    uint32_t demandSize;
    uint32_t demandMinorVersion;

    // object lifetime events
    uint32_t objectLifetimeEventHeaderSize;
    uint32_t objectLifetimeEventHeaderMinorVersion;
    // The payload will be a union so it does not itself have a minor version;
    // but each individual struct within the union must have a minor version for extensibility.
    uint32_t objectLifetimeEventPayloadSize;

    // lifetime event payloads
    uint32_t objectLifetimeEventPayload_StringTableEntryCreatedSize;
    uint32_t objectLifetimeEventPayload_StringTableEntryCreatedMinorVersion;
    uint32_t objectLifetimeEventPayload_DomainCreatedSize;
    uint32_t objectLifetimeEventPayload_DomainCreatedMinorVersion;
    uint32_t objectLifetimeEventPayload_PushPopRangeCreatedSize;
    uint32_t objectLifetimeEventPayload_PushPopRangeCreatedMinorVersion;
    uint32_t objectLifetimeEventPayload_StackGroupCreatedSize;
    uint32_t objectLifetimeEventPayload_StackGroupCreatedMinorVersion;

    // state structs
    uint32_t domainStateSize;
    uint32_t domainStateMinorVersion;
    uint32_t stateChunkHeaderSize;
    uint32_t stateChunkHeaderMinorVersion;
    uint32_t stateChunkRecordSize;
    uint32_t stateChunkRecordMinorVersion;
    uint32_t pushPopRangeStateSize;
    uint32_t pushPopRangeStateMinorVersion;
    uint32_t stackRefSize;
    uint32_t stackRefMinorVersion;
    uint32_t stackGroupStateSize;
    uint32_t stackGroupStateMinorVersion;

    // thread events
    uint32_t markSize;
    uint32_t markMinorVersion;
    uint32_t rangePopSize;
    uint32_t rangePopMinorVersion;
    uint32_t droppedRangePushSize;
    uint32_t droppedRangePushMinorVersion;
    uint32_t droppedRangePopSize;
    uint32_t droppedRangePopMinorVersion;

    // event attributes
    uint32_t eventAttributesSize;
    uint32_t eventAttributesMinorVersion;
};
#define NvTrace_NvtxTrace_EventVersionInfo_V1_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_NvtxTrace_EventVersionInfo_V1, eventAttributesMinorVersion)


struct NvTrace_NvtxTrace_Demand_V1
{
    uint64_t stringTableSize;
    uint32_t domainCount;
};
#define NvTrace_NvtxTrace_Demand_V1_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_NvtxTrace_Demand_V1, domainCount)


/// The V1 record types of the ThreadEvent-carried NVTX records, in the NVTX provider band.
enum NvTrace_NvtxTrace_EventType_V1
{
    NvTrace_NvtxTrace_EventType_ObjectLifetime_LO_V1 = 0,

    NvTrace_NvtxTrace_EventType_ObjectLifetime_StringTableEntryCreated_V1,
    NvTrace_NvtxTrace_EventType_ObjectLifetime_DomainCreated_V1,
    NvTrace_NvtxTrace_EventType_ObjectLifetime_PushPopRangeCreated_V1,
    NvTrace_NvtxTrace_EventType_ObjectLifetime_StackGroupCreated_V1,
    NvTrace_NvtxTrace_EventType_ObjectLifetime_StackGroupDestroyed_V1,

    NvTrace_NvtxTrace_EventType_ObjectLifetime_HI_V1,
    NvTrace_NvtxTrace_EventType_ObjectLifetime_MAX_V1 = 0x0ffff,

    NvTrace_NvtxTrace_EventType_Thread_LO_V1 = 0x10000,

    NvTrace_NvtxTrace_EventType_Thread_Mark_V1,
    NvTrace_NvtxTrace_EventType_Thread_RangePop_V1,
    NvTrace_NvtxTrace_EventType_Thread_DroppedRangePush_V1,
    NvTrace_NvtxTrace_EventType_Thread_DroppedRangePop_V1,

    NvTrace_NvtxTrace_EventType_Thread_HI_V1,
    NvTrace_NvtxTrace_EventType_Thread_MAX_V1 = 0x1ffff,

    NvTrace_NvtxTrace_EventType_FORCE_INT_V1 = 0x7fffffff
};


// The object the event names comes from its record type and the decoded identifier:
// StringTableEntryCreated:    tableEntryIndex (= offset in bytes of the string table entry), revision 0
// DomainCreated:              tableEntryIndex + tableEntryRevision
// PushPopRangeCreated:        tableEntryIndex + tableEntryRevision
// StackGroupCreated/Destroyed: tableEntryIndex + tableEntryRevision
struct NvTrace_NvtxTrace_ObjectLifetimeEventHeader_V1
{
    uint64_t timestampTsc;
    uint64_t tableEntryIndex;
    uint64_t tableEntryRevision;
};
#define NvTrace_NvtxTrace_ObjectLifetimeEventHeader_V1_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_NvtxTrace_ObjectLifetimeEventHeader_V1, tableEntryRevision)


struct NvTrace_NvtxTrace_DomainState_V1
{
    /// The data revision; used for consistency checks (odd = write in progress).
    volatile uint64_t revision;

    /// Byte offset into the string table where the domain name (UTF-8, null-terminated) is stored, or 0 if not available.
    uint64_t nameStringTableEntryOffset;
};
#define NvTrace_NvtxTrace_DomainState_V1_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_NvtxTrace_DomainState_V1, nameStringTableEntryOffset)


// NvTrace_RecordDesc::recordType values for records stored in the chunk chain of a
// NvTrace_NvtxTrace_PushPopRangeState_V1 entry.
enum NvTrace_NvtxTrace_PushPopRangeStateChunkRecordType_V1
{
    NvTrace_NvtxTrace_PushPopRangeStateChunkRecordType_Invalid_V1 = 0,

    /// Payload: message bytes beyond the inline trailer, concatenated in chain order after the inline bytes (total = messageSize).
    NvTrace_NvtxTrace_PushPopRangeStateChunkRecordType_MessageSegment_V1,
    /// Payload: NvTrace_NvtxTrace_EventAttributes.
    NvTrace_NvtxTrace_PushPopRangeStateChunkRecordType_EventAttributes_V1,

    NvTrace_NvtxTrace_PushPopRangeStateChunkRecordType_COUNT_V1,
    NvTrace_NvtxTrace_PushPopRangeStateChunkRecordType_FORCE_INT_V1 = 0x7fffffff
};


/// The "none" value of every V1 NVTX ref field -- ancestorRef, topRef, a V1 CUDA graph node's
/// nvtxTraceStackGroupRef; the generic format's is NVTRACE_INVALID_REVISIONED_REF.
NVTRACE_WEAK_CONST uint64_t NvTrace_NvtxTrace_INVALID_REF_V1 = 0xFFFFFFFFFFFFFFFFull;

struct NvTrace_NvtxTrace_PushPopRangeState_V1
{
    volatile uint64_t endTimestampTsc; // 0 if range is still active; not guarded by revision

    /// The data revision; used for consistency checks (odd = write in progress).
    volatile uint64_t revision;

    /// Head of the chunk chain holding the message bytes and attributes that don't fit inline. 0 if none.
    NvTrace_RevisionedRef headChunkRef;

    uint64_t startTimestampTsc;
    /// Packed (revision, index) of the nearest allocated ancestor, or NvTrace_NvtxTrace_INVALID_REF_V1 if none.
    uint64_t ancestorRef;
    uint64_t uniqueThreadId;
    uint32_t domainIndex;
    uint16_t nestingDepth;
    /// NvTrace_NvtxTrace_MessageType
    uint16_t messageType;
    /// Bytes of the message actually stored, inline and across chunk segments.
    uint32_t messageSize;
    // Trailer: pushPopRangeInlineMessageSize message bytes follow.
};
#define NvTrace_NvtxTrace_PushPopRangeState_V1_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_NvtxTrace_PushPopRangeState_V1, messageSize)


// NvTrace_RecordDesc::recordType values for records stored in the chunk chain of a
// NvTrace_NvtxTrace_StackGroupState_V1 entry.
enum NvTrace_NvtxTrace_StackGroupStateChunkRecordType_V1
{
    NvTrace_NvtxTrace_StackGroupStateChunkRecordType_Invalid_V1 = 0,

    /// Payload: NvTrace_NvtxTrace_StackRef.
    NvTrace_NvtxTrace_StackGroupStateChunkRecordType_StackRef_V1,

    NvTrace_NvtxTrace_StackGroupStateChunkRecordType_COUNT_V1,
    NvTrace_NvtxTrace_StackGroupStateChunkRecordType_FORCE_INT_V1 = 0x7fffffff
};


struct NvTrace_NvtxTrace_StackGroupState_V1
{
    /// The data revision; used for consistency checks (odd = write in progress).
    volatile uint64_t revision;
    /// Head of the chunk chain holding the StackRef entries that don't fit inline. 0 if none.
    NvTrace_RevisionedRef headChunkRef;
    /// Total count of StackRef entries across the inline trailer and chunk chain.
    uint32_t activeDomainCount;
    // Trailer: NvTrace_NvtxTrace_StackRef[stackGroupInlineStackRefCount] follows.
};
#define NvTrace_NvtxTrace_StackGroupState_V1_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_NvtxTrace_StackGroupState_V1, activeDomainCount)


// V1 Marks carry one message length, placing their type and attributes flag at different
// offsets from the generic Mark. This layout remains the old driver's wire contract.
struct NvTrace_NvtxTrace_Mark_V1
{
    uint64_t timestampTsc;
    uint32_t domainIndex;
    uint32_t messageSize;
    uint16_t messageType;
    uint8_t hasEventAttributes;
    uint8_t rsvd0013[5];
};
#define NvTrace_NvtxTrace_Mark_V1_MINOR_VERSION_A NVTRACE_FIELD_END_OFFSET(struct NvTrace_NvtxTrace_Mark_V1, hasEventAttributes)

#ifdef __cplusplus
} // extern "C"
#endif
NVTRACE_RESTORE_WARNINGS()
