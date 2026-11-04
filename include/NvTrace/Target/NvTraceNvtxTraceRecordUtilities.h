/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/DataFormats/nvtrace_nvtx_trace.h>

namespace nv::trace {

    inline constexpr uint32_t kNvtxTraceRecordTypeCount = NvTrace_NvtxTrace_RecordType_COUNT;

    // This build's producer publishes sizeof as the leading segment's size, so a struct that leads a
    // dynamic tail of 8-byte-aligned elements must be sized to that alignment.
    static_assert(sizeof(NvTrace_NvtxTrace_StackGroupCreated) % alignof(NvTrace_NvtxTrace_StackRef) == 0);
    static_assert(sizeof(NvTrace_NvtxTrace_StackGroupStackRefs) % alignof(NvTrace_NvtxTrace_StackRef) == 0);
    static_assert(sizeof(NvTrace_NvtxTrace_PushPopRangeCreated) % alignof(NvTrace_NvtxTrace_EventAttributes) == 0);
    static_assert(sizeof(NvTrace_NvtxTrace_Mark) % alignof(NvTrace_NvtxTrace_EventAttributes) == 0);

    inline void FillNvtxTraceRecordVersionInfo(NvTrace_NvtxTrace_RecordVersionInfo& recordVersionInfo)
    {
        recordVersionInfo.recordVersionInfoSize = sizeof(NvTrace_NvtxTrace_RecordVersionInfo);
        recordVersionInfo.recordVersionInfoMinorVersion = NvTrace_NvtxTrace_RecordVersionInfo_MINOR_VERSION_A;
        recordVersionInfo.providerStatisticsSize = sizeof(NvTrace_NvtxTrace_ProviderStatistics);
        recordVersionInfo.providerStatisticsMinorVersion = NvTrace_NvtxTrace_ProviderStatistics_MINOR_VERSION_A;

        recordVersionInfo.stringRegisteredSize = sizeof(NvTrace_NvtxTrace_StringRegistered);
        recordVersionInfo.stringRegisteredMinorVersion = NvTrace_NvtxTrace_StringRegistered_MINOR_VERSION_A;
        recordVersionInfo.stringSegmentSize = sizeof(NvTrace_NvtxTrace_StringSegment);
        recordVersionInfo.stringSegmentMinorVersion = NvTrace_NvtxTrace_StringSegment_MINOR_VERSION_A;
        recordVersionInfo.domainCreatedSize = sizeof(NvTrace_NvtxTrace_DomainCreated);
        recordVersionInfo.domainCreatedMinorVersion = NvTrace_NvtxTrace_DomainCreated_MINOR_VERSION_A;
        recordVersionInfo.pushPopRangeCreatedSize = sizeof(NvTrace_NvtxTrace_PushPopRangeCreated);
        recordVersionInfo.pushPopRangeCreatedMinorVersion = NvTrace_NvtxTrace_PushPopRangeCreated_MINOR_VERSION_A;
        recordVersionInfo.pushPopRangeMessageSegmentSize = sizeof(NvTrace_NvtxTrace_PushPopRangeMessageSegment);
        recordVersionInfo.pushPopRangeMessageSegmentMinorVersion = NvTrace_NvtxTrace_PushPopRangeMessageSegment_MINOR_VERSION_A;
        recordVersionInfo.stackGroupCreatedSize = sizeof(NvTrace_NvtxTrace_StackGroupCreated);
        recordVersionInfo.stackGroupCreatedMinorVersion = NvTrace_NvtxTrace_StackGroupCreated_MINOR_VERSION_A;
        recordVersionInfo.stackGroupStackRefsSize = sizeof(NvTrace_NvtxTrace_StackGroupStackRefs);
        recordVersionInfo.stackGroupStackRefsMinorVersion = NvTrace_NvtxTrace_StackGroupStackRefs_MINOR_VERSION_A;
        recordVersionInfo.stackGroupDestroyedSize = sizeof(NvTrace_NvtxTrace_StackGroupDestroyed);
        recordVersionInfo.stackGroupDestroyedMinorVersion = NvTrace_NvtxTrace_StackGroupDestroyed_MINOR_VERSION_A;

        recordVersionInfo.eventAttributesSize = sizeof(NvTrace_NvtxTrace_EventAttributes);
        recordVersionInfo.eventAttributesMinorVersion = NvTrace_NvtxTrace_EventAttributes_MINOR_VERSION_A;
        recordVersionInfo.stackRefSize = sizeof(NvTrace_NvtxTrace_StackRef);
        recordVersionInfo.stackRefMinorVersion = NvTrace_NvtxTrace_StackRef_MINOR_VERSION_A;

        recordVersionInfo.markSize = sizeof(NvTrace_NvtxTrace_Mark);
        recordVersionInfo.markMinorVersion = NvTrace_NvtxTrace_Mark_MINOR_VERSION_A;
        recordVersionInfo.markMessageSegmentSize = sizeof(NvTrace_NvtxTrace_MarkMessageSegment);
        recordVersionInfo.markMessageSegmentMinorVersion = NvTrace_NvtxTrace_MarkMessageSegment_MINOR_VERSION_A;
        recordVersionInfo.rangePopSize = sizeof(NvTrace_NvtxTrace_RangePop);
        recordVersionInfo.rangePopMinorVersion = NvTrace_NvtxTrace_RangePop_MINOR_VERSION_A;
        recordVersionInfo.droppedRangePushSize = sizeof(NvTrace_NvtxTrace_DroppedRangePush);
        recordVersionInfo.droppedRangePushMinorVersion = NvTrace_NvtxTrace_DroppedRangePush_MINOR_VERSION_A;
        recordVersionInfo.droppedRangePopSize = sizeof(NvTrace_NvtxTrace_DroppedRangePop);
        recordVersionInfo.droppedRangePopMinorVersion = NvTrace_NvtxTrace_DroppedRangePop_MINOR_VERSION_A;
    }

} // namespace nv::trace
