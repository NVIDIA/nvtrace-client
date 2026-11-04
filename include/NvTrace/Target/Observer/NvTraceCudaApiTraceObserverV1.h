/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/DataFormats/nvtrace_cuda_api_trace.h>
#include <NvTrace/DataFormats/nvtrace_cuda_api_trace_v1.h>
#include <NvTrace/DataFormats/nvtrace_nvtx_trace_v1.h>
#include <NvTrace/DataFormats/nvtrace_thread_event.h>
#include <NvTrace/DataFormats/nvtrace_thread_event_v1.h>
#include <NvTrace/Target/NvTraceClientLogger.h>
#include <NvTrace/Target/NvTraceCudaApiTraceProviderConfiguration_generated.h>
#include <NvTrace/Target/NvTraceCudaApiTraceProviderConstants.h>
#include <NvTrace/Target/NvTraceCudaApiTraceRecordUtilities.h>
#include <NvTrace/Target/NvTraceFlatBufferUtilities.h>
#include <NvTrace/Target/NvTraceMathUtilities.h>
#include <NvTrace/Target/NvTraceMemoryFence.h>
#include <NvTrace/Target/NvTraceScopeExitGuard.h>
#include <NvTrace/Target/NvTraceUuidUtilities.h>
#include <NvTrace/Target/Observer/NvTraceCircularBufferReaderV1.h>
#include <NvTrace/Target/Observer/NvTraceCudaApiTraceConsumerTypes.h>
#include <NvTrace/Target/Observer/NvTraceObserverUtilities.h>
#include <NvTrace/Target/Observer/NvTraceStateTableReader.h>
#include <NvTrace/Target/Observer/NvTraceThreadEventConsumerTypes.h>

#include <algorithm>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

namespace nv::trace {

    /// \brief Observer of a single V1 GPU timestamp FIFO, one per stream slot.
    class CudaApiTraceGpuTimestampFifoObserverV1
    {
    public:
        ~CudaApiTraceGpuTimestampFifoObserverV1() = default;
        CudaApiTraceGpuTimestampFifoObserverV1() = default;

        CudaApiTraceGpuTimestampFifoObserverV1(const CudaApiTraceGpuTimestampFifoObserverV1&) = delete;
        CudaApiTraceGpuTimestampFifoObserverV1(CudaApiTraceGpuTimestampFifoObserverV1&& rhs) noexcept
        {
            *this = std::move(rhs);
        }

        CudaApiTraceGpuTimestampFifoObserverV1& operator=(const CudaApiTraceGpuTimestampFifoObserverV1&) = delete;
        CudaApiTraceGpuTimestampFifoObserverV1& operator=(CudaApiTraceGpuTimestampFifoObserverV1&& rhs) noexcept
        {
            if (this != &rhs)
            {
                m_state = std::exchange(rhs.m_state, {});
            }
            return *this;
        }

        void Reset() noexcept
        {
            m_state = {};
        }

        void Initialize(
            const NvTrace_CircularBufferState* pCircularBufferState,
            const uint8_t* pGpuTimestampFifoCells,
            size_t gpuTimestampFifoCellSize,
            uint32_t gpuTimestampFifoCellCount,
            const uint8_t* pGpuTimestampReportSection,
            uint32_t gpuTimestampChunksCount,
            uint32_t gpuTimestampReportsPerChunk)
        {
            m_state.pGpuTimestampReportSection = reinterpret_cast<const NvTrace_CUDA_ApiTrace_GpuTimestampReport_V1*>(pGpuTimestampReportSection);
            m_state.gpuTimestampChunksCount = gpuTimestampChunksCount;
            m_state.gpuTimestampReportsPerChunk = gpuTimestampReportsPerChunk;
            m_state.gpuTimestampReportsPerChunkLog2 = CeilLog2(gpuTimestampReportsPerChunk);
            m_state.gpuTimestampReportsLog2 = CeilLog2(gpuTimestampChunksCount) + m_state.gpuTimestampReportsPerChunkLog2;

            const BufferAccessor bufferAccessor = {pGpuTimestampFifoCells, gpuTimestampFifoCellSize};
            m_state.cbReader.Initialize(pCircularBufferState, bufferAccessor, gpuTimestampFifoCellCount);
        }

        // Greedily consumes all available timestamp reports.
        //
        //   void fnOnTimestampEvent(uint64_t gpuTimestampRef, uint64_t ptimer)
        //     Accepts every delivered report.
        //
        //   bool fnOnTimestampEvent(uint64_t gpuTimestampRef, uint64_t ptimer)
        //     Return true to accept the report and advance the cursor.
        //     Return false to retain the report and stop until the next ConsumeEvents().
        template <typename TFN_OnTimestampEvent>
        void ConsumeEvents(TFN_OnTimestampEvent&& fnOnTimestampEvent)
        {
            auto fnDrainCurrentChunk = [&]() {
                const uint64_t streamSequenceId = m_state.currentFifoCell.gpuTimestampRef >> m_state.gpuTimestampReportsLog2;
                while (m_state.gpuTimestampReportsRemaining != 0)
                {
                    const uint64_t storedStreamSequenceId = LoadAcquire(&m_state.pGpuTimestampReport->streamSequenceId);
                    if (streamSequenceId < storedStreamSequenceId)
                    {
                        // The chunk was overwritten by newer commands; skip reading this chunk.
                        m_state.gpuTimestampReportsRemaining = 0;
                        return true;
                    }
                    if (storedStreamSequenceId < streamSequenceId)
                    {
                        // The chunk is not fully written yet; suspend reading on this consumption cycle.
                        return false;
                    }

                    // Copy out, then re-check the sequence id: an unchanged id proves the copied bytes
                    // belong to this report, so ptimer cannot have been paired with a stale/newer write.
                    NvTrace_CUDA_ApiTrace_GpuTimestampReport_V1 gpuTimestampReport;
                    Memcpy64(&gpuTimestampReport, m_state.pGpuTimestampReport, sizeof(gpuTimestampReport));
                    ReadFence();
                    if (LoadRelaxed(&m_state.pGpuTimestampReport->streamSequenceId) != storedStreamSequenceId)
                    {
                        return false; // Overwritten mid-copy; not yet consistently written.
                    }

                    if constexpr (std::is_same_v<std::invoke_result_t<TFN_OnTimestampEvent, uint64_t, uint64_t>, void>)
                    {
                        fnOnTimestampEvent(m_state.gpuTimestampRef, gpuTimestampReport.ptimer);
                    }
                    else if (!static_cast<bool>(fnOnTimestampEvent(m_state.gpuTimestampRef, gpuTimestampReport.ptimer)))
                    {
                        return false;
                    }

                    m_state.gpuTimestampReportsRemaining -= 1;
                    m_state.pGpuTimestampReport += 1;
                    m_state.gpuTimestampRef += 1;
                }
                return true;
            };

            if (m_state.gpuTimestampReportsRemaining != 0)
            {
                if (!fnDrainCurrentChunk())
                {
                    return;
                }
            }

            const uint64_t readEndSequenceNumber = m_state.cbReader.AcquireCurrentEndSequenceNumber();
            while (m_state.gpuTimestampReportsRemaining == 0)
            {
                const bool isValidFifoCell = m_state.cbReader.ConsumeOneRecord(
                    readEndSequenceNumber,
                    [&](const NvTrace_CUDA_ApiTrace_GpuTimestampFifoCell_V1* pFifoCell) {
                        m_state.currentFifoCell = *pFifoCell;
                    });
                if (!isValidFifoCell)
                {
                    return; // Buffer overrun (reader fell behind).
                }

                // This logic handles arbitrary non-power-of-2 chunk sizes. The mask shift is
                // CeilLog2 of a 32-bit count, so it can reach 32 -- 64-bit math keeps it defined.
                const uint64_t chunkIndex = (m_state.currentFifoCell.gpuTimestampRef >> m_state.gpuTimestampReportsPerChunkLog2) & ((1ull << (m_state.gpuTimestampReportsLog2 - m_state.gpuTimestampReportsPerChunkLog2)) - 1);
                if (m_state.gpuTimestampChunksCount <= chunkIndex)
                {
                    return; // Corrupt cell: the mask above admits indices the non-power-of-2 chunk count does not back.
                }

                m_state.pGpuTimestampReport = m_state.pGpuTimestampReportSection + chunkIndex * m_state.gpuTimestampReportsPerChunk;
                m_state.gpuTimestampReportsRemaining = m_state.gpuTimestampReportsPerChunk;
                m_state.gpuTimestampRef = m_state.currentFifoCell.gpuTimestampRef;

                if (!fnDrainCurrentChunk())
                {
                    return;
                }
            }
        }

    private:
        struct BufferAccessor
        {
            const uint8_t* pFifoCells = nullptr;
            size_t fifoCellSize = 0;

            template <typename TFN_ReadEvent>
            void ReadRecord(uint32_t readCellIndex, uint64_t /* readSequenceNumber */, TFN_ReadEvent&& fnReadEvent) const
            {
                const auto* pFifoCell = reinterpret_cast<const NvTrace_CUDA_ApiTrace_GpuTimestampFifoCell_V1*>(pFifoCells + readCellIndex * fifoCellSize);
                fnReadEvent(pFifoCell);
            }

            uint64_t ReadRecordTimestamp(uint32_t /* readCellIndex */) const
            {
                return UINT64_MAX;
            }
        };

    private:
        struct State
        {
            const NvTrace_CUDA_ApiTrace_GpuTimestampReport_V1* pGpuTimestampReportSection = nullptr;
            uint32_t gpuTimestampChunksCount = 0;
            uint32_t gpuTimestampReportsPerChunk = 0;
            uint32_t gpuTimestampReportsPerChunkLog2 = 0;
            uint32_t gpuTimestampReportsLog2 = 0;

            CircularBufferReaderV1<BufferAccessor> cbReader = {};
            NvTrace_CUDA_ApiTrace_GpuTimestampFifoCell_V1 currentFifoCell = {};
            const NvTrace_CUDA_ApiTrace_GpuTimestampReport_V1* pGpuTimestampReport = nullptr;
            size_t gpuTimestampReportsRemaining = 0;
            uint64_t gpuTimestampRef = 0;
        };
        State m_state;
    };

    inline NvTrace_CUDA_ApiTrace_GraphNodeType TranslateCudaApiTraceV1GraphNodeType(NvTrace_CUDA_ApiTrace_GraphNodeType_V1 graphNodeType)
    {
        static_assert(static_cast<int>(NvTrace_CUDA_ApiTrace_GraphNodeType_Other) == static_cast<int>(NvTrace_CUDA_ApiTrace_GraphNodeType_Other_V1) && static_cast<int>(NvTrace_CUDA_ApiTrace_GraphNodeType_SubgridSchedule) == static_cast<int>(NvTrace_CUDA_ApiTrace_GraphNodeType_SubgridSchedule_V1), "The V2 graph-node type enum must keep the frozen V1 wire values for the identity translation");

        if (graphNodeType < NvTrace_CUDA_ApiTrace_GraphNodeType_Other_V1 || NvTrace_CUDA_ApiTrace_GraphNodeType_SubgridSchedule_V1 < graphNodeType)
        {
            return NvTrace_CUDA_ApiTrace_GraphNodeType_Invalid;
        }
        return static_cast<NvTrace_CUDA_ApiTrace_GraphNodeType>(graphNodeType);
    }

    inline NvTrace_CUDA_ApiTrace_RecordType TranslateCudaApiTraceV1ObjectLifetimeCreatedRecordType(uint32_t v1RecordType)
    {
        switch (v1RecordType)
        {
            case NvTrace_CUDA_ApiTrace_RecordType_ObjectLifetime_ContextCreated_V1: return NvTrace_CUDA_ApiTrace_RecordType_ContextCreated;
            case NvTrace_CUDA_ApiTrace_RecordType_ObjectLifetime_StreamCreated_V1: return NvTrace_CUDA_ApiTrace_RecordType_StreamCreated;
            case NvTrace_CUDA_ApiTrace_RecordType_ObjectLifetime_ModuleLoaded_V1: return NvTrace_CUDA_ApiTrace_RecordType_ModuleLoaded;
            case NvTrace_CUDA_ApiTrace_RecordType_ObjectLifetime_GraphExecCreated_V1: return NvTrace_CUDA_ApiTrace_RecordType_GraphExecCreated;
            case NvTrace_CUDA_ApiTrace_RecordType_ObjectLifetime_ChannelCreated_V1: return NvTrace_CUDA_ApiTrace_RecordType_ChannelCreated;
            default: return NvTrace_CUDA_ApiTrace_RecordType_INVALID;
        }
    }

    inline NvTrace_CUDA_ApiTrace_RecordType TranslateCudaApiTraceV1ObjectLifetimeDestroyedRecordType(uint32_t v1RecordType)
    {
        switch (v1RecordType)
        {
            case NvTrace_CUDA_ApiTrace_RecordType_ObjectLifetime_ContextDestroyed_V1: return NvTrace_CUDA_ApiTrace_RecordType_ContextDestroyed;
            case NvTrace_CUDA_ApiTrace_RecordType_ObjectLifetime_StreamDestroyed_V1: return NvTrace_CUDA_ApiTrace_RecordType_StreamDestroyed;
            case NvTrace_CUDA_ApiTrace_RecordType_ObjectLifetime_ModuleUnloaded_V1: return NvTrace_CUDA_ApiTrace_RecordType_ModuleUnloaded;
            case NvTrace_CUDA_ApiTrace_RecordType_ObjectLifetime_GraphExecDestroyed_V1: return NvTrace_CUDA_ApiTrace_RecordType_GraphExecDestroyed;
            default: return NvTrace_CUDA_ApiTrace_RecordType_INVALID;
        }
    }

    /// The payload layouts are shared between the wires; only the numbering moved.
    inline NvTrace_CUDA_ApiTrace_RecordType TranslateCudaApiTraceV1ThreadEventRecordType(uint32_t v1RecordType)
    {
        switch (v1RecordType)
        {
            case NvTrace_CUDA_ApiTrace_RecordType_ThreadEvent_CudaDriverApiEnter_V1: return NvTrace_CUDA_ApiTrace_RecordType_CudaDriverApiEnter;
            case NvTrace_CUDA_ApiTrace_RecordType_ThreadEvent_CudaDriverApiExit_V1: return NvTrace_CUDA_ApiTrace_RecordType_CudaDriverApiExit;
            case NvTrace_CUDA_ApiTrace_RecordType_ThreadEvent_CudaRuntimeApiEnter_V1: return NvTrace_CUDA_ApiTrace_RecordType_CudaRuntimeApiEnter;
            case NvTrace_CUDA_ApiTrace_RecordType_ThreadEvent_CudaRuntimeApiExit_V1: return NvTrace_CUDA_ApiTrace_RecordType_CudaRuntimeApiExit;
            case NvTrace_CUDA_ApiTrace_RecordType_ThreadEvent_ImmediateKernelLaunch_V1: return NvTrace_CUDA_ApiTrace_RecordType_ImmediateKernelLaunch;
            case NvTrace_CUDA_ApiTrace_RecordType_ThreadEvent_GraphLaunch_V1: return NvTrace_CUDA_ApiTrace_RecordType_GraphLaunch;
            case NvTrace_CUDA_ApiTrace_RecordType_ThreadEvent_MemcpyBegin_V1: return NvTrace_CUDA_ApiTrace_RecordType_MemcpyBegin;
            case NvTrace_CUDA_ApiTrace_RecordType_ThreadEvent_MemcpyEnd_V1: return NvTrace_CUDA_ApiTrace_RecordType_MemcpyEnd;
            case NvTrace_CUDA_ApiTrace_RecordType_ThreadEvent_MemsetBegin_V1: return NvTrace_CUDA_ApiTrace_RecordType_MemsetBegin;
            case NvTrace_CUDA_ApiTrace_RecordType_ThreadEvent_MemsetEnd_V1: return NvTrace_CUDA_ApiTrace_RecordType_MemsetEnd;
            case NvTrace_CUDA_ApiTrace_RecordType_ThreadEvent_HostCallbackLaunch_V1: return NvTrace_CUDA_ApiTrace_RecordType_HostCallbackLaunch;
            case NvTrace_CUDA_ApiTrace_RecordType_ThreadEvent_HostCallbackExecBegin_V1: return NvTrace_CUDA_ApiTrace_RecordType_HostCallbackExecBegin;
            case NvTrace_CUDA_ApiTrace_RecordType_ThreadEvent_HostCallbackExecEnd_V1: return NvTrace_CUDA_ApiTrace_RecordType_HostCallbackExecEnd;
            case NvTrace_CUDA_ApiTrace_RecordType_ThreadEvent_GraphMemcpyBegin_V1: return NvTrace_CUDA_ApiTrace_RecordType_GraphMemcpyBegin;
            case NvTrace_CUDA_ApiTrace_RecordType_ThreadEvent_GraphMemcpyEnd_V1: return NvTrace_CUDA_ApiTrace_RecordType_GraphMemcpyEnd;
            case NvTrace_CUDA_ApiTrace_RecordType_ThreadEvent_GraphMemsetBegin_V1: return NvTrace_CUDA_ApiTrace_RecordType_GraphMemsetBegin;
            case NvTrace_CUDA_ApiTrace_RecordType_ThreadEvent_GraphMemsetEnd_V1: return NvTrace_CUDA_ApiTrace_RecordType_GraphMemsetEnd;
            case NvTrace_CUDA_ApiTrace_RecordType_ThreadEvent_GraphHostCallbackLaunch_V1: return NvTrace_CUDA_ApiTrace_RecordType_GraphHostCallbackLaunch;
            case NvTrace_CUDA_ApiTrace_RecordType_ThreadEvent_GraphHostCallbackExecBegin_V1: return NvTrace_CUDA_ApiTrace_RecordType_GraphHostCallbackExecBegin;
            case NvTrace_CUDA_ApiTrace_RecordType_ThreadEvent_GraphHostCallbackExecEnd_V1: return NvTrace_CUDA_ApiTrace_RecordType_GraphHostCallbackExecEnd;
            case NvTrace_CUDA_ApiTrace_RecordType_ThreadEvent_MemOpBegin_V1: return NvTrace_CUDA_ApiTrace_RecordType_MemOpBegin;
            case NvTrace_CUDA_ApiTrace_RecordType_ThreadEvent_MemOpEnd_V1: return NvTrace_CUDA_ApiTrace_RecordType_MemOpEnd;
            default: return NvTrace_CUDA_ApiTrace_RecordType_INVALID;
        }
    }

    /// \brief A record of the retired ObjectLifetime band, translated into the V2 record vocabulary.
    ///
    /// recordType discriminates the payload: NvTrace_ReservedRecordType_RecordReference for a created
    /// object, whose state the reference names, and the matching
    /// NvTrace_CUDA_ApiTrace_RecordType_*Destroyed value for a destroyed one.
    struct CudaApiTraceTranslatedRecordV1
    {
        uint32_t recordType = 0;
        size_t payloadSize = 0;
        union
        {
            NvTrace_RecordReference reference;
            NvTrace_CUDA_ApiTrace_ContextDestroyed contextDestroyed;
            NvTrace_CUDA_ApiTrace_StreamDestroyed streamDestroyed;
            NvTrace_CUDA_ApiTrace_ModuleUnloaded moduleUnloaded;
            NvTrace_CUDA_ApiTrace_GraphExecDestroyed graphExecDestroyed;
        } payload;
    };

    /// \brief CUDA API Trace Observer over the V1 provider buffer format -- the old-driver shim.
    ///
    /// Reads buffers published by drivers that predate the generic provider-buffer format and serves
    /// them through the same typed reads CudaApiTraceObserver exposes, converting every V1 state
    /// struct to its V2 created-record on delivery. The V1 correlation-id schemes are kept: the old
    /// driver stamps them into QMDs, so the consumer must derive the same values.
    ///
    /// Synthesized composite records lay out as sizeof(the fixed segment) followed immediately by the
    /// variable segment, matching what a generic producer of this build writes -- so a shimmed record
    /// is indistinguishable from a generic-format one downstream.
    ///
    /// Thread safety: not internally synchronized; one consumer thread. The producer may be concurrent.
    class CudaApiTraceObserverV1
    {
    public:
        using FbConfigurationView = FlatBufferView<FB::Provider::CudaApiTrace::ProviderConfiguration>;

        struct ThreadEventRecordStorage
        {
            CudaApiTraceTranslatedRecordV1 translated;
            ThreadEventRecord record;
        };

        ~CudaApiTraceObserverV1() = default;
        CudaApiTraceObserverV1() = default;

        CudaApiTraceObserverV1(const CudaApiTraceObserverV1&) = delete;
        CudaApiTraceObserverV1(CudaApiTraceObserverV1&& rhs) noexcept
        {
            *this = std::move(rhs);
        }

        CudaApiTraceObserverV1& operator=(const CudaApiTraceObserverV1&) = delete;
        CudaApiTraceObserverV1& operator=(CudaApiTraceObserverV1&& rhs) noexcept
        {
            if (this != &rhs)
            {
                m_sections = std::exchange(rhs.m_sections, {});
                m_recordVersionInfo = std::exchange(rhs.m_recordVersionInfo, {});
                m_objectLifetimePayloadOffset = std::exchange(rhs.m_objectLifetimePayloadOffset, 0);
                m_graphNodeStateSize = std::exchange(rhs.m_graphNodeStateSize, 0);
                m_graphNodeParamsSize = std::exchange(rhs.m_graphNodeParamsSize, 0);
                m_graphNodeTableEntryStride = std::exchange(rhs.m_graphNodeTableEntryStride, 0);
                m_graphExecIndexBits = std::exchange(rhs.m_graphExecIndexBits, 0);
                m_graphNodeIndexBits = std::exchange(rhs.m_graphNodeIndexBits, 0);
                m_gpuTimestampFifoObservers = std::move(rhs.m_gpuTimestampFifoObservers);
                m_functionNameBuffer = std::move(rhs.m_functionNameBuffer);
                m_functionRecordBytesBuffer = std::move(rhs.m_functionRecordBytesBuffer);
                m_graphNodeRecordBytesBuffer = std::move(rhs.m_graphNodeRecordBytesBuffer);
            }
            return *this;
        }

        void Reset() noexcept
        {
            m_graphNodeRecordBytesBuffer.clear();
            m_functionRecordBytesBuffer.clear();
            m_functionNameBuffer.clear();
            m_gpuTimestampFifoObservers.clear();
            m_graphNodeIndexBits = 0;
            m_graphExecIndexBits = 0;
            m_graphNodeTableEntryStride = 0;
            m_graphNodeParamsSize = 0;
            m_graphNodeStateSize = 0;
            m_objectLifetimePayloadOffset = 0;
            m_recordVersionInfo = {};
            m_sections = {};
        }

        bool Initialize(const uint8_t* pMappedProviderBufferMemory, size_t providerBufferSize)
        {
            auto errorGuard = ScopeExitGuard([&]() {
                NVTRACE_LOG_ERR(20, "CudaApiTraceObserverV1::Initialize failed (provider buffer=%p providerBufferSize=%zu).", static_cast<const void*>(pMappedProviderBufferMemory), providerBufferSize);
                Reset();
            });

            // The buffer is shared memory under the profiled process's control: every offset, stride
            // and count must be proven inside the mapping before it shapes a pointer or an allocation.
            const auto fnRangeFits = [&](uint64_t itemsOffset, uint64_t itemSize, uint64_t itemCount) {
                return itemsOffset == 0 || (itemSize != 0 && itemsOffset <= providerBufferSize && itemCount <= (providerBufferSize - itemsOffset) / itemSize);
            };

            // Entry N of a section sits at itemsOffset + itemSize * N, so both must carry the entry's
            // alignment for the 64-bit loads the readers perform on every entry to be aligned.
            const auto fnRangeAligned = [](uint64_t itemsOffset, uint64_t itemSize, uint64_t itemAlignment) {
                return itemsOffset == 0 || ((itemsOffset % itemAlignment) == 0 && (itemSize % itemAlignment) == 0);
            };

            // A section whose entries do not span everything the observer reads out of them is served
            // as absent, exactly like one the layout omits.
            const auto fnReadableSectionOffset = [](uint64_t sectionOffset, uint32_t entrySize, uint32_t entryMinorVersion, uint32_t readMinorVersion, const char* pSectionName) -> uint64_t {
                if (sectionOffset == 0)
                {
                    NVTRACE_LOG_INF(20, "V1 provider buffer %s section is not present; %s data will not be available.", pSectionName, pSectionName);
                    return 0;
                }

                if (entryMinorVersion < readMinorVersion || entrySize < readMinorVersion)
                {
                    NVTRACE_LOG_INF(20, "V1 provider buffer %s entries (size %u minor version %u) are below the %u bytes the observer reads; %s data will not be available.", pSectionName, entrySize, entryMinorVersion, readMinorVersion, pSectionName);
                    return 0;
                }

                return sectionOffset;
            };

            NVTRACE_LOG_ERR_RETURN_IF(pMappedProviderBufferMemory == nullptr || providerBufferSize < sizeof(NvTrace_ProviderBufferHeader_V1), false, 20, "The mapped V1 provider buffer cannot hold its header (providerBufferSize %zu).", providerBufferSize);

            // Header/layout/sections are read in place; require the base to meet the strictest alignment.
            NVTRACE_LOG_ERR_RETURN_IF((reinterpret_cast<uintptr_t>(pMappedProviderBufferMemory) % alignof(NvTrace_ProviderBufferHeader_V1)) != 0, false, 20, "Provider buffer base must be %zu-byte aligned (pMappedProviderBufferMemory=%p).", alignof(NvTrace_ProviderBufferHeader_V1), static_cast<const void*>(pMappedProviderBufferMemory));

            const NvTrace_ProviderBufferHeader_V1* const pProviderBufferHeader = reinterpret_cast<const NvTrace_ProviderBufferHeader_V1*>(pMappedProviderBufferMemory);
            NVTRACE_LOG_ERR_RETURN_IF(pProviderBufferHeader->providerTypeUuid != NvTrace_CUDA_ApiTrace_ProviderTypeUuid_V1, false, 20, "Invalid provider type UUID for CUDA API Trace V1 observer (got %s, expected %s).", ToString(pProviderBufferHeader->providerTypeUuid).c_str(), ToString(NvTrace_CUDA_ApiTrace_ProviderTypeUuid_V1).c_str());
            NVTRACE_LOG_ERR_RETURN_IF(pProviderBufferHeader->minorVersion < NvTrace_ProviderBufferHeader_V1_MINOR_VERSION_A, false, 20, "Provider buffer header minor version %u is incompatible with the CUDA API Trace V1 observer (expected %u).", pProviderBufferHeader->minorVersion, NvTrace_ProviderBufferHeader_V1_MINOR_VERSION_A);
            NVTRACE_LOG_ERR_RETURN_IF(pProviderBufferHeader->providerBufferLayoutMinorVersion < NvTrace_CUDA_ApiTrace_ProviderBufferLayout_V1_MINOR_VERSION_A, false, 20, "Provider buffer layout minor version %u is incompatible with the CUDA API Trace V1 observer (expected %u).", pProviderBufferHeader->providerBufferLayoutMinorVersion, NvTrace_CUDA_ApiTrace_ProviderBufferLayout_V1_MINOR_VERSION_A);

            const uint64_t providerBufferLayoutOffset = pProviderBufferHeader->providerBufferLayoutOffset;
            NVTRACE_LOG_ERR_RETURN_IF(providerBufferLayoutOffset == 0, false, 20, "The V1 provider buffer header names no provider buffer layout.");
            NVTRACE_LOG_ERR_RETURN_IF(!fnRangeAligned(providerBufferLayoutOffset, sizeof(NvTrace_CUDA_ApiTrace_ProviderBufferLayout_V1), alignof(NvTrace_CUDA_ApiTrace_ProviderBufferLayout_V1)), false, 20, "The provider buffer layout is misaligned (offset %" PRIu64 " alignment %zu).", providerBufferLayoutOffset, alignof(NvTrace_CUDA_ApiTrace_ProviderBufferLayout_V1));
            NVTRACE_LOG_ERR_RETURN_IF(!fnRangeFits(providerBufferLayoutOffset, sizeof(NvTrace_CUDA_ApiTrace_ProviderBufferLayout_V1), 1), false, 20, "The provider buffer layout lies outside the mapping (offset %" PRIu64 ").", providerBufferLayoutOffset);

            m_sections.pProviderBufferHeader = pProviderBufferHeader;
            AssignSectionPointer(pMappedProviderBufferMemory, providerBufferLayoutOffset, m_sections.pProviderBufferLayout);

            const NvTrace_CUDA_ApiTrace_ProviderBufferLayout_V1& providerBufferLayout = *m_sections.pProviderBufferLayout;
            NVTRACE_LOG_ERR_RETURN_IF(providerBufferLayout.eventVersionInfoMinorVersion < NvTrace_CUDA_ApiTrace_EventVersionInfo_V1_MINOR_VERSION_A, false, 20, "Event version info minor version %u is incompatible with the CUDA API Trace V1 observer (expected %u).", providerBufferLayout.eventVersionInfoMinorVersion, NvTrace_CUDA_ApiTrace_EventVersionInfo_V1_MINOR_VERSION_A);

            const uint64_t eventVersionInfoOffset = providerBufferLayout.eventVersionInfoOffset;
            NVTRACE_LOG_ERR_RETURN_IF(eventVersionInfoOffset == 0, false, 20, "The V1 provider buffer layout names no event version info.");
            NVTRACE_LOG_ERR_RETURN_IF(!fnRangeAligned(eventVersionInfoOffset, sizeof(NvTrace_CUDA_ApiTrace_EventVersionInfo_V1), alignof(NvTrace_CUDA_ApiTrace_EventVersionInfo_V1)), false, 20, "The event version info is misaligned (offset %" PRIu64 " alignment %zu).", eventVersionInfoOffset, alignof(NvTrace_CUDA_ApiTrace_EventVersionInfo_V1));
            NVTRACE_LOG_ERR_RETURN_IF(!fnRangeFits(eventVersionInfoOffset, sizeof(NvTrace_CUDA_ApiTrace_EventVersionInfo_V1), 1), false, 20, "The event version info lies outside the mapping (offset %" PRIu64 ").", eventVersionInfoOffset);

            AssignSectionPointer(pMappedProviderBufferMemory, eventVersionInfoOffset, m_sections.pEventVersionInfo);

            const NvTrace_CUDA_ApiTrace_EventVersionInfo_V1& eventVersionInfo = *m_sections.pEventVersionInfo;
            NVTRACE_LOG_ERR_RETURN_IF(eventVersionInfo.objectLifetimeEventHeaderMinorVersion < NvTrace_CUDA_ApiTrace_ObjectLifetime_EventHeader_V1_MINOR_VERSION_A, false, 20, "Object lifetime event header minor version %u is incompatible with the CUDA API Trace V1 observer (expected %u).", eventVersionInfo.objectLifetimeEventHeaderMinorVersion, NvTrace_CUDA_ApiTrace_ObjectLifetime_EventHeader_V1_MINOR_VERSION_A);
            // The payload arm begins at the published header size, which must cover the published
            // minor version and keep the arm's 64-bit fields aligned.
            NVTRACE_LOG_ERR_RETURN_IF(eventVersionInfo.objectLifetimeEventHeaderSize < eventVersionInfo.objectLifetimeEventHeaderMinorVersion || (eventVersionInfo.objectLifetimeEventHeaderSize % sizeof(uint64_t)) != 0, false, 20, "Object lifetime event header size cannot lead a composite record (size=%u minorVersion=%u).", eventVersionInfo.objectLifetimeEventHeaderSize, eventVersionInfo.objectLifetimeEventHeaderMinorVersion);

            const uint64_t deviceStatesOffset = fnReadableSectionOffset(providerBufferLayout.deviceStatesOffset, eventVersionInfo.deviceStateSize, eventVersionInfo.deviceStateMinorVersion, NvTrace_CUDA_ApiTrace_DeviceState_V1_MINOR_VERSION_A, "device state");
            const uint64_t contextStatesOffset = fnReadableSectionOffset(providerBufferLayout.contextStatesOffset, eventVersionInfo.contextStateSize, eventVersionInfo.contextStateMinorVersion, NvTrace_CUDA_ApiTrace_ContextState_V1_MINOR_VERSION_A, "context state");
            const uint64_t streamStatesOffset = fnReadableSectionOffset(providerBufferLayout.streamStatesOffset, eventVersionInfo.streamStateSize, eventVersionInfo.streamStateMinorVersion, NvTrace_CUDA_ApiTrace_StreamState_V1_MINOR_VERSION_A, "stream state");
            const uint64_t moduleStatesOffset = fnReadableSectionOffset(providerBufferLayout.moduleStatesOffset, eventVersionInfo.moduleStateSize, eventVersionInfo.moduleStateMinorVersion, NvTrace_CUDA_ApiTrace_ModuleState_V1_MINOR_VERSION_A, "module state");
            const uint64_t functionStatesOffset = fnReadableSectionOffset(providerBufferLayout.functionStatesOffset, eventVersionInfo.functionStateSize, eventVersionInfo.functionStateMinorVersion, NvTrace_CUDA_ApiTrace_FunctionState_V1_MINOR_VERSION_A, "function state");
            const uint64_t graphExecStatesOffset = fnReadableSectionOffset(providerBufferLayout.graphExecStatesOffset, eventVersionInfo.graphExecStateSize, eventVersionInfo.graphExecStateMinorVersion, NvTrace_CUDA_ApiTrace_GraphExecState_V1_MINOR_VERSION_A, "graph exec state");
            const uint64_t channelStatesOffset = fnReadableSectionOffset(providerBufferLayout.channelStatesOffset, eventVersionInfo.channelStateSize, eventVersionInfo.channelStateMinorVersion, NvTrace_CUDA_ApiTrace_ChannelState_V1_MINOR_VERSION_A, "channel state");

            const uint64_t graphNodeStatesSectionOffset = fnReadableSectionOffset(providerBufferLayout.graphNodeStatesOffset, eventVersionInfo.graphNodeStateSize, eventVersionInfo.graphNodeStateMinorVersion, NvTrace_CUDA_ApiTrace_GraphNodeState_V1_MINOR_VERSION_A, "graph node state");
            const GraphNodeTableLayout graphNodeTableLayout = graphNodeStatesSectionOffset == 0 ? GraphNodeTableLayout{} : ResolveGraphNodeTableLayout(eventVersionInfo);
            const uint64_t graphNodeStatesOffset = graphNodeTableLayout.entryStride == 0 ? 0 : graphNodeStatesSectionOffset;

            const uint64_t stringTableEntriesSectionOffset = fnReadableSectionOffset(providerBufferLayout.stringTableEntriesOffset, eventVersionInfo.stringTableEntrySize, eventVersionInfo.stringTableEntryMinorVersion, NvTrace_CUDA_ApiTrace_StringTableEntry_V1_MINOR_VERSION_A, "string table entry");
            // An entry names a substring of the data section, so neither section reads without the other.
            const bool stringTableReadable = stringTableEntriesSectionOffset != 0 && providerBufferLayout.stringTableDataOffset != 0;
            const uint64_t stringTableEntriesOffset = stringTableReadable ? stringTableEntriesSectionOffset : 0;
            const uint64_t stringTableDataOffset = stringTableReadable ? providerBufferLayout.stringTableDataOffset : 0;

            const uint64_t currentStateTableStatsSectionOffset = fnReadableSectionOffset(providerBufferLayout.currentStateTableStatsOffset, eventVersionInfo.stateTableStatsSize, eventVersionInfo.stateTableStatsMinorVersion, NvTrace_CUDA_ApiTrace_StateTableStats_V1_MINOR_VERSION_A, "current state table stats");
            const uint64_t maxStateTableStatsSectionOffset = fnReadableSectionOffset(providerBufferLayout.maxStateTableStatsOffset, eventVersionInfo.stateTableStatsSize, eventVersionInfo.stateTableStatsMinorVersion, NvTrace_CUDA_ApiTrace_StateTableStats_V1_MINOR_VERSION_A, "max state table stats");
            // The counts and their peaks fill one statistics snapshot, so the two sections are read as one.
            const bool stateTableStatsReadable = currentStateTableStatsSectionOffset != 0 && maxStateTableStatsSectionOffset != 0;
            const uint64_t currentStateTableStatsOffset = stateTableStatsReadable ? currentStateTableStatsSectionOffset : 0;
            const uint64_t maxStateTableStatsOffset = stateTableStatsReadable ? maxStateTableStatsSectionOffset : 0;

            const uint64_t gpuTimestampFifoStatesSectionOffset = fnReadableSectionOffset(providerBufferLayout.gpuTimestampFifoCircularBufferStateOffset, providerBufferLayout.gpuTimestampFifoCircularBufferStateSize, providerBufferLayout.gpuTimestampFifoCircularBufferStateMinorVersion, NvTrace_CircularBufferState_MINOR_VERSION_A, "GPU timestamp FIFO state");
            const uint64_t gpuTimestampFifoCellsSectionOffset = fnReadableSectionOffset(providerBufferLayout.gpuTimestampFifoCellsOffset, providerBufferLayout.gpuTimestampFifoCellSize, providerBufferLayout.gpuTimestampFifoCellMinorVersion, NvTrace_CUDA_ApiTrace_GpuTimestampFifoCell_V1_MINOR_VERSION_A, "GPU timestamp FIFO cell");
            // A FIFO cell names a chunk of reports and is drained through the FIFO's own circular
            // buffer state, so the three sections are read as one and disabled as one.
            const bool gpuTimestampsReadable = gpuTimestampFifoStatesSectionOffset != 0 && gpuTimestampFifoCellsSectionOffset != 0 && providerBufferLayout.gpuTimestampChunksOffset != 0;
            const uint64_t gpuTimestampFifoStatesOffset = gpuTimestampsReadable ? gpuTimestampFifoStatesSectionOffset : 0;
            const uint64_t gpuTimestampFifoCellsOffset = gpuTimestampsReadable ? gpuTimestampFifoCellsSectionOffset : 0;
            const uint64_t gpuTimestampReportsOffset = gpuTimestampsReadable ? providerBufferLayout.gpuTimestampChunksOffset : 0;

            // The FIFO reader divides the sequence number by the cell count, and the chunk walk steps
            // over the reports of a chunk; neither geometry may be empty.
            NVTRACE_LOG_ERR_RETURN_IF(gpuTimestampsReadable && providerBufferLayout.gpuTimestampFifoCellsPerStream == 0, false, 20, "The V1 GPU timestamp FIFOs hold no cells per stream.");
            NVTRACE_LOG_ERR_RETURN_IF(gpuTimestampsReadable && providerBufferLayout.gpuTimestampReportsPerChunk == 0, false, 20, "The V1 GPU timestamp chunks hold no reports per chunk.");

            NVTRACE_LOG_ERR_RETURN_IF((providerBufferLayout.configurationOffset % alignof(uint64_t)) != 0, false, 20, "The provider configuration flatbuffer is misaligned (offset %" PRIu64 " alignment %zu).", providerBufferLayout.configurationOffset, alignof(uint64_t));
            NVTRACE_LOG_ERR_RETURN_IF(!fnRangeAligned(deviceStatesOffset, eventVersionInfo.deviceStateSize, alignof(NvTrace_CUDA_ApiTrace_DeviceState_V1)), false, 20, "The device states are misaligned (offset %" PRIu64 " size %u alignment %zu).", deviceStatesOffset, eventVersionInfo.deviceStateSize, alignof(NvTrace_CUDA_ApiTrace_DeviceState_V1));
            NVTRACE_LOG_ERR_RETURN_IF(!fnRangeAligned(contextStatesOffset, eventVersionInfo.contextStateSize, alignof(NvTrace_CUDA_ApiTrace_ContextState_V1)), false, 20, "The context states are misaligned (offset %" PRIu64 " size %u alignment %zu).", contextStatesOffset, eventVersionInfo.contextStateSize, alignof(NvTrace_CUDA_ApiTrace_ContextState_V1));
            NVTRACE_LOG_ERR_RETURN_IF(!fnRangeAligned(streamStatesOffset, eventVersionInfo.streamStateSize, alignof(NvTrace_CUDA_ApiTrace_StreamState_V1)), false, 20, "The stream states are misaligned (offset %" PRIu64 " size %u alignment %zu).", streamStatesOffset, eventVersionInfo.streamStateSize, alignof(NvTrace_CUDA_ApiTrace_StreamState_V1));
            NVTRACE_LOG_ERR_RETURN_IF(!fnRangeAligned(moduleStatesOffset, eventVersionInfo.moduleStateSize, alignof(NvTrace_CUDA_ApiTrace_ModuleState_V1)), false, 20, "The module states are misaligned (offset %" PRIu64 " size %u alignment %zu).", moduleStatesOffset, eventVersionInfo.moduleStateSize, alignof(NvTrace_CUDA_ApiTrace_ModuleState_V1));
            NVTRACE_LOG_ERR_RETURN_IF(!fnRangeAligned(functionStatesOffset, eventVersionInfo.functionStateSize, alignof(NvTrace_CUDA_ApiTrace_FunctionState_V1)), false, 20, "The function states are misaligned (offset %" PRIu64 " size %u alignment %zu).", functionStatesOffset, eventVersionInfo.functionStateSize, alignof(NvTrace_CUDA_ApiTrace_FunctionState_V1));
            NVTRACE_LOG_ERR_RETURN_IF(!fnRangeAligned(graphExecStatesOffset, eventVersionInfo.graphExecStateSize, alignof(NvTrace_CUDA_ApiTrace_GraphExecState_V1)), false, 20, "The graph exec states are misaligned (offset %" PRIu64 " size %u alignment %zu).", graphExecStatesOffset, eventVersionInfo.graphExecStateSize, alignof(NvTrace_CUDA_ApiTrace_GraphExecState_V1));
            NVTRACE_LOG_ERR_RETURN_IF(!fnRangeAligned(graphNodeStatesOffset, graphNodeTableLayout.entryStride, alignof(NvTrace_CUDA_ApiTrace_GraphNodeState_V1)), false, 20, "The graph node states are misaligned (offset %" PRIu64 " stride %" PRIu64 " alignment %zu).", graphNodeStatesOffset, graphNodeTableLayout.entryStride, alignof(NvTrace_CUDA_ApiTrace_GraphNodeState_V1));
            NVTRACE_LOG_ERR_RETURN_IF(!fnRangeAligned(channelStatesOffset, eventVersionInfo.channelStateSize, alignof(NvTrace_CUDA_ApiTrace_ChannelState_V1)), false, 20, "The channel states are misaligned (offset %" PRIu64 " size %u alignment %zu).", channelStatesOffset, eventVersionInfo.channelStateSize, alignof(NvTrace_CUDA_ApiTrace_ChannelState_V1));
            NVTRACE_LOG_ERR_RETURN_IF(!fnRangeAligned(stringTableEntriesOffset, eventVersionInfo.stringTableEntrySize, alignof(NvTrace_CUDA_ApiTrace_StringTableEntry_V1)), false, 20, "The string table entries are misaligned (offset %" PRIu64 " size %u alignment %zu).", stringTableEntriesOffset, eventVersionInfo.stringTableEntrySize, alignof(NvTrace_CUDA_ApiTrace_StringTableEntry_V1));
            NVTRACE_LOG_ERR_RETURN_IF(!fnRangeAligned(currentStateTableStatsOffset, eventVersionInfo.stateTableStatsSize, alignof(NvTrace_CUDA_ApiTrace_StateTableStats_V1)), false, 20, "The current state table stats are misaligned (offset %" PRIu64 " size %u alignment %zu).", currentStateTableStatsOffset, eventVersionInfo.stateTableStatsSize, alignof(NvTrace_CUDA_ApiTrace_StateTableStats_V1));
            NVTRACE_LOG_ERR_RETURN_IF(!fnRangeAligned(maxStateTableStatsOffset, eventVersionInfo.stateTableStatsSize, alignof(NvTrace_CUDA_ApiTrace_StateTableStats_V1)), false, 20, "The max state table stats are misaligned (offset %" PRIu64 " size %u alignment %zu).", maxStateTableStatsOffset, eventVersionInfo.stateTableStatsSize, alignof(NvTrace_CUDA_ApiTrace_StateTableStats_V1));
            NVTRACE_LOG_ERR_RETURN_IF(!fnRangeAligned(gpuTimestampFifoStatesOffset, providerBufferLayout.gpuTimestampFifoCircularBufferStateSize, alignof(NvTrace_CircularBufferState)), false, 20, "The GPU timestamp FIFO states are misaligned (offset %" PRIu64 " size %u alignment %zu).", gpuTimestampFifoStatesOffset, providerBufferLayout.gpuTimestampFifoCircularBufferStateSize, alignof(NvTrace_CircularBufferState));
            NVTRACE_LOG_ERR_RETURN_IF(!fnRangeAligned(gpuTimestampFifoCellsOffset, providerBufferLayout.gpuTimestampFifoCellSize, alignof(NvTrace_CUDA_ApiTrace_GpuTimestampFifoCell_V1)), false, 20, "The GPU timestamp FIFO cells are misaligned (offset %" PRIu64 " size %u alignment %zu).", gpuTimestampFifoCellsOffset, providerBufferLayout.gpuTimestampFifoCellSize, alignof(NvTrace_CUDA_ApiTrace_GpuTimestampFifoCell_V1));
            NVTRACE_LOG_ERR_RETURN_IF(!fnRangeAligned(gpuTimestampReportsOffset, sizeof(NvTrace_CUDA_ApiTrace_GpuTimestampReport_V1), alignof(NvTrace_CUDA_ApiTrace_GpuTimestampReport_V1)), false, 20, "The GPU timestamp reports are misaligned (offset %" PRIu64 " alignment %zu).", gpuTimestampReportsOffset, alignof(NvTrace_CUDA_ApiTrace_GpuTimestampReport_V1));

            NVTRACE_LOG_ERR_RETURN_IF(!fnRangeFits(providerBufferLayout.configurationOffset, 1, providerBufferLayout.configurationSize), false, 20, "The provider configuration lies outside the mapping (offset %" PRIu64 " size %u).", providerBufferLayout.configurationOffset, providerBufferLayout.configurationSize);
            // The flatbuffer root offset is read out of the blob's first word before anything else.
            NVTRACE_LOG_ERR_RETURN_IF(providerBufferLayout.configurationOffset != 0 && providerBufferLayout.configurationSize < sizeof(flatbuffers::uoffset_t), false, 20, "The provider configuration is too small to hold a flatbuffer root (size %u, expected %zu at least).", providerBufferLayout.configurationSize, sizeof(flatbuffers::uoffset_t));
            NVTRACE_LOG_ERR_RETURN_IF(!fnRangeFits(deviceStatesOffset, eventVersionInfo.deviceStateSize, providerBufferLayout.deviceStatesCount), false, 20, "The device states lie outside the mapping (offset %" PRIu64 " size %u count %u).", deviceStatesOffset, eventVersionInfo.deviceStateSize, providerBufferLayout.deviceStatesCount);
            NVTRACE_LOG_ERR_RETURN_IF(!fnRangeFits(contextStatesOffset, eventVersionInfo.contextStateSize, providerBufferLayout.contextStatesCount), false, 20, "The context states lie outside the mapping (offset %" PRIu64 " size %u count %u).", contextStatesOffset, eventVersionInfo.contextStateSize, providerBufferLayout.contextStatesCount);
            NVTRACE_LOG_ERR_RETURN_IF(!fnRangeFits(streamStatesOffset, eventVersionInfo.streamStateSize, providerBufferLayout.streamStatesCount), false, 20, "The stream states lie outside the mapping (offset %" PRIu64 " size %u count %u).", streamStatesOffset, eventVersionInfo.streamStateSize, providerBufferLayout.streamStatesCount);
            NVTRACE_LOG_ERR_RETURN_IF(!fnRangeFits(moduleStatesOffset, eventVersionInfo.moduleStateSize, providerBufferLayout.moduleStatesCount), false, 20, "The module states lie outside the mapping (offset %" PRIu64 " size %u count %u).", moduleStatesOffset, eventVersionInfo.moduleStateSize, providerBufferLayout.moduleStatesCount);
            NVTRACE_LOG_ERR_RETURN_IF(!fnRangeFits(functionStatesOffset, eventVersionInfo.functionStateSize, providerBufferLayout.functionStatesCount), false, 20, "The function states lie outside the mapping (offset %" PRIu64 " size %u count %u).", functionStatesOffset, eventVersionInfo.functionStateSize, providerBufferLayout.functionStatesCount);
            NVTRACE_LOG_ERR_RETURN_IF(!fnRangeFits(graphExecStatesOffset, eventVersionInfo.graphExecStateSize, providerBufferLayout.graphExecStatesCount), false, 20, "The graph exec states lie outside the mapping (offset %" PRIu64 " size %u count %u).", graphExecStatesOffset, eventVersionInfo.graphExecStateSize, providerBufferLayout.graphExecStatesCount);
            NVTRACE_LOG_ERR_RETURN_IF(!fnRangeFits(graphNodeStatesOffset, graphNodeTableLayout.entryStride, providerBufferLayout.graphNodeStatesCount), false, 20, "The graph node states lie outside the mapping (offset %" PRIu64 " stride %" PRIu64 " count %u).", graphNodeStatesOffset, graphNodeTableLayout.entryStride, providerBufferLayout.graphNodeStatesCount);
            NVTRACE_LOG_ERR_RETURN_IF(!fnRangeFits(channelStatesOffset, eventVersionInfo.channelStateSize, providerBufferLayout.channelStatesCount), false, 20, "The channel states lie outside the mapping (offset %" PRIu64 " size %u count %u).", channelStatesOffset, eventVersionInfo.channelStateSize, providerBufferLayout.channelStatesCount);
            NVTRACE_LOG_ERR_RETURN_IF(!fnRangeFits(stringTableEntriesOffset, eventVersionInfo.stringTableEntrySize, providerBufferLayout.stringTableEntriesCount), false, 20, "The string table entries lie outside the mapping (offset %" PRIu64 " size %u count %u).", stringTableEntriesOffset, eventVersionInfo.stringTableEntrySize, providerBufferLayout.stringTableEntriesCount);
            NVTRACE_LOG_ERR_RETURN_IF(!fnRangeFits(stringTableDataOffset, 1, providerBufferLayout.stringTableDataSize), false, 20, "The string table data lies outside the mapping (offset %" PRIu64 " size %u).", stringTableDataOffset, providerBufferLayout.stringTableDataSize);
            NVTRACE_LOG_ERR_RETURN_IF(!fnRangeFits(currentStateTableStatsOffset, eventVersionInfo.stateTableStatsSize, 1), false, 20, "The current state table stats lie outside the mapping (offset %" PRIu64 " size %u).", currentStateTableStatsOffset, eventVersionInfo.stateTableStatsSize);
            NVTRACE_LOG_ERR_RETURN_IF(!fnRangeFits(maxStateTableStatsOffset, eventVersionInfo.stateTableStatsSize, 1), false, 20, "The max state table stats lie outside the mapping (offset %" PRIu64 " size %u).", maxStateTableStatsOffset, eventVersionInfo.stateTableStatsSize);
            NVTRACE_LOG_ERR_RETURN_IF(!fnRangeFits(gpuTimestampFifoStatesOffset, providerBufferLayout.gpuTimestampFifoCircularBufferStateSize, providerBufferLayout.streamStatesCount), false, 20, "The GPU timestamp FIFO states lie outside the mapping (offset %" PRIu64 " size %u count %u).", gpuTimestampFifoStatesOffset, providerBufferLayout.gpuTimestampFifoCircularBufferStateSize, providerBufferLayout.streamStatesCount);
            NVTRACE_LOG_ERR_RETURN_IF(!fnRangeFits(gpuTimestampFifoCellsOffset, providerBufferLayout.gpuTimestampFifoCellSize, static_cast<uint64_t>(providerBufferLayout.streamStatesCount) * providerBufferLayout.gpuTimestampFifoCellsPerStream), false, 20, "The GPU timestamp FIFO cells lie outside the mapping (offset %" PRIu64 " size %u count %u x %u).", gpuTimestampFifoCellsOffset, providerBufferLayout.gpuTimestampFifoCellSize, providerBufferLayout.streamStatesCount, providerBufferLayout.gpuTimestampFifoCellsPerStream);
            NVTRACE_LOG_ERR_RETURN_IF(!fnRangeFits(gpuTimestampReportsOffset, sizeof(NvTrace_CUDA_ApiTrace_GpuTimestampReport_V1), static_cast<uint64_t>(providerBufferLayout.gpuTimestampChunksCount) * providerBufferLayout.gpuTimestampReportsPerChunk), false, 20, "The GPU timestamp reports lie outside the mapping (offset %" PRIu64 " count %u x %u).", gpuTimestampReportsOffset, providerBufferLayout.gpuTimestampChunksCount, providerBufferLayout.gpuTimestampReportsPerChunk);

            AssignFlatbufferSectionPointer(pMappedProviderBufferMemory, providerBufferLayout.configurationOffset, m_sections.fbProviderConfiguration);
            AssignSectionPointer(pMappedProviderBufferMemory, deviceStatesOffset, m_sections.pDeviceStates);
            AssignSectionPointer(pMappedProviderBufferMemory, contextStatesOffset, m_sections.pContextStates);
            AssignSectionPointer(pMappedProviderBufferMemory, streamStatesOffset, m_sections.pStreamStates);
            AssignSectionPointer(pMappedProviderBufferMemory, moduleStatesOffset, m_sections.pModuleStates);
            AssignSectionPointer(pMappedProviderBufferMemory, functionStatesOffset, m_sections.pFunctionStates);
            AssignSectionPointer(pMappedProviderBufferMemory, graphExecStatesOffset, m_sections.pGraphExecStates);
            AssignSectionPointer(pMappedProviderBufferMemory, graphNodeStatesOffset, m_sections.pGraphNodeStates);
            AssignSectionPointer(pMappedProviderBufferMemory, channelStatesOffset, m_sections.pChannelStates);
            AssignSectionPointer(pMappedProviderBufferMemory, stringTableEntriesOffset, m_sections.pStringTableEntries);
            AssignSectionPointer(pMappedProviderBufferMemory, stringTableDataOffset, m_sections.pStringTableData);
            AssignSectionPointer(pMappedProviderBufferMemory, currentStateTableStatsOffset, m_sections.pCurrentStateTableStats);
            AssignSectionPointer(pMappedProviderBufferMemory, maxStateTableStatsOffset, m_sections.pMaxStateTableStats);
            AssignSectionPointer(pMappedProviderBufferMemory, gpuTimestampFifoStatesOffset, m_sections.pGpuTimestampFifoCircularBufferStates);
            AssignSectionPointer(pMappedProviderBufferMemory, gpuTimestampFifoCellsOffset, m_sections.pGpuTimestampFifoCells);
            AssignSectionPointer(pMappedProviderBufferMemory, gpuTimestampReportsOffset, m_sections.pGpuTimestampReports);

            m_recordVersionInfo = MakeRecordVersionInfoFromV1(eventVersionInfo);
            m_objectLifetimePayloadOffset = eventVersionInfo.objectLifetimeEventHeaderSize;
            m_graphNodeStateSize = graphNodeTableLayout.stateSize;
            m_graphNodeParamsSize = graphNodeTableLayout.paramsSize;
            m_graphNodeTableEntryStride = graphNodeTableLayout.entryStride;
            m_graphExecIndexBits = CeilLog2(providerBufferLayout.graphExecStatesCount);
            m_graphNodeIndexBits = CeilLog2(providerBufferLayout.graphNodeStatesCount);

            if (gpuTimestampsReadable)
            {
                m_gpuTimestampFifoObservers.resize(providerBufferLayout.streamStatesCount);
                for (size_t streamIndex = 0; streamIndex < providerBufferLayout.streamStatesCount; ++streamIndex)
                {
                    const auto* const pCircularBufferState = reinterpret_cast<const NvTrace_CircularBufferState*>(reinterpret_cast<const uint8_t*>(m_sections.pGpuTimestampFifoCircularBufferStates) + providerBufferLayout.gpuTimestampFifoCircularBufferStateSize * streamIndex);
                    const auto* const pFifoCells = reinterpret_cast<const uint8_t*>(m_sections.pGpuTimestampFifoCells) + static_cast<size_t>(providerBufferLayout.gpuTimestampFifoCellSize) * providerBufferLayout.gpuTimestampFifoCellsPerStream * streamIndex;

                    m_gpuTimestampFifoObservers[streamIndex].Initialize(
                        pCircularBufferState,
                        pFifoCells,
                        providerBufferLayout.gpuTimestampFifoCellSize,
                        providerBufferLayout.gpuTimestampFifoCellsPerStream,
                        reinterpret_cast<const uint8_t*>(m_sections.pGpuTimestampReports),
                        providerBufferLayout.gpuTimestampChunksCount,
                        providerBufferLayout.gpuTimestampReportsPerChunk);
                }
            }

            errorGuard.Dismiss();
            NVTRACE_LOG_INF(90, "Initialized successfully over a V1 provider buffer.");
            return true;
        }

        bool IsInitialized() const
        {
            return m_sections.pProviderBufferHeader != nullptr;
        }

        bool IsProviderMeasuring() const
        {
            return IsInitialized() && m_sections.pProviderBufferHeader->providerStatus == NvTrace_ProviderStatus_measuring;
        }

        const NvTrace_CUDA_ApiTrace_RecordVersionInfo& RecordVersionInfo() const
        {
            return m_recordVersionInfo;
        }

        FbConfigurationView GetProviderConfigurationFlatBufferView() const
        {
            if (!m_sections.fbProviderConfiguration)
            {
                return {};
            }

            // Whole words, as a generic data blob is sized; the V1 layout aligns the next section, so
            // the pad is the zeroed gap behind the blob.
            return {reinterpret_cast<const uint8_t*>(m_sections.pProviderBufferHeader) + m_sections.pProviderBufferLayout->configurationOffset, NVTRACE_ALIGN_PAYLOAD_SIZE_DIV8(m_sections.pProviderBufferLayout->configurationSize) * sizeof(uint64_t)};
        }

        const FB::Provider::CudaApiTrace::ProviderConfiguration* GetProviderConfiguration() const
        {
            return m_sections.fbProviderConfiguration;
        }

        bool ReadProviderStatistics(NvTrace_CUDA_ApiTrace_ProviderStatistics& out_providerStatistics) const
        {
            if (!m_sections.pCurrentStateTableStats)
            {
                return false;
            }

            const NvTrace_CUDA_ApiTrace_StateTableStats_V1& currentStats = *m_sections.pCurrentStateTableStats;
            const NvTrace_CUDA_ApiTrace_StateTableStats_V1& maxStats = *m_sections.pMaxStateTableStats;

            out_providerStatistics.contextCount = currentStats.contextCount;
            out_providerStatistics.streamCount = currentStats.streamCount;
            out_providerStatistics.moduleCount = currentStats.moduleCount;
            out_providerStatistics.functionCount = currentStats.functionCount;
            out_providerStatistics.graphExecCount = currentStats.graphExecCount;
            out_providerStatistics.graphNodeCount = currentStats.graphNodeCount;
            out_providerStatistics.channelCount = currentStats.channelCount;

            out_providerStatistics.peakContextCount = maxStats.contextCount;
            out_providerStatistics.peakStreamCount = maxStats.streamCount;
            out_providerStatistics.peakModuleCount = maxStats.moduleCount;
            out_providerStatistics.peakFunctionCount = maxStats.functionCount;
            out_providerStatistics.peakGraphExecCount = maxStats.graphExecCount;
            out_providerStatistics.peakGraphNodeCount = maxStats.graphNodeCount;
            out_providerStatistics.peakChannelCount = maxStats.channelCount;

            out_providerStatistics.droppedFunctionCount = 0;
            out_providerStatistics.droppedFunctionNameCount = 0;
            out_providerStatistics.wedgedStreamCount = 0;
            return true;
        }

        uint32_t GetDeviceStateCount() const
        {
            return m_sections.pDeviceStates ? m_sections.pProviderBufferLayout->deviceStatesCount : 0;
        }

        uint32_t GetContextStateCount() const
        {
            return m_sections.pContextStates ? m_sections.pProviderBufferLayout->contextStatesCount : 0;
        }

        uint32_t GetStreamStateCount() const
        {
            return m_sections.pStreamStates ? m_sections.pProviderBufferLayout->streamStatesCount : 0;
        }

        uint32_t GetModuleStateCount() const
        {
            return m_sections.pModuleStates ? m_sections.pProviderBufferLayout->moduleStatesCount : 0;
        }

        uint32_t GetGraphExecStateCount() const
        {
            return m_sections.pGraphExecStates ? m_sections.pProviderBufferLayout->graphExecStatesCount : 0;
        }

        uint32_t GetGraphNodeStateCount() const
        {
            return m_sections.pGraphNodeStates ? m_sections.pProviderBufferLayout->graphNodeStatesCount : 0;
        }

        uint32_t GetChannelStateCount() const
        {
            return m_sections.pChannelStates ? m_sections.pProviderBufferLayout->channelStatesCount : 0;
        }

        // Devices are static after initialization: no revision protects them and none is needed, so a
        // device read always commits.
        template <typename TFN_OnRecord, typename TFN_CommitRecord>
        void ReadDeviceState(uint32_t entryIndex, TFN_OnRecord&& fnOnRecord, TFN_CommitRecord&& fnCommitRecord)
        {
            if (!m_sections.pDeviceStates || m_sections.pProviderBufferLayout->deviceStatesCount <= entryIndex)
            {
                return;
            }

            const auto& deviceState = *reinterpret_cast<const NvTrace_CUDA_ApiTrace_DeviceState_V1*>(reinterpret_cast<const uint8_t*>(m_sections.pDeviceStates) + m_sections.pEventVersionInfo->deviceStateSize * entryIndex);

            NvTrace_CUDA_ApiTrace_DeviceCreated record = {};
            record.gpuUuid = deviceState.gpuUuid;
            record.cudaDeviceOrdinal = deviceState.cudaDeviceOrdinal;
            record.cudeviceHandle = deviceState.cudeviceHandle;

            fnOnRecord(entryIndex, uint64_t{1}, record);
            fnCommitRecord(true);
        }

        template <typename TFN_OnRecord, typename TFN_CommitRecord>
        void ReadContextState(uint32_t entryIndex, TFN_OnRecord&& fnOnRecord, TFN_CommitRecord&& fnCommitRecord)
        {
            ReadConvertedState<NvTrace_CUDA_ApiTrace_ContextState_V1, NvTrace_CUDA_ApiTrace_ContextCreated>(m_sections.pContextStates, m_sections.pEventVersionInfo->contextStateSize, GetContextStateCount(), entryIndex, ConvertContextState, fnOnRecord, fnCommitRecord);
        }

        template <typename TFN_OnRecord, typename TFN_CommitRecord>
        void ReadStreamState(uint32_t entryIndex, TFN_OnRecord&& fnOnRecord, TFN_CommitRecord&& fnCommitRecord)
        {
            ReadConvertedState<NvTrace_CUDA_ApiTrace_StreamState_V1, NvTrace_CUDA_ApiTrace_StreamCreated>(m_sections.pStreamStates, m_sections.pEventVersionInfo->streamStateSize, GetStreamStateCount(), entryIndex, ConvertStreamState, fnOnRecord, fnCommitRecord);
        }

        template <typename TFN_OnRecord, typename TFN_CommitRecord>
        void ReadModuleState(uint32_t entryIndex, TFN_OnRecord&& fnOnRecord, TFN_CommitRecord&& fnCommitRecord)
        {
            ReadConvertedState<NvTrace_CUDA_ApiTrace_ModuleState_V1, NvTrace_CUDA_ApiTrace_ModuleLoaded>(m_sections.pModuleStates, m_sections.pEventVersionInfo->moduleStateSize, GetModuleStateCount(), entryIndex, ConvertModuleState, fnOnRecord, fnCommitRecord);
        }

        template <typename TFN_OnRecord, typename TFN_CommitRecord>
        void ReadGraphExecState(uint32_t entryIndex, TFN_OnRecord&& fnOnRecord, TFN_CommitRecord&& fnCommitRecord)
        {
            ReadConvertedState<NvTrace_CUDA_ApiTrace_GraphExecState_V1, NvTrace_CUDA_ApiTrace_GraphExecCreated>(m_sections.pGraphExecStates, m_sections.pEventVersionInfo->graphExecStateSize, GetGraphExecStateCount(), entryIndex, ConvertGraphExecState, fnOnRecord, fnCommitRecord);
        }

        template <typename TFN_OnRecord, typename TFN_CommitRecord>
        void ReadChannelState(uint32_t entryIndex, TFN_OnRecord&& fnOnRecord, TFN_CommitRecord&& fnCommitRecord)
        {
            ReadConvertedState<NvTrace_CUDA_ApiTrace_ChannelState_V1, NvTrace_CUDA_ApiTrace_ChannelCreated>(m_sections.pChannelStates, m_sections.pEventVersionInfo->channelStateSize, GetChannelStateCount(), entryIndex, ConvertChannelState, fnOnRecord, fnCommitRecord);
        }

        // Drains every state section in section order -- contexts, streams, modules (with their
        // function chain), graph execs, channels -- each record under a synthetic correlation ID.
        // Function records share their module's synthetic ref; their payload (cufunctionId, moduleId)
        // identifies them individually.
        // fnReadRecord(NvTrace_CorrelationId recordCorrelationId, uint32_t recordType, const uint8_t* pRecordBytes, size_t recordByteCount)
        // fnCommitRecord(bool consistencyCheckPassed)
        template <typename TFN_ReadRecord, typename TFN_CommitRecord>
        void ConsumeAttachStateRecords(TFN_ReadRecord&& fnReadRecord, TFN_CommitRecord&& fnCommitRecord)
        {
            const auto fnDrainTable = [](uint32_t entryCount, auto&& fnReadEntry) {
                for (uint32_t entryIndex = 0; entryIndex < entryCount; ++entryIndex)
                {
                    fnReadEntry(entryIndex);
                }
            };

            fnDrainTable(GetContextStateCount(), [&](uint32_t entryIndex) {
                ReadContextState(entryIndex, [&](uint32_t readEntryIndex, uint64_t entryRevision, const NvTrace_CUDA_ApiTrace_ContextCreated& record) { fnReadRecord(MakeV1RecordRef(readEntryIndex, entryRevision), NvTrace_CUDA_ApiTrace_RecordType_ContextCreated, reinterpret_cast<const uint8_t*>(&record), sizeof(record)); }, fnCommitRecord);
            });

            fnDrainTable(GetStreamStateCount(), [&](uint32_t entryIndex) {
                ReadStreamState(entryIndex, [&](uint32_t readEntryIndex, uint64_t entryRevision, const NvTrace_CUDA_ApiTrace_StreamCreated& record) { fnReadRecord(MakeV1RecordRef(readEntryIndex, entryRevision), NvTrace_CUDA_ApiTrace_RecordType_StreamCreated, reinterpret_cast<const uint8_t*>(&record), sizeof(record)); }, fnCommitRecord);
            });

            fnDrainTable(GetModuleStateCount(), [&](uint32_t entryIndex) {
                ReadModuleGroupRecordBytes(entryIndex, [&](uint32_t readEntryIndex, uint64_t entryRevision, uint32_t recordType, const uint8_t* pRecordBytes, size_t recordByteCount) { fnReadRecord(MakeV1RecordRef(readEntryIndex, entryRevision), recordType, pRecordBytes, recordByteCount); }, fnCommitRecord);
            });

            fnDrainTable(GetGraphExecStateCount(), [&](uint32_t entryIndex) {
                ReadGraphExecState(entryIndex, [&](uint32_t readEntryIndex, uint64_t entryRevision, const NvTrace_CUDA_ApiTrace_GraphExecCreated& record) {

                        // EncodeGraphExecCorrelationId matches what ResolveRecordReference uses for graph execs.
                        fnReadRecord(EncodeGraphExecCorrelationId(entryRevision, readEntryIndex), NvTrace_CUDA_ApiTrace_RecordType_GraphExecCreated, reinterpret_cast<const uint8_t*>(&record), sizeof(record)); }, fnCommitRecord);
            });

            fnDrainTable(GetChannelStateCount(), [&](uint32_t entryIndex) {
                ReadChannelState(entryIndex, [&](uint32_t readEntryIndex, uint64_t entryRevision, const NvTrace_CUDA_ApiTrace_ChannelCreated& record) { fnReadRecord(MakeV1RecordRef(readEntryIndex, entryRevision), NvTrace_CUDA_ApiTrace_RecordType_ChannelCreated, reinterpret_cast<const uint8_t*>(&record), sizeof(record)); }, fnCommitRecord);
            });
        }

        uint32_t DecodeModuleEntryIndex(NvTrace_CorrelationId moduleEntryCorrelationId) const
        {
            return DecodeV1RecordRef(moduleEntryCorrelationId).entryIndex;
        }

        uint64_t DecodeModuleEntryRevision(NvTrace_CorrelationId moduleEntryCorrelationId) const
        {
            return DecodeV1RecordRef(moduleEntryCorrelationId).revision;
        }

        // Synthesizes the V2 record bytes the codec wire carries for a module's group -- the converted
        // ModuleLoaded record, then one FunctionCreated per function of that module, as the generic
        // format carries them in the module entry's chain. Every record is delivered with the module
        // entry's index and revision, and each is followed by its own verdict.
        //
        // A V1 function entry names its module, so the module's functions are found by scanning the
        // function table: O(F) per module read, the price the shim pays for the chain the generic
        // format has.
        // fnOnRecordBytes(uint32_t entryIndex, uint64_t entryRevision, uint32_t recordType, const uint8_t* pRecordBytes, size_t recordByteCount)
        // fnCommitRecord(bool consistencyCheckPassed)
        template <typename TFN_OnRecordBytes, typename TFN_CommitRecord>
        void ReadModuleGroupRecordBytes(uint32_t entryIndex, TFN_OnRecordBytes&& fnOnRecordBytes, TFN_CommitRecord&& fnCommitRecord)
        {
            uint64_t moduleId = 0;
            uint64_t moduleEntryRevision = 0;
            bool moduleDelivered = false;

            ReadModuleState(
                entryIndex,
                [&](uint32_t readEntryIndex, uint64_t readEntryRevision, const NvTrace_CUDA_ApiTrace_ModuleLoaded& record) {
                    moduleId = record.moduleId;
                    moduleEntryRevision = readEntryRevision;
                    moduleDelivered = true;
                    fnOnRecordBytes(readEntryIndex, readEntryRevision, NvTrace_CUDA_ApiTrace_RecordType_ModuleLoaded, reinterpret_cast<const uint8_t*>(&record), sizeof(record));
                },
                [&](bool consistencyCheckPassed) {
                    fnCommitRecord(consistencyCheckPassed);
                    moduleDelivered = moduleDelivered && consistencyCheckPassed;
                });

            // A module whose own record a racing writer disturbed carries nothing behind it, exactly as
            // a chunk chain the reader abandons at its entry's verdict.
            if (!moduleDelivered)
            {
                return;
            }

            const uint32_t functionStateCount = GetFunctionStateCount();
            for (uint32_t functionEntryIndex = 0; functionEntryIndex < functionStateCount; ++functionEntryIndex)
            {
                ReadFunctionRecordBytes(functionEntryIndex, moduleId, entryIndex, moduleEntryRevision, fnOnRecordBytes, fnCommitRecord);
            }
        }

        // fnOnRecord(uint32_t entryIndex, uint64_t entryRevision, const CudaApiTraceGraphNodeRecord&)
        // fnCommitRecord(bool consistencyCheckPassed)
        template <typename TFN_OnRecord, typename TFN_CommitRecord>
        void ReadGraphNodeState(uint32_t entryIndex, TFN_OnRecord&& fnOnRecord, TFN_CommitRecord&& fnCommitRecord)
        {
            ReadGraphNodeEntry(
                entryIndex,
                [&](uint32_t readEntryIndex, uint64_t entryRevision, const NvTrace_CUDA_ApiTrace_GraphNodeState_V1& /* graphNodeState */, const CudaApiTraceGraphNodeRecord& record) {
                    fnOnRecord(readEntryIndex, entryRevision, record);
                },
                fnCommitRecord);
        }

        // The bytes live in an observer-owned staging buffer and stay valid until the next graph node
        // record bytes read.
        // fnOnRecordBytes(uint32_t entryIndex, uint64_t entryRevision, const uint8_t* pRecordBytes, size_t recordByteCount)
        // fnCommitRecord(bool consistencyCheckPassed)
        template <typename TFN_OnRecordBytes, typename TFN_CommitRecord>
        void ReadGraphNodeRecordBytes(uint32_t entryIndex, TFN_OnRecordBytes&& fnOnRecordBytes, TFN_CommitRecord&& fnCommitRecord)
        {
            ReadGraphNodeState(
                entryIndex,
                [&](uint32_t readEntryIndex, uint64_t entryRevision, const CudaApiTraceGraphNodeRecord& record) {
                    const size_t recordByteCount = StageGraphNodeRecordBytes(record);
                    fnOnRecordBytes(readEntryIndex, entryRevision, m_graphNodeRecordBytesBuffer.data(), recordByteCount);
                },
                fnCommitRecord);
        }

        // A V1 driver announces a graph exec but never its nodes, so an exec announce resolves to its
        // nodes here: the exec's V1 chain, in chain order, each node under its own entry's index and
        // revision. The chain head is read under the exec's revision, so a stale announce delivers
        // nothing.
        // fnOnRecordBytes(uint32_t entryIndex, uint64_t entryRevision, const uint8_t* pRecordBytes, size_t recordByteCount)
        // fnCommitRecord(bool consistencyCheckPassed)
        template <typename TFN_OnRecordBytes, typename TFN_CommitRecord>
        void ResolveGraphExecNodeRecordBytes(const NvTrace_RecordReference& graphExecReference, TFN_OnRecordBytes&& fnOnRecordBytes, TFN_CommitRecord&& fnCommitRecord)
        {
            const DecodedV1RecordRef decodedRecordRef = DecodeV1RecordRef(graphExecReference.recordRef);

            uint64_t cugraphExecId = 0;
            uint64_t graphNodeTableIndex = UINT64_MAX;
            bool graphExecMatched = false;

            ReadConvertedState<NvTrace_CUDA_ApiTrace_GraphExecState_V1, NvTrace_CUDA_ApiTrace_GraphExecCreated>(
                m_sections.pGraphExecStates,
                m_sections.pEventVersionInfo->graphExecStateSize,
                GetGraphExecStateCount(),
                decodedRecordRef.entryIndex,
                [&](const NvTrace_CUDA_ApiTrace_GraphExecState_V1& graphExecState, NvTrace_CUDA_ApiTrace_GraphExecCreated& record) {
                    ConvertGraphExecState(graphExecState, record);
                    cugraphExecId = graphExecState.cugraphExecId;
                    graphNodeTableIndex = graphExecState.firstGraphNodeTableIndex;
                },
                [&](uint32_t entryIndex, uint64_t entryRevision, const NvTrace_CUDA_ApiTrace_GraphExecCreated& /* record */) {
                    graphExecMatched = entryRevision != 0 && MakeV1RecordRef(entryIndex, entryRevision) == graphExecReference.recordRef;
                },
                [&](bool consistencyCheckPassed) { graphExecMatched = graphExecMatched && consistencyCheckPassed; });

            if (!graphExecMatched)
            {
                return;
            }

            // Each link is guarded by its node's own revision, so a link read inconsistently -- or one
            // into a slot another exec now owns -- is not chased; the visit count bounds a corrupt
            // cycle by the table's size.
            const uint32_t graphNodeStateCount = GetGraphNodeStateCount();
            for (uint32_t visitedNodeCount = 0; graphNodeTableIndex != UINT64_MAX && visitedNodeCount < graphNodeStateCount; ++visitedNodeCount)
            {
                NVTRACE_LOG_ERR_RETURN_IF(graphNodeStateCount <= graphNodeTableIndex, , 50, "V1 graph exec %" PRIu64 " chains an out-of-range graph node table index %" PRIu64 " (graphNodeStatesCount=%u).", cugraphExecId, graphNodeTableIndex, graphNodeStateCount);

                uint64_t nextGraphNodeTableIndex = UINT64_MAX;
                bool nodeDelivered = false;
                bool nodeConsistent = false;

                ReadGraphNodeEntry(
                    static_cast<uint32_t>(graphNodeTableIndex),
                    [&](uint32_t entryIndex, uint64_t entryRevision, const NvTrace_CUDA_ApiTrace_GraphNodeState_V1& graphNodeState, const CudaApiTraceGraphNodeRecord& record) {
                        nextGraphNodeTableIndex = graphNodeState.nextGraphNodeTableIndex;
                        nodeDelivered = record.common.cugraphExecId == cugraphExecId;
                        if (!nodeDelivered)
                        {
                            return;
                        }

                        const size_t recordByteCount = StageGraphNodeRecordBytes(record);
                        fnOnRecordBytes(entryIndex, entryRevision, m_graphNodeRecordBytesBuffer.data(), recordByteCount);
                    },
                    [&](bool consistencyCheckPassed) {
                        nodeConsistent = consistencyCheckPassed;
                        if (nodeDelivered)
                        {
                            fnCommitRecord(consistencyCheckPassed);
                        }
                    });

                if (!nodeDelivered || !nodeConsistent)
                {
                    return;
                }

                graphNodeTableIndex = nextGraphNodeTableIndex;
            }
        }

        // The ThreadEvent V1 shim delivers V1 static id N as source provider index N - 1.
        static uint32_t ThreadEventSourceProviderIndex()
        {
            return NvTrace_ThreadEvent_StaticProviderId_V1_CUDA - 1;
        }

        static bool IsObjectLifetimeThreadEventRecord(uint32_t recordType)
        {
            return NvTrace_CUDA_ApiTrace_RecordType_ObjectLifetime_LO_V1 < recordType && recordType < NvTrace_CUDA_ApiTrace_RecordType_ObjectLifetime_MAX_V1;
        }

        const ThreadEventRecord* TranslateThreadEventRecord(const ThreadEventRecord& record, ThreadEventRecordStorage& storage) const
        {
            if (IsObjectLifetimeThreadEventRecord(record.recordType))
            {
                const bool translationSucceeded = TranslateObjectLifetimeEvent(record, storage.translated);
                if (!translationSucceeded)
                {
                    return nullptr;
                }

                storage.record = record;
                storage.record.recordType = storage.translated.recordType;
                storage.record.pPayload = reinterpret_cast<const uint8_t*>(&storage.translated.payload);
                storage.record.payloadSize = storage.translated.payloadSize;
                return &storage.record;
            }

            if (NvTrace_CUDA_ApiTrace_RecordType_ThreadEvent_CudaDriverApiEnter_V1 <= record.recordType && record.recordType <= NvTrace_CUDA_ApiTrace_RecordType_ThreadEvent_MemOpEnd_V1)
            {
                storage.record = record;
                storage.record.recordType = TranslateCudaApiTraceV1ThreadEventRecordType(record.recordType);
                return &storage.record;
            }

            return &record;
        }

        // Resolves a reference synthesized from a V1 ObjectLifetime Created event -- eager expansion of
        // the announce the old driver sends by (tableEntryIndex, tableEntryRevision). The section index
        // vocabulary is the created record types.
        // fnReadRecord(uint64_t entryRevision, uint32_t recordType, const uint8_t* pPayload, size_t payloadSize)
        // fnCommitRecord(bool consistencyCheckPassed)
        template <typename TFN_ReadRecord, typename TFN_CommitRecord>
        void ResolveRecordReference(const NvTrace_RecordReference& reference, TFN_ReadRecord&& fnReadRecord, TFN_CommitRecord&& fnCommitRecord)
        {
            const uint32_t entryIndex = static_cast<uint32_t>(reference.recordRef);
            const NvTrace_CUDA_ApiTrace_RecordType createdRecordType = static_cast<NvTrace_CUDA_ApiTrace_RecordType>(reference.sectionIndex);

            switch (createdRecordType)
            {
                case NvTrace_CUDA_ApiTrace_RecordType_ContextCreated:
                    ResolveCreatedReference([&](auto&& fnOnRecord, auto&& fnCommit) { ReadContextState(entryIndex, fnOnRecord, fnCommit); }, reference.recordRef, createdRecordType, fnReadRecord, fnCommitRecord);
                    return;
                case NvTrace_CUDA_ApiTrace_RecordType_StreamCreated:
                    ResolveCreatedReference([&](auto&& fnOnRecord, auto&& fnCommit) { ReadStreamState(entryIndex, fnOnRecord, fnCommit); }, reference.recordRef, createdRecordType, fnReadRecord, fnCommitRecord);
                    return;
                case NvTrace_CUDA_ApiTrace_RecordType_ModuleLoaded:
                    // A reference names an entry and the resolve delivers its whole group, so a module
                    // announce brings the module's functions with it.
                    ResolveModuleGroupReference(entryIndex, reference.recordRef, fnReadRecord, fnCommitRecord);
                    return;
                case NvTrace_CUDA_ApiTrace_RecordType_GraphExecCreated:
                    ResolveCreatedReference([&](auto&& fnOnRecord, auto&& fnCommit) { ReadGraphExecState(entryIndex, fnOnRecord, fnCommit); }, reference.recordRef, createdRecordType, fnReadRecord, fnCommitRecord);
                    return;
                case NvTrace_CUDA_ApiTrace_RecordType_ChannelCreated:
                    ResolveCreatedReference([&](auto&& fnOnRecord, auto&& fnCommit) { ReadChannelState(entryIndex, fnOnRecord, fnCommit); }, reference.recordRef, createdRecordType, fnReadRecord, fnCommitRecord);
                    return;
                default:
                    NVTRACE_LOG_ERR(50, "Unknown V1 record reference section index %u.", reference.sectionIndex);
                    return;
            }
        }

        // Greedily consumes all available GPU timestamp reports across all streams.
        //
        //   void fnOnTimestampEvent(uint64_t gpuTimestampRef, uint64_t ptimer)
        //     Accepts every delivered report.
        //
        //   bool fnOnTimestampEvent(uint64_t gpuTimestampRef, uint64_t ptimer)
        //     Return true to accept the report and advance the cursor.
        //     Return false to retain the report and stop that stream until the next ConsumeGpuTimestampEvents().
        template <typename TFN_OnTimestampEvent>
        void ConsumeGpuTimestampEvents(TFN_OnTimestampEvent&& fnOnTimestampEvent)
        {
            for (CudaApiTraceGpuTimestampFifoObserverV1& gpuTimestampFifoObserver : m_gpuTimestampFifoObservers)
            {
                gpuTimestampFifoObserver.ConsumeEvents(fnOnTimestampEvent);
            }
        }

        // The V1 correlation-id scheme: a V1 revision moved by two per publish, so its seqlock bit is
        // shifted out before the index is packed. The old driver stamps QMDs with exactly these values.
        uint64_t EncodeGraphNodeCorrelationId(const uint64_t revision, const uint64_t graphNodeTableIndex) const
        {
            return ((revision >> 1) << m_graphNodeIndexBits) | graphNodeTableIndex;
        }

        uint64_t EncodeGraphExecCorrelationId(const uint64_t revision, const uint64_t graphExecTableIndex) const
        {
            return ((revision >> 1) << m_graphExecIndexBits) | graphExecTableIndex;
        }

        // A shim ref packs the (revision, entryIndex) pair the correlation encoding is defined over,
        // so the graph-exec announce's reference is unpacked here rather than by a caller that would
        // have to know this shim's private ref currency.
        uint64_t EncodeGraphExecCorrelationId(const NvTrace_RecordReference& graphExecReference) const
        {
            const DecodedV1RecordRef decodedRecordRef = DecodeV1RecordRef(graphExecReference.recordRef);
            return EncodeGraphExecCorrelationId(decodedRecordRef.revision, decodedRecordRef.entryIndex);
        }

    private:
        // The V1 revision protocol (see ThreadEventObserverV1::ReadThreadState); destructionTimestampTsc is stamped outside it, like the V2 in-place store.
        template <typename TStateV1, typename TRecord, typename TFN_Convert, typename TFN_OnRecord, typename TFN_CommitRecord>
        void ReadConvertedState(const TStateV1* pStates, uint32_t stateSize, uint32_t stateCount, uint32_t entryIndex, TFN_Convert&& fnConvert, TFN_OnRecord&& fnOnRecord, TFN_CommitRecord&& fnCommitRecord)
        {
            if (!pStates || stateCount <= entryIndex)
            {
                return;
            }

            const TStateV1& state = *reinterpret_cast<const TStateV1*>(reinterpret_cast<const uint8_t*>(pStates) + static_cast<size_t>(stateSize) * entryIndex);

            const uint64_t revisionBefore = LoadAcquire(&state.revision);
            if (revisionBefore == 0 || (revisionBefore & 1) != 0)
            {
                return;
            }

            RevisionCursor cursor{&state.revision, revisionBefore};

            TRecord record = {};
            fnConvert(state, record);
            fnOnRecord(entryIndex, revisionBefore, record);

            fnCommitRecord(cursor.CheckConsistency());
        }

        // The graph node table's ReadConvertedState: the entry's params slot follows its state, and
        // the V1 state is delivered alongside the converted record for the chain link only it carries.
        // fnOnEntry(uint32_t entryIndex, uint64_t entryRevision, const NvTrace_CUDA_ApiTrace_GraphNodeState_V1&, const CudaApiTraceGraphNodeRecord&)
        // fnCommitRecord(bool consistencyCheckPassed)
        template <typename TFN_OnEntry, typename TFN_CommitRecord>
        void ReadGraphNodeEntry(uint32_t entryIndex, TFN_OnEntry&& fnOnEntry, TFN_CommitRecord&& fnCommitRecord)
        {
            if (!m_sections.pGraphNodeStates || m_sections.pProviderBufferLayout->graphNodeStatesCount <= entryIndex)
            {
                return;
            }

            const uint8_t* const pEntryBytes = reinterpret_cast<const uint8_t*>(m_sections.pGraphNodeStates) + m_graphNodeTableEntryStride * entryIndex;
            const auto& graphNodeState = *reinterpret_cast<const NvTrace_CUDA_ApiTrace_GraphNodeState_V1*>(pEntryBytes);

            const uint64_t revisionBefore = LoadAcquire(&graphNodeState.revision);
            if (revisionBefore == 0 || (revisionBefore & 1) != 0)
            {
                return;
            }

            RevisionCursor cursor{&graphNodeState.revision, revisionBefore};

            CudaApiTraceGraphNodeRecord record = {};
            record.common.creationTimestampTsc = graphNodeState.creationTimestampTsc;
            record.common.destructionTimestampTsc = graphNodeState.destructionTimestampTsc;
            record.common.cugraphNodeId = graphNodeState.cugraphNodeId;
            record.common.cugraphNodeHandle = graphNodeState.cugraphNodeHandle;
            record.common.cugraphExecId = graphNodeState.cugraphExecId;
            record.common.nvtxTraceStackGroupRef = graphNodeState.nvtxTraceStackGroupRef == NvTrace_NvtxTrace_INVALID_REF_V1 ? NVTRACE_INVALID_REVISIONED_REF : graphNodeState.nvtxTraceStackGroupRef;
            record.common.cudaDeviceOrdinal = graphNodeState.cudaDeviceOrdinal;

            const NvTrace_CUDA_ApiTrace_GraphNodeType_V1 graphNodeTypeV1 = graphNodeState.graphNodeType;
            const NvTrace_CUDA_ApiTrace_GraphNodeType graphNodeType = TranslateCudaApiTraceV1GraphNodeType(graphNodeTypeV1);
            const uint32_t publishedParamsMinorVersion = CudaApiTraceGraphNodeParamsMinorVersion(m_recordVersionInfo, graphNodeType);

            // The reported minor version is what the entry's params slot actually spans, so a consumer
            // trusting it never reads absent or zeroed params bytes as data.
            const size_t paramsCopySize = publishedParamsMinorVersion == 0 ? 0 : AlignDnP2(std::min(static_cast<size_t>(m_graphNodeParamsSize), sizeof(record.params)), sizeof(uint64_t));
            const uint32_t paramsMinorVersion = static_cast<uint32_t>(std::min(static_cast<size_t>(publishedParamsMinorVersion), paramsCopySize));
            if (paramsCopySize != 0)
            {
                Memcpy64(&record.params, pEntryBytes + m_graphNodeStateSize, paramsCopySize);
            }

            // A type this build cannot translate makes the entry unreadable, not merely empty, so it is
            // delivered and then rejected rather than silently skipped.
            if (graphNodeType == NvTrace_CUDA_ApiTrace_GraphNodeType_Invalid)
            {
                NVTRACE_LOG_ERR(50, "V1 graph node table entry %u names a graph node type this build cannot translate (graphNodeType=%u).", entryIndex, static_cast<uint32_t>(graphNodeTypeV1));
                fnOnEntry(entryIndex, revisionBefore, graphNodeState, record);
                fnCommitRecord(false);
                return;
            }

            record.common.graphNodeType = graphNodeType;
            record.paramsMinorVersion = paramsMinorVersion;
            fnOnEntry(entryIndex, revisionBefore, graphNodeState, record);

            fnCommitRecord(cursor.CheckConsistency());
        }

        size_t StageGraphNodeRecordBytes(const CudaApiTraceGraphNodeRecord& record)
        {
            constexpr size_t commonSize = sizeof(NvTrace_CUDA_ApiTrace_GraphNodeCreated);
            const size_t paramsSize = AlignUpP2(static_cast<size_t>(record.paramsMinorVersion), sizeof(uint64_t));

            m_graphNodeRecordBytesBuffer.resize(commonSize + paramsSize);
            memcpy(m_graphNodeRecordBytesBuffer.data(), &record.common, commonSize);
            memcpy(m_graphNodeRecordBytesBuffer.data() + commonSize, &record.params, paramsSize);
            return m_graphNodeRecordBytesBuffer.size();
        }

        static void ConvertContextState(const NvTrace_CUDA_ApiTrace_ContextState_V1& state, NvTrace_CUDA_ApiTrace_ContextCreated& record)
        {
            record.creationTimestampTsc = state.creationTimestampTsc;
            record.destructionTimestampTsc = state.destructionTimestampTsc;
            record.contextId = state.contextId;
            record.cucontextHandle = state.cucontextHandle;
            record.cudaDeviceOrdinal = state.cudaDeviceOrdinal;
        }

        static void ConvertStreamState(const NvTrace_CUDA_ApiTrace_StreamState_V1& state, NvTrace_CUDA_ApiTrace_StreamCreated& record)
        {
            record.creationTimestampTsc = state.creationTimestampTsc;
            record.destructionTimestampTsc = state.destructionTimestampTsc;
            record.streamId = state.streamId;
            record.custreamHandle = state.custreamHandle;
            record.contextId = state.contextId;
            record.cudaDeviceOrdinal = state.cudaDeviceOrdinal;
        }

        static void ConvertModuleState(const NvTrace_CUDA_ApiTrace_ModuleState_V1& state, NvTrace_CUDA_ApiTrace_ModuleLoaded& record)
        {
            record.creationTimestampTsc = state.creationTimestampTsc;
            record.destructionTimestampTsc = state.destructionTimestampTsc;
            record.moduleId = state.moduleId;
            record.cumoduleHandle = state.cumoduleHandle;
            record.cudaDeviceOrdinal = state.cudaDeviceOrdinal;
        }

        static void ConvertFunctionState(const NvTrace_CUDA_ApiTrace_FunctionState_V1& state, NvTrace_CUDA_ApiTrace_FunctionCreated& record)
        {
            record.cufunctionId = state.cufunctionId;
            record.cufunctionHandle = state.cufunctionHandle;
            record.moduleId = state.moduleId;
        }

        static void ConvertGraphExecState(const NvTrace_CUDA_ApiTrace_GraphExecState_V1& state, NvTrace_CUDA_ApiTrace_GraphExecCreated& record)
        {
            record.creationTimestampTsc = state.creationTimestampTsc;
            record.destructionTimestampTsc = state.destructionTimestampTsc;
            record.cugraphExecId = state.cugraphExecId;
            record.cugraphExecHandle = state.cugraphExecHandle;
            record.cugraphId = state.cugraphId;
        }

        static void ConvertChannelState(const NvTrace_CUDA_ApiTrace_ChannelState_V1& state, NvTrace_CUDA_ApiTrace_ChannelCreated& record)
        {
            record.creationTimestampTsc = state.creationTimestampTsc;
            record.destructionTimestampTsc = state.destructionTimestampTsc;
            record.cucontextId = state.cucontextId;
            record.rmCid = state.rmCid;
        }

        uint32_t GetFunctionStateCount() const
        {
            return m_sections.pFunctionStates ? m_sections.pProviderBufferLayout->functionStatesCount : 0;
        }

        // The V1 string table has no revision protocol, so the name is best-effort: a function whose
        // name cannot be read consistently is delivered as (nullptr, 0), never dropped. The name
        // bytes live in an observer-owned staging buffer and stay valid until the next
        // ReadFunctionState call.
        // fnOnFunction(uint32_t entryIndex, uint64_t entryRevision, const NvTrace_CUDA_ApiTrace_FunctionCreated&, const char* pName, uint32_t nameLength)
        // fnCommitRecord(bool consistencyCheckPassed)
        template <typename TFN_OnFunction, typename TFN_CommitRecord>
        void ReadFunctionState(uint32_t entryIndex, TFN_OnFunction&& fnOnFunction, TFN_CommitRecord&& fnCommitRecord)
        {
            uint64_t nameStringTableEntryIndex = 0;
            uint32_t nameStringLength = 0;
            ReadConvertedState<NvTrace_CUDA_ApiTrace_FunctionState_V1, NvTrace_CUDA_ApiTrace_FunctionCreated>(
                m_sections.pFunctionStates,
                m_sections.pEventVersionInfo->functionStateSize,
                GetFunctionStateCount(),
                entryIndex,
                [&](const NvTrace_CUDA_ApiTrace_FunctionState_V1& functionState, NvTrace_CUDA_ApiTrace_FunctionCreated& record) {
                    ConvertFunctionState(functionState, record);
                    nameStringTableEntryIndex = functionState.nameStringTableEntryIndex;
                    nameStringLength = functionState.nameStringLength;
                },
                [&](uint32_t readEntryIndex, uint64_t entryRevision, const NvTrace_CUDA_ApiTrace_FunctionCreated& record) {
                    // The name is read with the record still tentative, so it rides the same verdict.
                    // A name longer than the whole string-table data cannot be valid; reject before
                    // allocating.
                    const bool nameLengthValid = nameStringLength <= m_sections.pProviderBufferLayout->stringTableDataSize;
                    if (nameLengthValid)
                    {
                        m_functionNameBuffer.resize(nameStringLength);
                    }
                    if (!nameLengthValid || nameStringLength == 0 || !ReadString(nameStringTableEntryIndex, nameStringLength, m_functionNameBuffer.data()))
                    {
                        m_functionNameBuffer.clear();
                    }

                    // The counts are set over what the read actually delivered, so a name the table did
                    // not serve reports none.
                    const uint32_t nameByteCount = static_cast<uint32_t>(m_functionNameBuffer.size());

                    NvTrace_CUDA_ApiTrace_FunctionCreated deliveredRecord = record;
                    deliveredRecord.totalNameLength = nameByteCount;
                    deliveredRecord.inlineNameSegmentLength = nameByteCount;

                    fnOnFunction(readEntryIndex, entryRevision, deliveredRecord, m_functionNameBuffer.empty() ? nullptr : m_functionNameBuffer.data(), nameByteCount);
                },
                fnCommitRecord);
        }

        // fnOnRecordBytes(uint32_t entryIndex, uint64_t entryRevision, uint32_t recordType, const uint8_t* pRecordBytes, size_t recordByteCount)
        // fnCommitRecord(bool consistencyCheckPassed)
        template <typename TFN_OnRecordBytes, typename TFN_CommitRecord>
        void ReadFunctionRecordBytes(uint32_t functionEntryIndex, uint64_t moduleId, uint32_t moduleEntryIndex, uint64_t moduleEntryRevision, TFN_OnRecordBytes&& fnOnRecordBytes, TFN_CommitRecord&& fnCommitRecord)
        {
            bool functionDelivered = false;
            ReadFunctionState(
                functionEntryIndex,
                [&](uint32_t /* readEntryIndex */, uint64_t /* entryRevision */, const NvTrace_CUDA_ApiTrace_FunctionCreated& record, const char* pName, uint32_t nameLength) {
                    functionDelivered = record.moduleId == moduleId;
                    if (!functionDelivered)
                    {
                        return;
                    }

                    const uint8_t* const pRecordBytes = reinterpret_cast<const uint8_t*>(&record);
                    const uint8_t* const pNameBytes = reinterpret_cast<const uint8_t*>(pName);
                    m_functionRecordBytesBuffer.assign(NVTRACE_ALIGN_PAYLOAD_SIZE_DIV8(sizeof(record) + nameLength) * sizeof(uint64_t), 0);
                    memcpy(m_functionRecordBytesBuffer.data(), pRecordBytes, sizeof(record));
                    memcpy(m_functionRecordBytesBuffer.data() + sizeof(record), pNameBytes, nameLength);

                    // A function is a chain record of its module's entry, so it travels under the
                    // module's identity rather than the V1 function table's.
                    fnOnRecordBytes(moduleEntryIndex, moduleEntryRevision, NvTrace_CUDA_ApiTrace_RecordType_FunctionCreated, m_functionRecordBytesBuffer.data(), m_functionRecordBytesBuffer.size());
                },
                [&](bool consistencyCheckPassed) {
                    if (functionDelivered)
                    {
                        fnCommitRecord(consistencyCheckPassed);
                    }
                });
        }

        // Translates a record of the retired ObjectLifetime band -- the vocabulary only a V1 provider
        // emits -- into its V2 shape: a created object into the reference its announce resolves
        // through; a destroyed object into the self-contained ThreadEvent_*Destroyed record.
        bool TranslateObjectLifetimeEvent(const ThreadEventRecord& record, CudaApiTraceTranslatedRecordV1& out_translated) const
        {
            const NvTrace_CUDA_ApiTrace_RecordType createdRecordType = TranslateCudaApiTraceV1ObjectLifetimeCreatedRecordType(record.recordType);
            if (createdRecordType != NvTrace_CUDA_ApiTrace_RecordType_INVALID)
            {
                return TranslateObjectCreatedEvent(record, createdRecordType, out_translated);
            }

            return TranslateObjectDestroyedEvent(record, out_translated);
        }

        // A created announce carries nothing but its event header: the object's state is read from the
        // table on resolve.
        bool TranslateObjectCreatedEvent(const ThreadEventRecord& record, NvTrace_CUDA_ApiTrace_RecordType createdRecordType, CudaApiTraceTranslatedRecordV1& out_translated) const
        {
            NvTrace_CUDA_ApiTrace_ObjectLifetime_EventHeader_V1 header = {};
            const bool headerRead = TryReadObjectLifetimeEventHeader(record, header);
            if (!headerRead)
            {
                return false;
            }

            MakeCreatedReference(createdRecordType, MakeV1RecordRef(static_cast<uint32_t>(header.tableEntryIndex), header.tableEntryRevision), out_translated.payload.reference);

            out_translated.recordType = NvTrace_ReservedRecordType_RecordReference;
            out_translated.payloadSize = sizeof(NvTrace_RecordReference);
            return true;
        }

        bool TranslateObjectDestroyedEvent(const ThreadEventRecord& record, CudaApiTraceTranslatedRecordV1& out_translated) const
        {
            const NvTrace_CUDA_ApiTrace_RecordType destroyedRecordType = TranslateCudaApiTraceV1ObjectLifetimeDestroyedRecordType(record.recordType);
            NVTRACE_LOG_ERR_RETURN_IF(destroyedRecordType == NvTrace_CUDA_ApiTrace_RecordType_INVALID, false, 50, "Unknown V1 CUDA object-lifetime event type %u.", record.recordType);

            NvTrace_CUDA_ApiTrace_ObjectLifetime_EventHeader_V1 header = {};
            const bool headerRead = TryReadObjectLifetimeEventHeader(record, header);
            if (!headerRead)
            {
                return false;
            }

            const size_t payloadArmSize = record.payloadSize - m_objectLifetimePayloadOffset;
            NVTRACE_LOG_ERR_RETURN_IF(payloadArmSize < sizeof(uint64_t), false, 50, "V1 CUDA object-lifetime ThreadEvent record carries no destroyed payload (recordType=%u, payloadSize=%zu).", record.recordType, record.payloadSize);

            // Every V1 destroyed payload is the destroyed object's id and nothing else, so the arm the
            // record type selects need not be known to read it.
            static_assert(sizeof(NvTrace_CUDA_ApiTrace_ObjectLifetime_ContextDestroyed_V1) == sizeof(uint64_t) && sizeof(NvTrace_CUDA_ApiTrace_ObjectLifetime_StreamDestroyed_V1) == sizeof(uint64_t) && sizeof(NvTrace_CUDA_ApiTrace_ObjectLifetime_ModuleUnloaded_V1) == sizeof(uint64_t) && sizeof(NvTrace_CUDA_ApiTrace_ObjectLifetime_GraphExecDestroyed_V1) == sizeof(uint64_t), "Every V1 object-lifetime destroyed payload must be a single object id");

            uint64_t objectId;
            memcpy(&objectId, record.pPayload + m_objectLifetimePayloadOffset, sizeof(objectId));

            // The four destroyed records are one layout under four sets of field names, so the
            // canonical arm's write serves every family.
            static_assert(sizeof(NvTrace_CUDA_ApiTrace_StreamDestroyed) == sizeof(NvTrace_CUDA_ApiTrace_ContextDestroyed) && offsetof(NvTrace_CUDA_ApiTrace_StreamDestroyed, timestampTsc) == offsetof(NvTrace_CUDA_ApiTrace_ContextDestroyed, timestampTsc) && offsetof(NvTrace_CUDA_ApiTrace_StreamDestroyed, streamId) == offsetof(NvTrace_CUDA_ApiTrace_ContextDestroyed, contextId) && NvTrace_CUDA_ApiTrace_StreamDestroyed_MINOR_VERSION_A == NvTrace_CUDA_ApiTrace_ContextDestroyed_MINOR_VERSION_A, "The stream-destroyed record must share the context-destroyed layout the canonical arm writes");
            static_assert(sizeof(NvTrace_CUDA_ApiTrace_ModuleUnloaded) == sizeof(NvTrace_CUDA_ApiTrace_ContextDestroyed) && offsetof(NvTrace_CUDA_ApiTrace_ModuleUnloaded, timestampTsc) == offsetof(NvTrace_CUDA_ApiTrace_ContextDestroyed, timestampTsc) && offsetof(NvTrace_CUDA_ApiTrace_ModuleUnloaded, moduleId) == offsetof(NvTrace_CUDA_ApiTrace_ContextDestroyed, contextId) && NvTrace_CUDA_ApiTrace_ModuleUnloaded_MINOR_VERSION_A == NvTrace_CUDA_ApiTrace_ContextDestroyed_MINOR_VERSION_A, "The module-unloaded record must share the context-destroyed layout the canonical arm writes");
            static_assert(sizeof(NvTrace_CUDA_ApiTrace_GraphExecDestroyed) == sizeof(NvTrace_CUDA_ApiTrace_ContextDestroyed) && offsetof(NvTrace_CUDA_ApiTrace_GraphExecDestroyed, timestampTsc) == offsetof(NvTrace_CUDA_ApiTrace_ContextDestroyed, timestampTsc) && offsetof(NvTrace_CUDA_ApiTrace_GraphExecDestroyed, cugraphExecId) == offsetof(NvTrace_CUDA_ApiTrace_ContextDestroyed, contextId) && NvTrace_CUDA_ApiTrace_GraphExecDestroyed_MINOR_VERSION_A == NvTrace_CUDA_ApiTrace_ContextDestroyed_MINOR_VERSION_A, "The graph-exec-destroyed record must share the context-destroyed layout the canonical arm writes");

            out_translated.recordType = destroyedRecordType;
            out_translated.payloadSize = NvTrace_CUDA_ApiTrace_ContextDestroyed_MINOR_VERSION_A;
            out_translated.payload.contextDestroyed = {header.timestampTsc, objectId};
            return true;
        }

        // The payload arm -- if the record carries one -- begins at the V1 buffer's own published
        // header size. False when the record cannot hold the header, or names a table entry index
        // wider than the shim's ref currency carries.
        bool TryReadObjectLifetimeEventHeader(const ThreadEventRecord& record, NvTrace_CUDA_ApiTrace_ObjectLifetime_EventHeader_V1& out_header) const
        {
            NVTRACE_LOG_ERR_RETURN_IF(record.pPayload == nullptr || record.payloadSize < m_objectLifetimePayloadOffset, false, 50, "Invalid V1 CUDA object-lifetime ThreadEvent record (recordType=%u, payloadSize=%zu).", record.recordType, record.payloadSize);

            memcpy(&out_header, record.pPayload, std::min(sizeof(out_header), static_cast<size_t>(m_objectLifetimePayloadOffset)));

            NVTRACE_LOG_ERR_RETURN_IF(std::numeric_limits<uint32_t>::max() < out_header.tableEntryIndex, false, 50, "V1 CUDA object-lifetime ThreadEvent record names an out-of-range table entry (recordType=%u, tableEntryIndex=%" PRIu64 ").", record.recordType, out_header.tableEntryIndex);

            return true;
        }

        // recordRef packs (revision, entryIndex) high-to-low, the revision truncated to 32 bits; this
        // packing is private to the shim, which both mints and resolves these refs, so no
        // codec-derived field widths apply.
        static uint64_t MakeV1RecordRef(uint32_t entryIndex, uint64_t revision)
        {
            return (revision << 32) | entryIndex;
        }

        struct DecodedV1RecordRef
        {
            uint64_t revision = 0;
            uint32_t entryIndex = 0;
        };

        static DecodedV1RecordRef DecodeV1RecordRef(uint64_t recordRef)
        {
            return {recordRef >> 32, static_cast<uint32_t>(recordRef)};
        }

        // Mints the reference a created announce resolves through. A V1 buffer has no section indices,
        // so the shim's private sectionIndex vocabulary is the created record type itself.
        static void MakeCreatedReference(NvTrace_CUDA_ApiTrace_RecordType createdRecordType, uint64_t entryRecordRef, NvTrace_RecordReference& out_reference)
        {
            out_reference = {};
            out_reference.sectionIndex = createdRecordType;
            out_reference.recordRef = entryRecordRef;
        }

        // A resolve validates in ref space -- re-encoding the read revision -- so a stale announce
        // (the entry modified or reassigned since the event) resolves to nothing, exactly like a V2
        // reference. Revision 0 is a never-published entry, which no valid announce can name.
        template <typename TFN_ReadState, typename TFN_ReadRecord, typename TFN_CommitRecord>
        static void ResolveCreatedReference(TFN_ReadState&& fnReadState, uint64_t recordRef, NvTrace_CUDA_ApiTrace_RecordType recordType, TFN_ReadRecord&& fnReadRecord, TFN_CommitRecord&& fnCommitRecord)
        {
            bool refMatched = false;
            fnReadState(
                [&](uint32_t entryIndex, uint64_t entryRevision, const auto& record) {
                    refMatched = entryRevision != 0 && MakeV1RecordRef(entryIndex, entryRevision) == recordRef;
                    if (refMatched)
                    {
                        fnReadRecord(entryRevision, recordType, reinterpret_cast<const uint8_t*>(&record), sizeof(record));
                    }
                },
                [&](bool consistencyCheckPassed) {
                    if (refMatched)
                    {
                        fnCommitRecord(consistencyCheckPassed);
                    }
                });
        }

        // ResolveCreatedReference over a module's whole group. Every record of the group carries the
        // module entry's ref, so the same stale-announce check answers for each of them.
        template <typename TFN_ReadRecord, typename TFN_CommitRecord>
        void ResolveModuleGroupReference(uint32_t entryIndex, uint64_t recordRef, TFN_ReadRecord&& fnReadRecord, TFN_CommitRecord&& fnCommitRecord)
        {
            bool refMatched = false;
            ReadModuleGroupRecordBytes(
                entryIndex,
                [&](uint32_t readEntryIndex, uint64_t entryRevision, uint32_t recordType, const uint8_t* pRecordBytes, size_t recordByteCount) {
                    refMatched = entryRevision != 0 && MakeV1RecordRef(readEntryIndex, entryRevision) == recordRef;
                    if (refMatched)
                    {
                        fnReadRecord(entryRevision, recordType, pRecordBytes, recordByteCount);
                    }
                },
                [&](bool consistencyCheckPassed) {
                    if (refMatched)
                    {
                        fnCommitRecord(consistencyCheckPassed);
                    }
                });
        }

        /// Deserializes the string at stringTableEntryIndex into pString, which must hold stringLength
        /// bytes; no NUL terminator is written. False on any inconsistency -- the table is unprotected.
        bool ReadString(uint64_t stringTableEntryIndex, size_t stringLength, char* pString) const
        {
            if (!m_sections.pStringTableEntries || m_sections.pProviderBufferLayout->stringTableEntriesCount <= stringTableEntryIndex)
            {
                return false;
            }

            const size_t stringTableEntrySize = m_sections.pEventVersionInfo->stringTableEntrySize;
            const uint8_t* const pStringTableEntryBase = reinterpret_cast<const uint8_t*>(m_sections.pStringTableEntries) + stringTableEntryIndex * stringTableEntrySize;
            const uint64_t remainingEntryCount = m_sections.pProviderBufferLayout->stringTableEntriesCount - stringTableEntryIndex;

            bool isLast = false;
            size_t stringOffset = 0;
            for (uint64_t entryIndex = 0; !isLast && stringOffset < stringLength; ++entryIndex)
            {
                if (remainingEntryCount <= entryIndex)
                {
                    return false;
                }

                const uint64_t encodedValue = static_cast<uint64_t>(reinterpret_cast<const NvTrace_CUDA_ApiTrace_StringTableEntry_V1*>(pStringTableEntryBase + entryIndex * stringTableEntrySize)->value);
                const uint64_t offset = NVTRACE_GETFIELD64(encodedValue, NVTRACE_STRING_TABLE_ENTRY_OFFSET);
                const uint64_t length = NVTRACE_GETFIELD64(encodedValue, NVTRACE_STRING_TABLE_ENTRY_LENGTH);
                isLast = NVTRACE_GETFIELD64(encodedValue, NVTRACE_STRING_TABLE_ENTRY_ISLAST) != 0;

                if (stringLength < stringOffset + length || m_sections.pProviderBufferLayout->stringTableDataSize < offset + length)
                {
                    return false;
                }

                memcpy(pString + stringOffset, m_sections.pStringTableData + offset, length);
                stringOffset += length;
            }

            return stringOffset == stringLength;
        }

        /// The layout of a V1 graph node table entry -- a NvTrace_CUDA_ApiTrace_GraphNodeState_V1 followed
        /// by the params slot its graphNodeType selects. A zero entryStride is an unreadable table; a
        /// zero paramsSize is a readable table whose params slot is not.
        struct GraphNodeTableLayout
        {
            uint32_t stateSize = 0;
            uint32_t paramsSize = 0;
            uint64_t entryStride = 0;
        };

        static GraphNodeTableLayout ResolveGraphNodeTableLayout(const NvTrace_CUDA_ApiTrace_EventVersionInfo_V1& eventVersionInfo)
        {
            GraphNodeTableLayout graphNodeTableLayout;

            // The params slot spans the entry whether or not its contents turn out to be usable.
            const uint64_t entryStride = static_cast<uint64_t>(eventVersionInfo.graphNodeStateSize) + eventVersionInfo.graphNodeParamsSize;

            // Every entry after the first is reached by striding, so a stride that is not a multiple of
            // the state's alignment misaligns the state itself. That disables the whole table.
            constexpr uint64_t graphNodeStateAlignment = alignof(NvTrace_CUDA_ApiTrace_GraphNodeState_V1);
            if ((entryStride % graphNodeStateAlignment) != 0)
            {
                NVTRACE_LOG_INF(20, "V1 graph node table entry stride %" PRIu64 " is not a multiple of the %" PRIu64 " byte graph node state alignment; graph node state data will not be available.", entryStride, graphNodeStateAlignment);
                return graphNodeTableLayout;
            }

            graphNodeTableLayout.stateSize = eventVersionInfo.graphNodeStateSize;
            graphNodeTableLayout.entryStride = entryStride;

            // The params begin at graphNodeStateSize within each entry, and the shim lifts them out
            // with Memcpy64, which reads 64-bit words -- so that offset must be 8-aligned whatever
            // the params themselves need.
            if ((eventVersionInfo.graphNodeStateSize % alignof(uint64_t)) != 0)
            {
                NVTRACE_LOG_INF(20, "V1 graph node state size %u is not a multiple of the %zu byte word the params are copied in; graph node params will not be available.", eventVersionInfo.graphNodeStateSize, alignof(uint64_t));
                return graphNodeTableLayout;
            }

            graphNodeTableLayout.paramsSize = eventVersionInfo.graphNodeParamsSize;
            return graphNodeTableLayout;
        }

        // This build's values for the records the shim converts or synthesizes, destruction records
        // included; the producer's V1 values for the payloads that pass through untranslated (thread
        // events, memory operations); and per-arm graph-node params minors clamped to what the shim
        // stages.
        static NvTrace_CUDA_ApiTrace_RecordVersionInfo MakeRecordVersionInfoFromV1(const NvTrace_CUDA_ApiTrace_EventVersionInfo_V1& eventVersionInfo)
        {
            NvTrace_CUDA_ApiTrace_RecordVersionInfo recordVersionInfo = {};

            recordVersionInfo.recordVersionInfoSize = sizeof(NvTrace_CUDA_ApiTrace_RecordVersionInfo);
            recordVersionInfo.recordVersionInfoMinorVersion = NvTrace_CUDA_ApiTrace_RecordVersionInfo_MINOR_VERSION_A;

            recordVersionInfo.deviceCreatedSize = sizeof(NvTrace_CUDA_ApiTrace_DeviceCreated);
            recordVersionInfo.deviceCreatedMinorVersion = NvTrace_CUDA_ApiTrace_DeviceCreated_MINOR_VERSION_A;
            recordVersionInfo.contextCreatedSize = sizeof(NvTrace_CUDA_ApiTrace_ContextCreated);
            recordVersionInfo.contextCreatedMinorVersion = NvTrace_CUDA_ApiTrace_ContextCreated_MINOR_VERSION_A;
            recordVersionInfo.streamCreatedSize = sizeof(NvTrace_CUDA_ApiTrace_StreamCreated);
            recordVersionInfo.streamCreatedMinorVersion = NvTrace_CUDA_ApiTrace_StreamCreated_MINOR_VERSION_A;
            recordVersionInfo.moduleLoadedSize = sizeof(NvTrace_CUDA_ApiTrace_ModuleLoaded);
            recordVersionInfo.moduleLoadedMinorVersion = NvTrace_CUDA_ApiTrace_ModuleLoaded_MINOR_VERSION_A;
            recordVersionInfo.functionCreatedSize = sizeof(NvTrace_CUDA_ApiTrace_FunctionCreated);
            recordVersionInfo.functionCreatedMinorVersion = NvTrace_CUDA_ApiTrace_FunctionCreated_MINOR_VERSION_A;
            recordVersionInfo.graphExecCreatedSize = sizeof(NvTrace_CUDA_ApiTrace_GraphExecCreated);
            recordVersionInfo.graphExecCreatedMinorVersion = NvTrace_CUDA_ApiTrace_GraphExecCreated_MINOR_VERSION_A;
            recordVersionInfo.graphNodeCreatedSize = sizeof(NvTrace_CUDA_ApiTrace_GraphNodeCreated);
            recordVersionInfo.graphNodeCreatedMinorVersion = NvTrace_CUDA_ApiTrace_GraphNodeCreated_MINOR_VERSION_A;
            recordVersionInfo.channelCreatedSize = sizeof(NvTrace_CUDA_ApiTrace_ChannelCreated);
            recordVersionInfo.channelCreatedMinorVersion = NvTrace_CUDA_ApiTrace_ChannelCreated_MINOR_VERSION_A;
            recordVersionInfo.functionNameSegmentSize = sizeof(NvTrace_CUDA_ApiTrace_FunctionNameSegment);
            recordVersionInfo.functionNameSegmentMinorVersion = NvTrace_CUDA_ApiTrace_FunctionNameSegment_MINOR_VERSION_A;
            recordVersionInfo.gpuTimestampReportSize = sizeof(NvTrace_CUDA_ApiTrace_GpuTimestampReport);
            recordVersionInfo.gpuTimestampReportMinorVersion = NvTrace_CUDA_ApiTrace_GpuTimestampReport_MINOR_VERSION_A;
            recordVersionInfo.cudaDriverApiEnterSize = eventVersionInfo.threadEventCudaDriverApiEnterSize;
            recordVersionInfo.cudaDriverApiEnterMinorVersion = eventVersionInfo.threadEventCudaDriverApiEnterMinorVersion;
            recordVersionInfo.cudaDriverApiExitSize = eventVersionInfo.threadEventCudaDriverApiExitSize;
            recordVersionInfo.cudaDriverApiExitMinorVersion = eventVersionInfo.threadEventCudaDriverApiExitMinorVersion;
            recordVersionInfo.cudaRuntimeApiEnterSize = eventVersionInfo.threadEventCudaRuntimeApiEnterSize;
            recordVersionInfo.cudaRuntimeApiEnterMinorVersion = eventVersionInfo.threadEventCudaRuntimeApiEnterMinorVersion;
            recordVersionInfo.cudaRuntimeApiExitSize = eventVersionInfo.threadEventCudaRuntimeApiExitSize;
            recordVersionInfo.cudaRuntimeApiExitMinorVersion = eventVersionInfo.threadEventCudaRuntimeApiExitMinorVersion;
            recordVersionInfo.immediateKernelLaunchSize = eventVersionInfo.threadEventImmediateKernelLaunchSize;
            recordVersionInfo.immediateKernelLaunchMinorVersion = eventVersionInfo.threadEventImmediateKernelLaunchMinorVersion;
            recordVersionInfo.graphLaunchSize = eventVersionInfo.threadEventGraphLaunchSize;
            recordVersionInfo.graphLaunchMinorVersion = eventVersionInfo.threadEventGraphLaunchMinorVersion;
            recordVersionInfo.memcpyBeginSize = eventVersionInfo.threadEventMemcpyBeginSize;
            recordVersionInfo.memcpyBeginMinorVersion = eventVersionInfo.threadEventMemcpyBeginMinorVersion;
            recordVersionInfo.memsetBeginSize = eventVersionInfo.threadEventMemsetBeginSize;
            recordVersionInfo.memsetBeginMinorVersion = eventVersionInfo.threadEventMemsetBeginMinorVersion;
            recordVersionInfo.hostCallbackLaunchSize = eventVersionInfo.threadEventHostCallbackLaunchSize;
            recordVersionInfo.hostCallbackLaunchMinorVersion = eventVersionInfo.threadEventHostCallbackLaunchMinorVersion;
            recordVersionInfo.hostCallbackExecBeginSize = eventVersionInfo.threadEventHostCallbackExecBeginSize;
            recordVersionInfo.hostCallbackExecBeginMinorVersion = eventVersionInfo.threadEventHostCallbackExecBeginMinorVersion;
            recordVersionInfo.hostCallbackExecEndSize = eventVersionInfo.threadEventHostCallbackExecEndSize;
            recordVersionInfo.hostCallbackExecEndMinorVersion = eventVersionInfo.threadEventHostCallbackExecEndMinorVersion;
            recordVersionInfo.graphMemcpyBeginSize = eventVersionInfo.threadEventGraphMemcpyBeginSize;
            recordVersionInfo.graphMemcpyBeginMinorVersion = eventVersionInfo.threadEventGraphMemcpyBeginMinorVersion;
            recordVersionInfo.graphMemsetBeginSize = eventVersionInfo.threadEventGraphMemsetBeginSize;
            recordVersionInfo.graphMemsetBeginMinorVersion = eventVersionInfo.threadEventGraphMemsetBeginMinorVersion;
            recordVersionInfo.graphHostCallbackLaunchSize = eventVersionInfo.threadEventGraphHostCallbackLaunchSize;
            recordVersionInfo.graphHostCallbackLaunchMinorVersion = eventVersionInfo.threadEventGraphHostCallbackLaunchMinorVersion;
            recordVersionInfo.graphHostCallbackExecBeginSize = eventVersionInfo.threadEventGraphHostCallbackExecBeginSize;
            recordVersionInfo.graphHostCallbackExecBeginMinorVersion = eventVersionInfo.threadEventGraphHostCallbackExecBeginMinorVersion;
            recordVersionInfo.graphHostCallbackExecEndSize = eventVersionInfo.threadEventGraphHostCallbackExecEndSize;
            recordVersionInfo.graphHostCallbackExecEndMinorVersion = eventVersionInfo.threadEventGraphHostCallbackExecEndMinorVersion;
            recordVersionInfo.memOpBeginSize = eventVersionInfo.threadEventMemOpBeginSize;
            recordVersionInfo.memOpBeginMinorVersion = eventVersionInfo.threadEventMemOpBeginMinorVersion;

            // The destruction records are synthesized out of the V1 object-lifetime band, so the shim
            // publishes the shape it produces rather than anything the producer published.
            recordVersionInfo.contextDestroyedSize = sizeof(NvTrace_CUDA_ApiTrace_ContextDestroyed);
            recordVersionInfo.contextDestroyedMinorVersion = NvTrace_CUDA_ApiTrace_ContextDestroyed_MINOR_VERSION_A;
            recordVersionInfo.streamDestroyedSize = sizeof(NvTrace_CUDA_ApiTrace_StreamDestroyed);
            recordVersionInfo.streamDestroyedMinorVersion = NvTrace_CUDA_ApiTrace_StreamDestroyed_MINOR_VERSION_A;
            recordVersionInfo.moduleUnloadedSize = sizeof(NvTrace_CUDA_ApiTrace_ModuleUnloaded);
            recordVersionInfo.moduleUnloadedMinorVersion = NvTrace_CUDA_ApiTrace_ModuleUnloaded_MINOR_VERSION_A;
            recordVersionInfo.graphExecDestroyedSize = sizeof(NvTrace_CUDA_ApiTrace_GraphExecDestroyed);
            recordVersionInfo.graphExecDestroyedMinorVersion = NvTrace_CUDA_ApiTrace_GraphExecDestroyed_MINOR_VERSION_A;

            recordVersionInfo.graphNodeKernelParamsSize = eventVersionInfo.graphNodeKernelParamsSize;
            recordVersionInfo.graphNodeKernelParamsMinorVersion = std::min<uint32_t>(eventVersionInfo.graphNodeKernelParamsMinorVersion, NvTrace_CUDA_ApiTrace_GraphNode_KernelParams_MINOR_VERSION_A);
            recordVersionInfo.graphNodeMemcpyParamsSize = eventVersionInfo.graphNodeMemcpyParamsSize;
            recordVersionInfo.graphNodeMemcpyParamsMinorVersion = std::min<uint32_t>(eventVersionInfo.graphNodeMemcpyParamsMinorVersion, NvTrace_CUDA_ApiTrace_GraphNode_MemcpyParams_MINOR_VERSION_A);
            recordVersionInfo.graphNodeMemsetParamsSize = eventVersionInfo.graphNodeMemsetParamsSize;
            recordVersionInfo.graphNodeMemsetParamsMinorVersion = std::min<uint32_t>(eventVersionInfo.graphNodeMemsetParamsMinorVersion, NvTrace_CUDA_ApiTrace_GraphNode_MemsetParams_MINOR_VERSION_A);
            recordVersionInfo.graphNodeHostParamsSize = eventVersionInfo.graphNodeHostParamsSize;
            recordVersionInfo.graphNodeHostParamsMinorVersion = std::min<uint32_t>(eventVersionInfo.graphNodeHostParamsMinorVersion, NvTrace_CUDA_ApiTrace_GraphNode_HostParams_MINOR_VERSION_A);
            recordVersionInfo.graphNodeChildGraphParamsSize = eventVersionInfo.graphNodeChildGraphParamsSize;
            recordVersionInfo.graphNodeChildGraphParamsMinorVersion = std::min<uint32_t>(eventVersionInfo.graphNodeChildGraphParamsMinorVersion, NvTrace_CUDA_ApiTrace_GraphNode_ChildGraphParams_MINOR_VERSION_A);
            recordVersionInfo.graphNodeWaitEventParamsSize = eventVersionInfo.graphNodeWaitEventParamsSize;
            recordVersionInfo.graphNodeWaitEventParamsMinorVersion = std::min<uint32_t>(eventVersionInfo.graphNodeWaitEventParamsMinorVersion, NvTrace_CUDA_ApiTrace_GraphNode_WaitEventParams_MINOR_VERSION_A);
            recordVersionInfo.graphNodeEventRecordParamsSize = eventVersionInfo.graphNodeEventRecordParamsSize;
            recordVersionInfo.graphNodeEventRecordParamsMinorVersion = std::min<uint32_t>(eventVersionInfo.graphNodeEventRecordParamsMinorVersion, NvTrace_CUDA_ApiTrace_GraphNode_EventRecordParams_MINOR_VERSION_A);
            recordVersionInfo.graphNodeExtSemasSignalParamsSize = eventVersionInfo.graphNodeExtSemasSignalParamsSize;
            recordVersionInfo.graphNodeExtSemasSignalParamsMinorVersion = std::min<uint32_t>(eventVersionInfo.graphNodeExtSemasSignalParamsMinorVersion, NvTrace_CUDA_ApiTrace_GraphNode_ExtSemasSignalParams_MINOR_VERSION_A);
            recordVersionInfo.graphNodeExtSemasWaitParamsSize = eventVersionInfo.graphNodeExtSemasWaitParamsSize;
            recordVersionInfo.graphNodeExtSemasWaitParamsMinorVersion = std::min<uint32_t>(eventVersionInfo.graphNodeExtSemasWaitParamsMinorVersion, NvTrace_CUDA_ApiTrace_GraphNode_ExtSemasWaitParams_MINOR_VERSION_A);
            recordVersionInfo.graphNodeMemAllocParamsSize = eventVersionInfo.graphNodeMemAllocParamsSize;
            recordVersionInfo.graphNodeMemAllocParamsMinorVersion = std::min<uint32_t>(eventVersionInfo.graphNodeMemAllocParamsMinorVersion, NvTrace_CUDA_ApiTrace_GraphNode_MemAllocParams_MINOR_VERSION_A);
            recordVersionInfo.graphNodeMemFreeParamsSize = eventVersionInfo.graphNodeMemFreeParamsSize;
            recordVersionInfo.graphNodeMemFreeParamsMinorVersion = std::min<uint32_t>(eventVersionInfo.graphNodeMemFreeParamsMinorVersion, NvTrace_CUDA_ApiTrace_GraphNode_MemFreeParams_MINOR_VERSION_A);
            recordVersionInfo.graphNodeBatchMemOpParamsSize = eventVersionInfo.graphNodeBatchMemOpParamsSize;
            recordVersionInfo.graphNodeBatchMemOpParamsMinorVersion = std::min<uint32_t>(eventVersionInfo.graphNodeBatchMemOpParamsMinorVersion, NvTrace_CUDA_ApiTrace_GraphNode_BatchMemOpParams_MINOR_VERSION_A);
            recordVersionInfo.graphNodeConditionalParamsSize = eventVersionInfo.graphNodeConditionalParamsSize;
            recordVersionInfo.graphNodeConditionalParamsMinorVersion = std::min<uint32_t>(eventVersionInfo.graphNodeConditionalParamsMinorVersion, NvTrace_CUDA_ApiTrace_GraphNode_ConditionalParams_MINOR_VERSION_A);
            recordVersionInfo.graphNodeMemcpyBatchParamsSize = eventVersionInfo.graphNodeMemcpyBatchParamsSize;
            recordVersionInfo.graphNodeMemcpyBatchParamsMinorVersion = std::min<uint32_t>(eventVersionInfo.graphNodeMemcpyBatchParamsMinorVersion, NvTrace_CUDA_ApiTrace_GraphNode_MemcpyBatchParams_MINOR_VERSION_A);
            recordVersionInfo.graphNodeMemcpyBatch3DParamsSize = eventVersionInfo.graphNodeMemcpyBatch3DParamsSize;
            recordVersionInfo.graphNodeMemcpyBatch3DParamsMinorVersion = std::min<uint32_t>(eventVersionInfo.graphNodeMemcpyBatch3DParamsMinorVersion, NvTrace_CUDA_ApiTrace_GraphNode_MemcpyBatch3DParams_MINOR_VERSION_A);

            recordVersionInfo.memOpWriteValueParamsSize = eventVersionInfo.memOpWriteValueParamsSize;
            recordVersionInfo.memOpWriteValueParamsMinorVersion = eventVersionInfo.memOpWriteValueParamsMinorVersion;
            recordVersionInfo.memOpWaitValueParamsSize = eventVersionInfo.memOpWaitValueParamsSize;
            recordVersionInfo.memOpWaitValueParamsMinorVersion = eventVersionInfo.memOpWaitValueParamsMinorVersion;
            recordVersionInfo.memOpBarrierParamsSize = eventVersionInfo.memOpBarrierParamsSize;
            recordVersionInfo.memOpBarrierParamsMinorVersion = eventVersionInfo.memOpBarrierParamsMinorVersion;
            recordVersionInfo.memOpAtomicReductionParamsSize = eventVersionInfo.memOpAtomicReductionParamsSize;
            recordVersionInfo.memOpAtomicReductionParamsMinorVersion = eventVersionInfo.memOpAtomicReductionParamsMinorVersion;

            return recordVersionInfo;
        }

    private:
        struct ProviderBufferSections
        {
            const NvTrace_ProviderBufferHeader_V1* pProviderBufferHeader = nullptr;
            const NvTrace_CUDA_ApiTrace_ProviderBufferLayout_V1* pProviderBufferLayout = nullptr;
            const NvTrace_CUDA_ApiTrace_EventVersionInfo_V1* pEventVersionInfo = nullptr;
            const FB::Provider::CudaApiTrace::ProviderConfiguration* fbProviderConfiguration = nullptr;
            const NvTrace_CUDA_ApiTrace_DeviceState_V1* pDeviceStates = nullptr;
            const NvTrace_CUDA_ApiTrace_ContextState_V1* pContextStates = nullptr;
            const NvTrace_CUDA_ApiTrace_StreamState_V1* pStreamStates = nullptr;
            const NvTrace_CUDA_ApiTrace_ModuleState_V1* pModuleStates = nullptr;
            const NvTrace_CUDA_ApiTrace_FunctionState_V1* pFunctionStates = nullptr;
            const NvTrace_CUDA_ApiTrace_GraphExecState_V1* pGraphExecStates = nullptr;
            const NvTrace_CUDA_ApiTrace_GraphNodeState_V1* pGraphNodeStates = nullptr;
            const NvTrace_CUDA_ApiTrace_ChannelState_V1* pChannelStates = nullptr;
            const NvTrace_CUDA_ApiTrace_StringTableEntry_V1* pStringTableEntries = nullptr;
            const char* pStringTableData = nullptr;
            const NvTrace_CUDA_ApiTrace_StateTableStats_V1* pCurrentStateTableStats = nullptr;
            const NvTrace_CUDA_ApiTrace_StateTableStats_V1* pMaxStateTableStats = nullptr;
            const NvTrace_CircularBufferState* pGpuTimestampFifoCircularBufferStates = nullptr;    // 1:1 with streams.
            const NvTrace_CUDA_ApiTrace_GpuTimestampFifoCell_V1* pGpuTimestampFifoCells = nullptr; // Each block is 1:1 with streams.
            const NvTrace_CUDA_ApiTrace_GpuTimestampReport_V1* pGpuTimestampReports = nullptr;
        };

        ProviderBufferSections m_sections;
        NvTrace_CUDA_ApiTrace_RecordVersionInfo m_recordVersionInfo = {};

        uint32_t m_objectLifetimePayloadOffset = 0;

        uint32_t m_graphNodeStateSize = 0;
        uint32_t m_graphNodeParamsSize = 0;
        uint64_t m_graphNodeTableEntryStride = 0;
        uint32_t m_graphExecIndexBits = 0;
        uint32_t m_graphNodeIndexBits = 0;

        std::vector<CudaApiTraceGpuTimestampFifoObserverV1> m_gpuTimestampFifoObservers;

        std::vector<char> m_functionNameBuffer;
        std::vector<uint8_t> m_functionRecordBytesBuffer;
        std::vector<uint8_t> m_graphNodeRecordBytesBuffer;
    };

} // namespace nv::trace
