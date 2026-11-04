/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/DataFormats/nvtrace_common_types.h>
#include <NvTrace/DataFormats/nvtrace_cuda_api_trace.h>
#include <NvTrace/DataFormats/nvtrace_cuda_api_trace_v1.h>
#include <NvTrace/Target/NvTraceClientLogger.h>
#include <NvTrace/Target/NvTraceCudaApiTraceProviderConfiguration_generated.h>
#include <NvTrace/Target/NvTraceCudaApiTraceProviderConstants.h>
#include <NvTrace/Target/NvTraceCudaApiTraceRecordUtilities.h>
#include <NvTrace/Target/NvTraceFlatBufferUtilities.h>
#include <NvTrace/Target/NvTraceMathUtilities.h>
#include <NvTrace/Target/NvTraceMemoryFence.h>
#include <NvTrace/Target/NvTraceScopeExitGuard.h>
#include <NvTrace/Target/NvTraceUuidUtilities.h>
#include <NvTrace/Target/Observer/NvTraceCudaApiTraceConsumerTypes.h>
#include <NvTrace/Target/Observer/NvTraceCudaApiTraceObserverV1.h>
#include <NvTrace/Target/Observer/NvTraceGenericObserver.h>
#include <NvTrace/Target/Observer/NvTraceThreadEventConsumerTypes.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

namespace nv::trace {

    /// \brief CUDA API Trace Observer
    ///
    /// Typed wrapper over GenericObserver. Each state table is drained at attach and resolved by ref
    /// thereafter -- against the state table references the event stream carries -- both delivering
    /// records tentatively, each closed by its own commit, so a consumer never keeps a record a
    /// racing writer disturbed. The per-stream GPU report buffers are drained through
    /// ConsumeGpuTimestampEvents.
    ///
    /// Initialize sniffs the buffer's format: a generic (V2) buffer opens with its major version, a
    /// V1 buffer opens with the provider type UUID. A V1 buffer is served through the old-driver
    /// shim (CudaApiTraceObserverV1), which converts every V1 state struct to its V2 created-record
    /// on delivery, so consumers see one format either way: for a generic buffer the record bytes are
    /// the producer's verbatim, for a V1 buffer they are the shim's converted struct.
    ///
    /// A reference may announce the same entry more than once (a pre-attach announce drained after
    /// the attach pass already emitted it, or a stale announce resolving to a slot's new occupant);
    /// every delivery is internally consistent, so downstream keying on (id, revision) must be
    /// idempotent.
    ///
    /// Thread safety: not internally synchronized; one consumer thread. The producer may be concurrent.
    class CudaApiTraceObserver
    {
    public:
        using FbConfigurationView = FlatBufferView<FB::Provider::CudaApiTrace::ProviderConfiguration>;
        using GraphNodeRecord = CudaApiTraceGraphNodeRecord;
        using ThreadEventRecordStorage = CudaApiTraceObserverV1::ThreadEventRecordStorage;

        ~CudaApiTraceObserver() = default;
        CudaApiTraceObserver() = default;

        CudaApiTraceObserver(const CudaApiTraceObserver&) = delete;
        CudaApiTraceObserver(CudaApiTraceObserver&& rhs) noexcept
        {
            *this = std::move(rhs);
        }

        CudaApiTraceObserver& operator=(const CudaApiTraceObserver&) = delete;
        CudaApiTraceObserver& operator=(CudaApiTraceObserver&& rhs) noexcept
        {
            if (this != &rhs)
            {
                m_genericObserver = std::move(rhs.m_genericObserver);
                m_recordVersionInfo = std::exchange(rhs.m_recordVersionInfo, {});
                m_configurationView = std::exchange(rhs.m_configurationView, {});
                m_pProviderStatistics = std::exchange(rhs.m_pProviderStatistics, nullptr);
                m_pContexts = std::exchange(rhs.m_pContexts, &s_emptyStateTable);
                m_pStreams = std::exchange(rhs.m_pStreams, &s_emptyStateTable);
                m_pModules = std::exchange(rhs.m_pModules, &s_emptyStateTable);
                m_pGraphExecs = std::exchange(rhs.m_pGraphExecs, &s_emptyStateTable);
                m_pGraphNodes = std::exchange(rhs.m_pGraphNodes, &s_emptyStateTable);
                m_pChannels = std::exchange(rhs.m_pChannels, &s_emptyStateTable);
                m_deviceBlobs = std::move(rhs.m_deviceBlobs);
                m_gpuReportBuffers = std::move(rhs.m_gpuReportBuffers);
                m_stagedGpuTimestampReports = std::move(rhs.m_stagedGpuTimestampReports);
                m_graphNodeParamsOffsetInPayload = std::exchange(rhs.m_graphNodeParamsOffsetInPayload, 0);
                m_v1Observer = std::move(rhs.m_v1Observer);
            }
            return *this;
        }

        void Reset() noexcept
        {
            m_v1Observer.Reset();
            m_graphNodeParamsOffsetInPayload = 0;
            m_stagedGpuTimestampReports.clear();
            m_gpuReportBuffers.clear();
            m_deviceBlobs.clear();
            m_pChannels = &s_emptyStateTable;
            m_pGraphNodes = &s_emptyStateTable;
            m_pGraphExecs = &s_emptyStateTable;
            m_pModules = &s_emptyStateTable;
            m_pStreams = &s_emptyStateTable;
            m_pContexts = &s_emptyStateTable;
            m_pProviderStatistics = nullptr;
            m_configurationView = {};
            m_recordVersionInfo = {};
            m_genericObserver.Reset();
        }

        // providerTypeUuid arrives out-of-band from the Router, which took it from the provider's
        // advertisement: the V1 id means an old driver serving a V1 buffer. The generic path
        // re-reads the id from the buffer and requires it to match.
        bool Initialize(const NVTRACE_UUID& providerTypeUuid, const uint8_t* pMappedProviderBufferMemory, size_t providerBufferSize)
        {
            if (providerTypeUuid == NvTrace_CUDA_ApiTrace_ProviderTypeUuid_V1)
            {
                return m_v1Observer.Initialize(pMappedProviderBufferMemory, providerBufferSize);
            }

            auto errorGuard = ScopeExitGuard([&]() {
                NVTRACE_LOG_ERR(20, "CudaApiTraceObserver::Initialize failed (provider buffer=%p providerBufferSize=%zu).", static_cast<const void*>(pMappedProviderBufferMemory), providerBufferSize);
                Reset();
            });

            const bool genericObserverInitialized = m_genericObserver.Initialize(pMappedProviderBufferMemory, providerBufferSize);
            if (!genericObserverInitialized)
            {
                return false;
            }

            const NVTRACE_UUID bufferProviderTypeUuid = m_genericObserver.ProviderTypeUuid();
            NVTRACE_LOG_ERR_RETURN_IF(bufferProviderTypeUuid != providerTypeUuid, false, 20, "Provider type UUID mismatch for CUDA API Trace observer (buffer %s, router %s).", ToString(bufferProviderTypeUuid).c_str(), ToString(providerTypeUuid).c_str());

            DataBlobReader* const pRecordVersionInfoReader = FindDataBlobReader(NvTrace_CUDA_ApiTrace_SectionTag_RecordVersionInfo);
            NVTRACE_LOG_ERR_RETURN_IF(!pRecordVersionInfoReader, false, 20, "CUDA API Trace record version info section not found or invalid.");

            static_assert((sizeof(NvTrace_CUDA_ApiTrace_RecordVersionInfo) % sizeof(uint64_t)) == 0, "Memcpy64 copies whole words only");
            alignas(uint64_t) NvTrace_CUDA_ApiTrace_RecordVersionInfo recordVersionInfo = {};
            size_t recordVersionInfoPayloadSize = 0;
            pRecordVersionInfoReader->ReadRecord([&](uint32_t /* recordType */, const uint8_t* pPayload, size_t payloadSize) {
                recordVersionInfoPayloadSize = payloadSize;
                Memcpy64(&recordVersionInfo, pPayload, std::min(sizeof(recordVersionInfo), payloadSize));
            });
            NVTRACE_LOG_ERR_RETURN_IF(recordVersionInfoPayloadSize < NvTrace_CUDA_ApiTrace_RecordVersionInfo_MINOR_VERSION_A, false, 20, "Record version info is incompatible for CUDA API Trace (publishedSize=%zu expectedMinorVersion=%u).", recordVersionInfoPayloadSize, NvTrace_CUDA_ApiTrace_RecordVersionInfo_MINOR_VERSION_A);
            NVTRACE_LOG_ERR_RETURN_IF(recordVersionInfo.recordVersionInfoMinorVersion < NvTrace_CUDA_ApiTrace_RecordVersionInfo_MINOR_VERSION_A, false, 20, "Record version info describes itself below the consumer's floor (recordVersionInfoMinorVersion=%u expectedMinorVersion=%u).", recordVersionInfo.recordVersionInfoMinorVersion, NvTrace_CUDA_ApiTrace_RecordVersionInfo_MINOR_VERSION_A);

            // What flows downstream (GetRecordVersionInfoByteSpan) is this build's extent of the struct,
            // so the self-description must not claim a newer producer's larger shape over those bytes.
            recordVersionInfo.recordVersionInfoSize = std::min(recordVersionInfo.recordVersionInfoSize, static_cast<uint32_t>(sizeof(NvTrace_CUDA_ApiTrace_RecordVersionInfo)));
            recordVersionInfo.recordVersionInfoMinorVersion = std::min(recordVersionInfo.recordVersionInfoMinorVersion, static_cast<uint32_t>(NvTrace_CUDA_ApiTrace_RecordVersionInfo_MINOR_VERSION_A));

            // Gate only the records this observer itself decodes; the event payloads pass through to
            // the consumers, which size-check each record as they always have.
            NVTRACE_LOG_ERR_RETURN_IF(recordVersionInfo.deviceCreatedMinorVersion < NvTrace_CUDA_ApiTrace_DeviceCreated_MINOR_VERSION_A, false, 20, "Device created record layout is incompatible (minorVersion=%u expected=%u).", recordVersionInfo.deviceCreatedMinorVersion, NvTrace_CUDA_ApiTrace_DeviceCreated_MINOR_VERSION_A);
            NVTRACE_LOG_ERR_RETURN_IF(recordVersionInfo.contextCreatedMinorVersion < NvTrace_CUDA_ApiTrace_ContextCreated_MINOR_VERSION_A, false, 20, "Context created record layout is incompatible (minorVersion=%u expected=%u).", recordVersionInfo.contextCreatedMinorVersion, NvTrace_CUDA_ApiTrace_ContextCreated_MINOR_VERSION_A);
            NVTRACE_LOG_ERR_RETURN_IF(recordVersionInfo.streamCreatedMinorVersion < NvTrace_CUDA_ApiTrace_StreamCreated_MINOR_VERSION_A, false, 20, "Stream created record layout is incompatible (minorVersion=%u expected=%u).", recordVersionInfo.streamCreatedMinorVersion, NvTrace_CUDA_ApiTrace_StreamCreated_MINOR_VERSION_A);
            NVTRACE_LOG_ERR_RETURN_IF(recordVersionInfo.moduleLoadedMinorVersion < NvTrace_CUDA_ApiTrace_ModuleLoaded_MINOR_VERSION_A, false, 20, "Module loaded record layout is incompatible (minorVersion=%u expected=%u).", recordVersionInfo.moduleLoadedMinorVersion, NvTrace_CUDA_ApiTrace_ModuleLoaded_MINOR_VERSION_A);
            NVTRACE_LOG_ERR_RETURN_IF(recordVersionInfo.functionCreatedMinorVersion < NvTrace_CUDA_ApiTrace_FunctionCreated_MINOR_VERSION_A, false, 20, "Function created record layout is incompatible (minorVersion=%u expected=%u).", recordVersionInfo.functionCreatedMinorVersion, NvTrace_CUDA_ApiTrace_FunctionCreated_MINOR_VERSION_A);
            NVTRACE_LOG_ERR_RETURN_IF(recordVersionInfo.functionNameSegmentMinorVersion < NvTrace_CUDA_ApiTrace_FunctionNameSegment_MINOR_VERSION_A, false, 20, "Function name segment record layout is incompatible (minorVersion=%u expected=%u).", recordVersionInfo.functionNameSegmentMinorVersion, NvTrace_CUDA_ApiTrace_FunctionNameSegment_MINOR_VERSION_A);
            NVTRACE_LOG_ERR_RETURN_IF(recordVersionInfo.graphExecCreatedMinorVersion < NvTrace_CUDA_ApiTrace_GraphExecCreated_MINOR_VERSION_A, false, 20, "Graph exec created record layout is incompatible (minorVersion=%u expected=%u).", recordVersionInfo.graphExecCreatedMinorVersion, NvTrace_CUDA_ApiTrace_GraphExecCreated_MINOR_VERSION_A);
            NVTRACE_LOG_ERR_RETURN_IF(recordVersionInfo.graphNodeCreatedMinorVersion < NvTrace_CUDA_ApiTrace_GraphNodeCreated_MINOR_VERSION_A, false, 20, "Graph node created record layout is incompatible (minorVersion=%u expected=%u).", recordVersionInfo.graphNodeCreatedMinorVersion, NvTrace_CUDA_ApiTrace_GraphNodeCreated_MINOR_VERSION_A);
            NVTRACE_LOG_ERR_RETURN_IF(recordVersionInfo.channelCreatedMinorVersion < NvTrace_CUDA_ApiTrace_ChannelCreated_MINOR_VERSION_A, false, 20, "Channel created record layout is incompatible (minorVersion=%u expected=%u).", recordVersionInfo.channelCreatedMinorVersion, NvTrace_CUDA_ApiTrace_ChannelCreated_MINOR_VERSION_A);
            NVTRACE_LOG_ERR_RETURN_IF(recordVersionInfo.gpuTimestampReportMinorVersion < NvTrace_CUDA_ApiTrace_GpuTimestampReport_MINOR_VERSION_A, false, 20, "GPU timestamp report layout is incompatible (minorVersion=%u expected=%u).", recordVersionInfo.gpuTimestampReportMinorVersion, NvTrace_CUDA_ApiTrace_GpuTimestampReport_MINOR_VERSION_A);

            // The graph node params are lifted with Memcpy64, which reads 64-bit words, so that offset
            // must also be 8-aligned regardless of what the params themselves want; downstream readers
            // rely on this one check for it.
            NVTRACE_LOG_ERR_RETURN_IF(recordVersionInfo.graphNodeCreatedSize < recordVersionInfo.graphNodeCreatedMinorVersion || (recordVersionInfo.graphNodeCreatedSize % alignof(uint64_t)) != 0, false, 20, "Graph node created record size cannot lead a composite record (size=%u minorVersion=%u).", recordVersionInfo.graphNodeCreatedSize, recordVersionInfo.graphNodeCreatedMinorVersion);
            NVTRACE_LOG_ERR_RETURN_IF(recordVersionInfo.functionCreatedSize < recordVersionInfo.functionCreatedMinorVersion, false, 20, "Function created record size cannot lead its inline name characters (size=%u minorVersion=%u).", recordVersionInfo.functionCreatedSize, recordVersionInfo.functionCreatedMinorVersion);
            NVTRACE_LOG_ERR_RETURN_IF(recordVersionInfo.functionNameSegmentSize < recordVersionInfo.functionNameSegmentMinorVersion, false, 20, "Function name segment record size cannot lead its name characters (size=%u minorVersion=%u).", recordVersionInfo.functionNameSegmentSize, recordVersionInfo.functionNameSegmentMinorVersion);

            DataBlobReader* const pConfigurationReader = FindDataBlobReader(NvTrace_CUDA_ApiTrace_SectionTag_Configuration);
            NVTRACE_LOG_ERR_RETURN_IF(!pConfigurationReader, false, 20, "CUDA API Trace configuration section not found or invalid.");

            FbConfigurationView configurationView = {};
            pConfigurationReader->ReadRecord([&](uint32_t /* recordType */, const uint8_t* pPayload, size_t payloadSize) {
                configurationView = {pPayload, payloadSize};
            });

            // The statistics blob is best-effort telemetry; a buffer without one still attaches.
            const NvTrace_CUDA_ApiTrace_ProviderStatistics* pProviderStatistics = nullptr;
            if (DataBlobReader* const pProviderStatisticsReader = FindDataBlobReader(NvTrace_CUDA_ApiTrace_SectionTag_ProviderStatistics))
            {
                pProviderStatisticsReader->ReadRecord([&](uint32_t /* recordType */, const uint8_t* pPayload, size_t payloadSize) {
                    if (recordVersionInfo.providerStatisticsMinorVersion < NvTrace_CUDA_ApiTrace_ProviderStatistics_MINOR_VERSION_A || payloadSize < NvTrace_CUDA_ApiTrace_ProviderStatistics_MINOR_VERSION_A)
                    {
                        return;
                    }
                    pProviderStatistics = reinterpret_cast<const NvTrace_CUDA_ApiTrace_ProviderStatistics*>(pPayload);
                });
            }

            const StateTableReader* const pContexts = FindStateTableReader(NvTrace_CUDA_ApiTrace_SectionTag_Contexts);
            const StateTableReader* const pStreams = FindStateTableReader(NvTrace_CUDA_ApiTrace_SectionTag_Streams);
            const StateTableReader* const pModules = FindStateTableReader(NvTrace_CUDA_ApiTrace_SectionTag_Modules);
            const StateTableReader* const pGraphExecs = FindStateTableReader(NvTrace_CUDA_ApiTrace_SectionTag_GraphExecs);
            const StateTableReader* const pGraphNodes = FindStateTableReader(NvTrace_CUDA_ApiTrace_SectionTag_GraphNodes);
            const StateTableReader* const pChannels = FindStateTableReader(NvTrace_CUDA_ApiTrace_SectionTag_Channels);
            if (!pContexts || !pStreams || !pModules || !pGraphExecs || !pGraphNodes || !pChannels)
            {
                return false;
            }

            std::vector<DataBlobReader*> deviceBlobs = CollectDeviceBlobs();
            NVTRACE_LOG_ERR_RETURN_IF(deviceBlobs.empty(), false, 20, "CUDA API Trace device blobs not found or invalid.");

            std::vector<GpuReportBufferReader*> gpuReportBuffers = CollectGpuReportBuffers();

            // Commit observer state only now, after all validation has succeeded.
            m_recordVersionInfo = recordVersionInfo;
            m_configurationView = configurationView;
            m_pProviderStatistics = pProviderStatistics;
            m_pContexts = pContexts;
            m_pStreams = pStreams;
            m_pModules = pModules;
            m_pGraphExecs = pGraphExecs;
            m_pGraphNodes = pGraphNodes;
            m_pChannels = pChannels;
            m_deviceBlobs = std::move(deviceBlobs);
            m_gpuReportBuffers = std::move(gpuReportBuffers);
            m_graphNodeParamsOffsetInPayload = recordVersionInfo.graphNodeCreatedSize;

            errorGuard.Dismiss();
            NVTRACE_LOG_INF(90, "Initialized successfully.");
            return true;
        }

        bool IsInitialized() const
        {
            return m_genericObserver.IsInitialized() || m_v1Observer.IsInitialized();
        }

        uint32_t ThreadEventSourceProviderIndex() const
        {
            return m_v1Observer.IsInitialized() ? m_v1Observer.ThreadEventSourceProviderIndex() : m_genericObserver.ThreadEventSourceProviderIndex();
        }

        bool IsProviderMeasuring() const
        {
            if (m_v1Observer.IsInitialized())
            {
                return m_v1Observer.IsProviderMeasuring();
            }
            return m_genericObserver.ProviderStatus() == NvTrace_ProviderStatus_measuring;
        }

        const NvTrace_CUDA_ApiTrace_RecordVersionInfo& GetRecordVersionInfo() const
        {
            return m_v1Observer.IsInitialized() ? m_v1Observer.RecordVersionInfo() : m_recordVersionInfo;
        }

        // The version info the codec wire embeds in ProviderCreated records, gating every record the
        // offline DataModel decodes: the producer's own blob for a generic buffer, the shim's
        // synthesized one for a V1 buffer.
        NvTrace_ByteSpan GetRecordVersionInfoByteSpan() const
        {
            return {reinterpret_cast<const uint8_t*>(&GetRecordVersionInfo()), sizeof(NvTrace_CUDA_ApiTrace_RecordVersionInfo)};
        }

        FbConfigurationView GetProviderConfigurationFlatBufferView() const
        {
            return m_v1Observer.IsInitialized() ? m_v1Observer.GetProviderConfigurationFlatBufferView() : m_configurationView;
        }

        const FB::Provider::CudaApiTrace::ProviderConfiguration* GetProviderConfiguration() const
        {
            if (m_v1Observer.IsInitialized())
            {
                return m_v1Observer.GetProviderConfiguration();
            }
            return m_configurationView.pData ? m_configurationView.GetRoot() : nullptr;
        }

        // A V1 provider keeps no dropped-function, dropped-name or wedged-stream counters; those read as 0.
        bool ReadProviderStatistics(NvTrace_CUDA_ApiTrace_ProviderStatistics& out_providerStatistics) const
        {
            if (m_v1Observer.IsInitialized())
            {
                return m_v1Observer.ReadProviderStatistics(out_providerStatistics);
            }

            if (!m_pProviderStatistics)
            {
                return false;
            }

            Memcpy64(&out_providerStatistics, m_pProviderStatistics, sizeof(out_providerStatistics));
            return true;
        }

        uint32_t GetDeviceStateCount() const
        {
            return m_v1Observer.IsInitialized() ? m_v1Observer.GetDeviceStateCount() : static_cast<uint32_t>(m_deviceBlobs.size());
        }

        uint32_t GetContextStateCount() const
        {
            return m_v1Observer.IsInitialized() ? m_v1Observer.GetContextStateCount() : m_pContexts->EntryCount();
        }

        uint32_t GetStreamStateCount() const
        {
            return m_v1Observer.IsInitialized() ? m_v1Observer.GetStreamStateCount() : m_pStreams->EntryCount();
        }

        uint32_t GetModuleStateCount() const
        {
            return m_v1Observer.IsInitialized() ? m_v1Observer.GetModuleStateCount() : m_pModules->EntryCount();
        }

        uint32_t GetGraphExecStateCount() const
        {
            return m_v1Observer.IsInitialized() ? m_v1Observer.GetGraphExecStateCount() : m_pGraphExecs->EntryCount();
        }

        uint32_t GetGraphNodeStateCount() const
        {
            return m_v1Observer.IsInitialized() ? m_v1Observer.GetGraphNodeStateCount() : m_pGraphNodes->EntryCount();
        }

        uint32_t GetChannelStateCount() const
        {
            return m_v1Observer.IsInitialized() ? m_v1Observer.GetChannelStateCount() : m_pChannels->EntryCount();
        }

        // The attach drains: each delivers every published entry of one table, tentatively. fnOnRecord
        // stages a record and the fnCommitRecord that follows renders the verdict on it -- true to
        // keep it, false to drop it. A slot that was never published, or that a writer holds
        // mid-update, delivers nothing and so commits nothing.
        //
        // fnOnRecord(uint32_t entryIndex, uint64_t entryRevision, const TRecord& record)
        // fnCommitRecord(bool consistencyCheckPassed)
        template <typename TFN_OnRecord, typename TFN_CommitRecord>
        void ConsumeAttachDevices(TFN_OnRecord&& fnOnRecord, TFN_CommitRecord&& fnCommitRecord)
        {
            if (m_v1Observer.IsInitialized())
            {
                for (uint32_t entryIndex = 0; entryIndex < m_v1Observer.GetDeviceStateCount(); ++entryIndex)
                {
                    m_v1Observer.ReadDeviceState(entryIndex, fnOnRecord, fnCommitRecord);
                }
                return;
            }

            // A device is a data blob, not a table entry: its index is its position among the blobs, it
            // has no revision, and its record is static, so it always commits.
            ConsumeAttachDeviceRecordBytes(
                [&](uint32_t deviceIndex, const uint8_t* pRecordBytes, size_t recordByteCount) {
                    NvTrace_CUDA_ApiTrace_DeviceCreated record = {};
                    Memcpy64(&record, pRecordBytes, std::min(sizeof(record), AlignDnP2(recordByteCount, sizeof(uint64_t))));
                    fnOnRecord(deviceIndex, 0, record);
                },
                fnCommitRecord);
        }

        // The device records as the codec wire carries them. fnCommitRecord(bool) follows each; a
        // blob's verdict is always true.
        // fnOnRecordBytes(uint32_t deviceIndex, const uint8_t* pRecordBytes, size_t recordByteCount)
        template <typename TFN_OnRecordBytes, typename TFN_CommitRecord>
        void ConsumeAttachDeviceRecordBytes(TFN_OnRecordBytes&& fnOnRecordBytes, TFN_CommitRecord&& fnCommitRecord)
        {
            if (m_v1Observer.IsInitialized())
            {
                for (uint32_t entryIndex = 0; entryIndex < m_v1Observer.GetDeviceStateCount(); ++entryIndex)
                {
                    m_v1Observer.ReadDeviceState(
                        entryIndex,
                        [&](uint32_t readEntryIndex, uint64_t /* entryRevision */, const NvTrace_CUDA_ApiTrace_DeviceCreated& record) { fnOnRecordBytes(readEntryIndex, reinterpret_cast<const uint8_t*>(&record), sizeof(record)); },
                        fnCommitRecord);
                }
                return;
            }

            for (uint32_t deviceIndex = 0; deviceIndex < m_deviceBlobs.size(); ++deviceIndex)
            {
                m_deviceBlobs[deviceIndex]->ReadRecord([&](uint32_t /* recordType */, const uint8_t* pPayload, size_t payloadSize) {
                    fnOnRecordBytes(deviceIndex, pPayload, payloadSize);
                });
                fnCommitRecord(true);
            }
        }

        template <typename TFN_OnRecord, typename TFN_CommitRecord>
        void ConsumeAttachContexts(TFN_OnRecord&& fnOnRecord, TFN_CommitRecord&& fnCommitRecord)
        {
            if (m_v1Observer.IsInitialized())
            {
                for (uint32_t entryIndex = 0; entryIndex < m_v1Observer.GetContextStateCount(); ++entryIndex)
                {
                    m_v1Observer.ReadContextState(entryIndex, fnOnRecord, fnCommitRecord);
                }
                return;
            }
            ConsumeAttachStateTableRecords<NvTrace_CUDA_ApiTrace_ContextCreated>(*m_pContexts, NvTrace_CUDA_ApiTrace_RecordType_ContextCreated, fnOnRecord, fnCommitRecord);
        }

        template <typename TFN_OnRecord, typename TFN_CommitRecord>
        void ConsumeAttachStreams(TFN_OnRecord&& fnOnRecord, TFN_CommitRecord&& fnCommitRecord)
        {
            if (m_v1Observer.IsInitialized())
            {
                for (uint32_t entryIndex = 0; entryIndex < m_v1Observer.GetStreamStateCount(); ++entryIndex)
                {
                    m_v1Observer.ReadStreamState(entryIndex, fnOnRecord, fnCommitRecord);
                }
                return;
            }
            ConsumeAttachStateTableRecords<NvTrace_CUDA_ApiTrace_StreamCreated>(*m_pStreams, NvTrace_CUDA_ApiTrace_RecordType_StreamCreated, fnOnRecord, fnCommitRecord);
        }

        template <typename TFN_OnRecord, typename TFN_CommitRecord>
        void ConsumeAttachModules(TFN_OnRecord&& fnOnRecord, TFN_CommitRecord&& fnCommitRecord)
        {
            if (m_v1Observer.IsInitialized())
            {
                for (uint32_t entryIndex = 0; entryIndex < m_v1Observer.GetModuleStateCount(); ++entryIndex)
                {
                    m_v1Observer.ReadModuleState(entryIndex, fnOnRecord, fnCommitRecord);
                }
                return;
            }
            ConsumeAttachStateTableRecords<NvTrace_CUDA_ApiTrace_ModuleLoaded>(*m_pModules, NvTrace_CUDA_ApiTrace_RecordType_ModuleLoaded, fnOnRecord, fnCommitRecord);
        }

        template <typename TFN_OnRecord, typename TFN_CommitRecord>
        void ConsumeAttachGraphExecs(TFN_OnRecord&& fnOnRecord, TFN_CommitRecord&& fnCommitRecord)
        {
            if (m_v1Observer.IsInitialized())
            {
                for (uint32_t entryIndex = 0; entryIndex < m_v1Observer.GetGraphExecStateCount(); ++entryIndex)
                {
                    m_v1Observer.ReadGraphExecState(entryIndex, fnOnRecord, fnCommitRecord);
                }
                return;
            }
            ConsumeAttachStateTableRecords<NvTrace_CUDA_ApiTrace_GraphExecCreated>(*m_pGraphExecs, NvTrace_CUDA_ApiTrace_RecordType_GraphExecCreated, fnOnRecord, fnCommitRecord);
        }

        template <typename TFN_OnRecord, typename TFN_CommitRecord>
        void ConsumeAttachChannels(TFN_OnRecord&& fnOnRecord, TFN_CommitRecord&& fnCommitRecord)
        {
            if (m_v1Observer.IsInitialized())
            {
                for (uint32_t entryIndex = 0; entryIndex < m_v1Observer.GetChannelStateCount(); ++entryIndex)
                {
                    m_v1Observer.ReadChannelState(entryIndex, fnOnRecord, fnCommitRecord);
                }
                return;
            }
            ConsumeAttachStateTableRecords<NvTrace_CUDA_ApiTrace_ChannelCreated>(*m_pChannels, NvTrace_CUDA_ApiTrace_RecordType_ChannelCreated, fnOnRecord, fnCommitRecord);
        }

        // Drains every state section's records verbatim, in section order, each followed by
        // fnCommitRecord(bool). Every record names what it belongs to in its own payload, so the
        // drain carries no grouping and needs no knowledge of which record type continues which.
        // fnReadRecord(NvTrace_CorrelationId recordCorrelationId, uint32_t recordType, const uint8_t* pRecordBytes, size_t recordByteCount)
        //   -- the record verbatim, valid only during the call.
        template <typename TFN_ReadRecord, typename TFN_CommitRecord>
        void ConsumeAttachStateRecords(TFN_ReadRecord&& fnReadRecord, TFN_CommitRecord&& fnCommitRecord)
        {
            if (m_v1Observer.IsInitialized())
            {
                m_v1Observer.ConsumeAttachStateRecords(fnReadRecord, fnCommitRecord);
                return;
            }

            m_genericObserver.ConsumeAttachStateRecords(fnReadRecord, fnCommitRecord);
        }

        // Decodes the entry identity from a ModuleLoaded entry record's correlation ID as delivered
        // by ConsumeAttachStateRecords. Chain record correlation IDs are buffer addresses, not entry
        // refs, so this is not meaningful for FunctionCreated or FunctionNameSegment records.
        uint32_t DecodeModuleEntryIndex(NvTrace_CorrelationId moduleEntryCorrelationId) const
        {
            if (m_v1Observer.IsInitialized())
            {
                return m_v1Observer.DecodeModuleEntryIndex(moduleEntryCorrelationId);
            }
            return m_pModules->DecodeEntryRef(moduleEntryCorrelationId).first;
        }

        uint64_t DecodeModuleEntryRevision(NvTrace_CorrelationId moduleEntryCorrelationId) const
        {
            if (m_v1Observer.IsInitialized())
            {
                return m_v1Observer.DecodeModuleEntryRevision(moduleEntryCorrelationId);
            }
            return m_pModules->DecodeEntryRef(moduleEntryCorrelationId).second;
        }

        // fnOnRecord(uint32_t entryIndex, uint64_t entryRevision, const GraphNodeRecord& record)
        template <typename TFN_OnRecord, typename TFN_CommitRecord>
        void ConsumeAttachGraphNodes(TFN_OnRecord&& fnOnRecord, TFN_CommitRecord&& fnCommitRecord)
        {
            if (m_v1Observer.IsInitialized())
            {
                for (uint32_t entryIndex = 0; entryIndex < m_v1Observer.GetGraphNodeStateCount(); ++entryIndex)
                {
                    m_v1Observer.ReadGraphNodeState(entryIndex, fnOnRecord, fnCommitRecord);
                }
                return;
            }

            m_pGraphNodes->ConsumeAttachRecords<false>(
                [&](uint64_t entryRef, uint32_t recordType, const uint8_t* pPayload, size_t payloadSize) {
                    if (recordType != NvTrace_CUDA_ApiTrace_RecordType_GraphNodeCreated)
                    {
                        return;
                    }

                    GraphNodeRecord record = {};
                    DecodeGraphNodeRecord(pPayload, payloadSize, record);

                    const auto [entryIndex, entryRevision] = m_pGraphNodes->DecodeEntryRef(entryRef);
                    fnOnRecord(entryIndex, entryRevision, record);
                },
                fnCommitRecord);
        }

        // Drains the graph-node table as the raw composite record bytes the codec wire carries. The
        // bytes are only valid for the duration of the call.
        // fnOnRecordBytes(NvTrace_CorrelationId graphNodeCorrelationId, const uint8_t* pRecordBytes, size_t recordByteCount)
        template <typename TFN_OnRecordBytes, typename TFN_CommitRecord>
        void ConsumeAttachGraphNodeRecordBytes(TFN_OnRecordBytes&& fnOnRecordBytes, TFN_CommitRecord&& fnCommitRecord)
        {
            if (m_v1Observer.IsInitialized())
            {
                for (uint32_t entryIndex = 0; entryIndex < m_v1Observer.GetGraphNodeStateCount(); ++entryIndex)
                {
                    m_v1Observer.ReadGraphNodeRecordBytes(
                        entryIndex,
                        [&](uint32_t readEntryIndex, uint64_t entryRevision, const uint8_t* pRecordBytes, size_t recordByteCount) {
                            fnOnRecordBytes(m_v1Observer.EncodeGraphNodeCorrelationId(entryRevision, readEntryIndex), pRecordBytes, recordByteCount);
                        },
                        fnCommitRecord);
                }
                return;
            }

            m_pGraphNodes->ConsumeAttachRecords(
                [&](uint64_t entryRef, uint32_t recordType, const uint8_t* pPayload, size_t payloadSize) {
                    if (recordType != NvTrace_CUDA_ApiTrace_RecordType_GraphNodeCreated)
                    {
                        return;
                    }

                    // 64-bit-wide reads per the in-place-store contract; a record spans whole 8-byte
                    // words by construction, so aligning down trims nothing.
                    const size_t copySize = AlignDnP2(payloadSize, sizeof(uint64_t));
                    if (copySize < m_graphNodeParamsOffsetInPayload)
                    {
                        return;
                    }

                    fnOnRecordBytes(entryRef, pPayload, copySize);
                },
                fnCommitRecord);
        }

        // Resolves a record reference the event stream carries against this observer's tables,
        // delivering the entry's whole group; the bytes stay tentative until fnCommitRecord(true). A
        // nested reference in the group is dropped: resolution is one hop.
        // fnReadRecord(uint64_t entryRevision, uint32_t recordType, const uint8_t* pPayload, size_t payloadSize)
        template <typename TFN_ReadRecord, typename TFN_CommitRecord>
        void ResolveRecordReference(const NvTrace_RecordReference& reference, TFN_ReadRecord&& fnReadRecord, TFN_CommitRecord&& fnCommitRecord)
        {
            if (m_v1Observer.IsInitialized())
            {
                m_v1Observer.ResolveRecordReference(reference, std::forward<TFN_ReadRecord>(fnReadRecord), std::forward<TFN_CommitRecord>(fnCommitRecord));
                return;
            }

            const StateTableReader* const pStateTable = FindStateTableReaderBySectionIndex(reference.sectionIndex);
            if (!pStateTable)
            {
                return;
            }

            // Only the entry record's delivery carries the entry ref -- a chain record's first argument
            // is its own correlation id -- but a read keyed by a ref only ever delivers the entry that
            // ref names, so the revision is the reference's own.
            const uint64_t entryRevision = pStateTable->DecodeEntryRef(reference.recordRef).second;
            pStateTable->ReadEntry(
                reference.recordRef,
                [&](uint64_t /* entryRefOrCorrelationId */, uint32_t resolvedRecordType, const uint8_t* pResolvedPayload, size_t resolvedPayloadSize) {
                    if (resolvedRecordType == NvTrace_ReservedRecordType_RecordReference)
                    {
                        return;
                    }
                    fnReadRecord(entryRevision, resolvedRecordType, pResolvedPayload, resolvedPayloadSize);
                },
                fnCommitRecord);
        }

        // A generic provider announces every node itself; only the shim has anything to deliver.
        // fnOnRecordBytes(uint32_t entryIndex, uint64_t entryRevision, const uint8_t* pRecordBytes, size_t recordByteCount)
        template <typename TFN_OnRecordBytes, typename TFN_CommitRecord>
        void ResolveGraphExecNodeRecordBytes(const NvTrace_RecordReference& graphExecReference, TFN_OnRecordBytes&& fnOnRecordBytes, TFN_CommitRecord&& fnCommitRecord)
        {
            if (m_v1Observer.IsInitialized())
            {
                m_v1Observer.ResolveGraphExecNodeRecordBytes(graphExecReference, std::forward<TFN_OnRecordBytes>(fnOnRecordBytes), std::forward<TFN_CommitRecord>(fnCommitRecord));
            }
        }

        // Judges the record type as read, before TranslateThreadEventRecord.
        bool IsObjectLifetimeThreadEventRecord(uint32_t recordType) const
        {
            if (m_v1Observer.IsInitialized())
            {
                return m_v1Observer.IsObjectLifetimeThreadEventRecord(recordType);
            }

            return recordType == NvTrace_ReservedRecordType_RecordReference
                || recordType == NvTrace_CUDA_ApiTrace_RecordType_ContextDestroyed
                || recordType == NvTrace_CUDA_ApiTrace_RecordType_StreamDestroyed
                || recordType == NvTrace_CUDA_ApiTrace_RecordType_ModuleUnloaded
                || recordType == NvTrace_CUDA_ApiTrace_RecordType_GraphExecDestroyed;
        }

        // The result may point into storage, which must outlive it.
        const ThreadEventRecord* TranslateThreadEventRecord(const ThreadEventRecord& record, ThreadEventRecordStorage& storage) const
        {
            if (m_v1Observer.IsInitialized())
            {
                return m_v1Observer.TranslateThreadEventRecord(record, storage);
            }

            return &record;
        }

        // Consumes all available GPU timestamp reports across all streams.
        // void fnOnTimestampEvent(uint64_t gpuTimestampRef, uint64_t ptimer);
        template <typename TFN_OnTimestampEvent>
        void ConsumeGpuTimestampEvents(TFN_OnTimestampEvent&& fnOnTimestampEvent)
        {
            if (m_v1Observer.IsInitialized())
            {
                m_v1Observer.ConsumeGpuTimestampEvents(std::forward<TFN_OnTimestampEvent>(fnOnTimestampEvent));
                return;
            }

            for (GpuReportBufferReader* const pReader : m_gpuReportBuffers)
            {
                m_stagedGpuTimestampReports.clear();
                pReader->ConsumeRecords(
                    [&](uint32_t /* recordType */, const uint8_t* pPayload, size_t payloadSize) {
                        if (payloadSize < sizeof(NvTrace_CUDA_ApiTrace_GpuTimestampReport))
                        {
                            return;
                        }

                        // Staged, not forwarded: a report the reader delivered is only proven
                        // untouched by the group verdict's chunk-revision check.
                        NvTrace_CUDA_ApiTrace_GpuTimestampReport report;
                        Memcpy64(&report, pPayload, sizeof(report));
                        m_stagedGpuTimestampReports.push_back(report);
                    },
                    [&](bool consistencyCheckPassed) {
                        if (consistencyCheckPassed)
                        {
                            for (const NvTrace_CUDA_ApiTrace_GpuTimestampReport& report : m_stagedGpuTimestampReports)
                            {
                                fnOnTimestampEvent(report.reportRef, report.ptimer);
                            }
                        }
                        m_stagedGpuTimestampReports.clear();
                    });
            }
        }

        //   void fnOnTimestampEvent(uint64_t gpuTimestampRef, uint64_t ptimer)
        //     Accepts every delivered report.
        //
        //   bool fnOnTimestampEvent(uint64_t gpuTimestampRef, uint64_t ptimer)
        //     Return true to accept the report and advance the cursor.
        //     Return false to retain the report and stop until the next ConsumeGpuTimestampEvents().
        //
        //   void fnCommitTimestampEvent(bool consistencyCheckPassed)
        //     Called once per accepted report. If false, roll back state staged by fnOnTimestampEvent.
        template <typename TFN_OnTimestampEvent, typename TFN_CommitTimestampEvent>
        void ConsumeGpuTimestampEvents(TFN_OnTimestampEvent&& fnOnTimestampEvent, TFN_CommitTimestampEvent&& fnCommitTimestampEvent)
        {
            bool recordForwarded = false;
            const auto fnCommitForwardedRecord = [&](bool consistencyCheckPassed) {
                if (std::exchange(recordForwarded, false))
                {
                    fnCommitTimestampEvent(consistencyCheckPassed);
                }
            };
            const auto fnForwardRecord = [&](uint64_t gpuTimestampRef, uint64_t ptimer) {
                if constexpr (std::is_same_v<std::invoke_result_t<TFN_OnTimestampEvent, uint64_t, uint64_t>, void>)
                {
                    recordForwarded = true;
                    fnOnTimestampEvent(gpuTimestampRef, ptimer);
                    return true;
                }
                else
                {
                    recordForwarded = static_cast<bool>(fnOnTimestampEvent(gpuTimestampRef, ptimer));
                    return recordForwarded;
                }
            };

            if (m_v1Observer.IsInitialized())
            {
                m_v1Observer.ConsumeGpuTimestampEvents([&](uint64_t gpuTimestampRef, uint64_t ptimer) {
                    const bool recordAccepted = fnForwardRecord(gpuTimestampRef, ptimer);
                    fnCommitForwardedRecord(true);
                    return recordAccepted;
                });
                return;
            }

            for (GpuReportBufferReader* const pReader : m_gpuReportBuffers)
            {
                pReader->ConsumeRecords(
                    [&](uint32_t /* recordType */, const uint8_t* pPayload, size_t payloadSize) -> bool {
                        if (payloadSize < sizeof(NvTrace_CUDA_ApiTrace_GpuTimestampReport))
                        {
                            return true;
                        }

                        NvTrace_CUDA_ApiTrace_GpuTimestampReport report;
                        Memcpy64(&report, pPayload, sizeof(report));
                        return fnForwardRecord(report.reportRef, report.ptimer);
                    },
                    fnCommitForwardedRecord);
            }
        }

        uint64_t EncodeGraphNodeCorrelationId(const uint64_t revision, const uint64_t graphNodeTableIndex) const
        {
            if (m_v1Observer.IsInitialized())
            {
                return m_v1Observer.EncodeGraphNodeCorrelationId(revision, graphNodeTableIndex);
            }
            return m_pGraphNodes->EncodeEntryRef(graphNodeTableIndex, revision);
        }

        uint64_t EncodeGraphExecCorrelationId(const uint64_t revision, const uint64_t graphExecTableIndex) const
        {
            if (m_v1Observer.IsInitialized())
            {
                return m_v1Observer.EncodeGraphExecCorrelationId(revision, graphExecTableIndex);
            }
            return m_pGraphExecs->EncodeEntryRef(graphExecTableIndex, revision);
        }

        // The graph-exec announce carries only the reference; decode it back to the (revision,
        // index) pair the correlation encoding is defined over.
        uint64_t EncodeGraphExecCorrelationId(const NvTrace_RecordReference& graphExecReference) const
        {
            if (m_v1Observer.IsInitialized())
            {
                return m_v1Observer.EncodeGraphExecCorrelationId(graphExecReference);
            }
            const auto [entryIndex, entryRevision] = m_pGraphExecs->DecodeEntryRef(graphExecReference.recordRef);
            return EncodeGraphExecCorrelationId(entryRevision, entryIndex);
        }

        // A V1 buffer announces no graph nodes, so no shim arm.
        uint64_t EncodeGraphNodeCorrelationId(const NvTrace_RecordReference& graphNodeReference) const
        {
            const auto [entryIndex, entryRevision] = m_pGraphNodes->DecodeEntryRef(graphNodeReference.recordRef);
            return EncodeGraphNodeCorrelationId(entryRevision, entryIndex);
        }

        // Decodes a GraphNodeCreated record's bytes, from a drain or a resolved announce, into the
        // common segment plus the params arm its graph node type selects.
        void DecodeGraphNodeRecord(const uint8_t* pPayload, size_t payloadSize, GraphNodeRecord& record) const
        {
            // m_recordVersionInfo is unset for a V1 buffer; the shim serves its own.
            const NvTrace_CUDA_ApiTrace_RecordVersionInfo& recordVersionInfo = GetRecordVersionInfo();
            const size_t paramsOffsetInPayload = recordVersionInfo.graphNodeCreatedSize;

            // graphNodeType arrives with the common segment; a node type this build does not know
            // still delivers the common fields, its params staying zero with paramsMinorVersion 0.
            Memcpy64(&record.common, pPayload, std::min({sizeof(record.common), paramsOffsetInPayload, AlignDnP2(payloadSize, sizeof(uint64_t))}));

            // The reported minor version is what the payload actually spans, so a consumer trusting it
            // never reads absent or zeroed params bytes as data.
            const uint32_t publishedParamsMinorVersion = CudaApiTraceGraphNodeParamsMinorVersion(recordVersionInfo, record.common.graphNodeType);
            const size_t paramsAvailableSize = paramsOffsetInPayload < payloadSize ? std::min(sizeof(record.params), AlignDnP2(payloadSize - paramsOffsetInPayload, sizeof(uint64_t))) : 0;
            const size_t paramsCopySize = std::min(AlignUpP2(static_cast<size_t>(publishedParamsMinorVersion), sizeof(uint64_t)), paramsAvailableSize);
            const uint32_t paramsMinorVersion = static_cast<uint32_t>(std::min(static_cast<size_t>(publishedParamsMinorVersion), paramsCopySize));
            if (paramsCopySize != 0)
            {
                Memcpy64(&record.params, pPayload + paramsOffsetInPayload, paramsCopySize);
            }

            record.paramsMinorVersion = paramsMinorVersion;
        }

    private:
        DataBlobReader* FindDataBlobReader(uint32_t sectionTag)
        {
            GenericObserver::Section* const pSection = m_genericObserver.FindSection(sectionTag);
            return pSection ? pSection->AsDataBlob() : nullptr;
        }

        const StateTableReader* FindStateTableReader(uint32_t sectionTag)
        {
            GenericObserver::Section* const pSection = m_genericObserver.FindSection(sectionTag);
            NVTRACE_LOG_ERR_RETURN_IF(!pSection, nullptr, 20, "CUDA API Trace state table section not found (sectionTag=%u).", sectionTag);

            const StateTableReader* const pStateTableReader = pSection->AsStateTable();
            NVTRACE_LOG_ERR_RETURN_IF(!pStateTableReader, nullptr, 20, "CUDA API Trace state table section is reader-less (sectionTag=%u).", sectionTag);

            return pStateTableReader;
        }

        // A reference names its table by section index.
        const StateTableReader* FindStateTableReaderBySectionIndex(uint32_t stateTableSectionIndex) const
        {
            if (m_genericObserver.SectionCount() <= stateTableSectionIndex)
            {
                return nullptr;
            }
            return m_genericObserver.GetSection(stateTableSectionIndex).AsStateTable();
        }

        std::vector<DataBlobReader*> CollectDeviceBlobs()
        {
            std::vector<DataBlobReader*> deviceBlobs;
            for (GenericObserver::Section* const pSection : m_genericObserver.RecordSections())
            {
                if (pSection->SectionTag() != NvTrace_CUDA_ApiTrace_SectionTag_Device)
                {
                    continue;
                }

                if (DataBlobReader* const pReader = pSection->AsDataBlob())
                {
                    deviceBlobs.push_back(pReader);
                }
            }
            return deviceBlobs;
        }

        std::vector<GpuReportBufferReader*> CollectGpuReportBuffers()
        {
            std::vector<GpuReportBufferReader*> gpuReportBuffers;
            for (GenericObserver::Section* const pSection : m_genericObserver.RecordSections())
            {
                if (pSection->SectionTag() != NvTrace_CUDA_ApiTrace_SectionTag_GpuReportBuffer)
                {
                    continue;
                }

                if (GpuReportBufferReader* const pReader = pSection->AsGpuReportBuffer())
                {
                    gpuReportBuffers.push_back(pReader);
                }
            }
            return gpuReportBuffers;
        }

        // A typed drain delivers entry records only: chains are not followed, and an entry carrying
        // some other record type is passed over.
        template <typename TRecord, typename TFN_OnRecord, typename TFN_CommitRecord>
        void ConsumeAttachStateTableRecords(const StateTableReader& stateTableReader, uint32_t expectedRecordType, TFN_OnRecord&& fnOnRecord, TFN_CommitRecord&& fnCommitRecord)
        {
            static_assert((sizeof(TRecord) % sizeof(uint64_t)) == 0, "A state-table record must be a whole number of 64-bit words, since Memcpy64 copies whole words only");

            stateTableReader.ConsumeAttachRecords<false>(
                [&](uint64_t entryRef, uint32_t recordType, const uint8_t* pPayload, size_t payloadSize) {
                    if (recordType != expectedRecordType)
                    {
                        return;
                    }

                    // The shared prefix only. A producer built against an older record publishes a
                    // shorter one, and the fields past it stay zero; which of them a consumer may
                    // trust is what the record's minor version, gated at Initialize, says.
                    TRecord record = {};
                    Memcpy64(&record, pPayload, std::min(sizeof(record), AlignDnP2(payloadSize, sizeof(uint64_t))));

                    const auto [entryIndex, entryRevision] = stateTableReader.DecodeEntryRef(entryRef);
                    fnOnRecord(entryIndex, entryRevision, record);
                },
                fnCommitRecord);
        }

    private:
        // What the table pointers name until Initialize succeeds, and again after Reset: an
        // uninitialized reader reports no entries, so every accessor answers as for an empty table.
        inline static const StateTableReader s_emptyStateTable;

        GenericObserver m_genericObserver;
        NvTrace_CUDA_ApiTrace_RecordVersionInfo m_recordVersionInfo = {};
        FbConfigurationView m_configurationView = {};
        const NvTrace_CUDA_ApiTrace_ProviderStatistics* m_pProviderStatistics = nullptr;

        const StateTableReader* m_pContexts = &s_emptyStateTable;
        const StateTableReader* m_pStreams = &s_emptyStateTable;
        const StateTableReader* m_pModules = &s_emptyStateTable;
        const StateTableReader* m_pGraphExecs = &s_emptyStateTable;
        const StateTableReader* m_pGraphNodes = &s_emptyStateTable;
        const StateTableReader* m_pChannels = &s_emptyStateTable;

        // Non-owning: the GenericObserver's reader vector owns them, and its heap storage keeps them
        // valid across a move of this observer.
        std::vector<DataBlobReader*> m_deviceBlobs;
        std::vector<GpuReportBufferReader*> m_gpuReportBuffers;

        std::vector<NvTrace_CUDA_ApiTrace_GpuTimestampReport> m_stagedGpuTimestampReports;

        uint32_t m_graphNodeParamsOffsetInPayload = 0;

        // The old-driver shim; at most one of it and m_genericObserver is initialized.
        CudaApiTraceObserverV1 m_v1Observer;
    };

} // namespace nv::trace
