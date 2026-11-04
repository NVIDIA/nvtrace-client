/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/DataFormats/nvtrace_gpu_pm_counters.h>
#include <NvTrace/Target/NvTraceSharedMemory.h>
#include <NvTrace/Target/NvTraceUuidUtilities.h>
#include <NvTrace/Target/Observer/NvTraceCircularBufferReader.h>
#include <NvTrace/Target/Observer/NvTraceObserverUtilities.h>

#include <NvTrace/Target/NvTraceFlatBufferUtilities.h>
#include <NvTrace/Target/NvTraceGpuPmCounterProviderConfiguration_generated.h>

#include <NvTrace/Target/Observer/NvTraceGenericObserver.h>

namespace nv::trace {

    /// \brief GPU PM Counter Observer
    ///
    /// Thin typed wrapper over GenericObserver: TIDI
    class GpuPmCounterObserver
    {
    public:
        using FbConfigurationView = FlatBufferView<FB::Provider::GpuPmCounters::ProviderConfiguration>;

        ~GpuPmCounterObserver() = default;
        GpuPmCounterObserver() noexcept = default;

        GpuPmCounterObserver(const GpuPmCounterObserver&) = delete;
        GpuPmCounterObserver(GpuPmCounterObserver&& rhs) noexcept
        {
            *this = std::move(rhs);
        }

        GpuPmCounterObserver& operator=(const GpuPmCounterObserver&) = delete;
        GpuPmCounterObserver& operator=(GpuPmCounterObserver&& rhs) noexcept
        {
            if (this != &rhs)
            {
                m_genericObserver = std::move(rhs.m_genericObserver);
                m_providerConfigurationStorage = std::move(rhs.m_providerConfigurationStorage);
                m_pFbProviderConfiguration = std::exchange(rhs.m_pFbProviderConfiguration, nullptr);
            }
            return *this;
        }

        void Reset() noexcept
        {
            m_genericObserver.Reset();
            m_providerConfigurationStorage.clear();
            m_pFbProviderConfiguration = nullptr;
        }

        // Migration bridge for TypedProviderState. SIZE_MAX degrades the validator's containment check
        // to overflow-only; prefer the size-aware overload.
        bool Initialize(const uint8_t* pMappedProviderBufferMemory)
        {
            return Initialize(pMappedProviderBufferMemory, std::numeric_limits<size_t>::max());
        }

        bool Initialize(const uint8_t* pMappedProviderBufferMemory, size_t providerBufferSize)
        {
            auto errorGuard = ScopeExitGuard([&]() {
                NVTRACE_LOG_ERR(20, "GpuPmCounterObserver::Initialize failed (provider buffer=%p providerBufferSize=%zu).", static_cast<const void*>(pMappedProviderBufferMemory), providerBufferSize);
                Reset();
            });

            const bool genericObserverInitialized = m_genericObserver.Initialize(pMappedProviderBufferMemory, providerBufferSize);
            if (!genericObserverInitialized)
            {
                NVTRACE_LOG_ERR(20, "GenericObserver initialization failed.");
                return false;
            }

            // UUID
            const NVTRACE_UUID providerTypeUuid = m_genericObserver.ProviderTypeUuid();
            NVTRACE_LOG_ERR_RETURN_IF(providerTypeUuid != NvTrace_GPU_PmCounters_ProviderTypeUuid, false, 20, "Invalid provider type UUID for GPU PM Counter observer (got %s, expected %s).", ToString(providerTypeUuid).c_str(), ToString(NvTrace_GPU_PmCounters_ProviderTypeUuid).c_str());

            // Record version info
            DataBlobReader* pRecordVersionInfoReader = FindDataBlobReader(NvTrace_GPU_PmCounters_SectionTag_RecordVersionInfo);
            NVTRACE_LOG_ERR_RETURN_IF(!pRecordVersionInfoReader, false, 20, "GPU PM Counter record version info section not found or invalid.");

            // Provider configuration
            DataBlobReader* pConfigurationReader = FindDataBlobReader(NvTrace_GPU_PmCounters_SectionTag_ProviderConfiguration);
            NVTRACE_LOG_ERR_RETURN_IF(!pConfigurationReader, false, 20, "GPU PM Provider configuration section not found or invalid.");

            // Provider diagnostic event circular buffer
            CircularBufferReader* const pProviderDiagnosticEventCbReader = FindCircularBufferReader(NvTrace_GPU_PmCounters_SectionTag_ProviderDiagnosticEventCircularBuffer);
            NVTRACE_LOG_ERR_RETURN_IF(!pProviderDiagnosticEventCbReader, false, 20, "GPU PM Provider diagnostic event circular buffer section not found or invalid.");

            // A diagnostic-only faulted provider may omit the following circular buffers; log the error and continue.
            CircularBufferReader* const pCounterSampleCbReader = FindCircularBufferReader(NvTrace_GPU_PmCounters_SectionTag_CounterSampleCircularBuffer);
            NVTRACE_LOG_WRN_IF(!pCounterSampleCbReader, 20, "GPU PM Counter sample circular buffer section not found or invalid.");

            CircularBufferReader* const pLifeCycleEventCbReader = FindCircularBufferReader(NvTrace_GPU_PmCounters_SectionTag_LifecycleEventCircularBuffer);
            NVTRACE_LOG_WRN_IF(!pLifeCycleEventCbReader, 20, "GPU PM Life cycle event circular buffer section not found or invalid.");

            pConfigurationReader->ReadRecord([&](uint32_t /* recordType */, const uint8_t* pPayload, size_t payloadSize) {
                m_providerConfigurationStorage.assign(pPayload, pPayload + payloadSize);
            });
            FbConfigurationView configurationView = {m_providerConfigurationStorage.data(), m_providerConfigurationStorage.size()};

            NVTRACE_LOG_ERR_RETURN_IF(
                !configurationView.Verify(),
                false,
                20,
                "GPU PM Provider configuration verification failed.");
            m_pFbProviderConfiguration = configurationView.GetRoot();

            errorGuard.Dismiss();
            NVTRACE_LOG_INF(90, "Initialized successfully.");
            return true;
        }

        bool IsInitialized() const
        {
            return m_genericObserver.IsInitialized();
        }

        bool IsProviderMeasuring() const
        {
            return IsInitialized() && m_genericObserver.ProviderStatus() == NvTrace_ProviderStatus_measuring;
        }

        const FB::Provider::GpuPmCounters::ProviderConfiguration* GetProviderConfiguration() const
        {
            return m_pFbProviderConfiguration;
        }

        FbConfigurationView GetProviderConfigurationFlatBufferView() const
        {
            return {m_providerConfigurationStorage.data(), (size_t)m_providerConfigurationStorage.size()};
        }

        uint32_t GetPmSampleRateHz() const
        {
            return m_pFbProviderConfiguration ? m_pFbProviderConfiguration->pmSampleRateHz() : 0;
        }

        uint64_t GetUpdateIntervalNsec() const
        {
            return m_pFbProviderConfiguration ? m_pFbProviderConfiguration->updateIntervalNsec() : 0;
        }

        uint32_t GetOnAttachRecordsTotalCount() const
        {
            return m_genericObserver.GetOnAttachRecordsTotalCount();
        }

        template <typename TFN_ReadRecord>
        void ConsumeOnAttachDataBlobRecords(TFN_ReadRecord&& fnReadRecord)
        {
            if (!IsInitialized())
            {
                return;
            }
            m_genericObserver.ConsumeOnAttachDataBlobRecords(fnReadRecord);
        }

        template <typename TFN_ReadRecord>
        void ConsumeOnAttachDataBlobRecords_IgnoringConsumedState(TFN_ReadRecord&& fnReadRecord)
        {
            if (!IsInitialized())
            {
                return;
            }
            m_genericObserver.ConsumeOnAttachDataBlobRecords_IgnoringConsumedState(fnReadRecord);
        }

        template <typename TFN_ReadRecord, typename TFN_CommitRecord>
        void ConsumeCircularBufferRecords(const ReadEndpoints& readEndpoints, TFN_ReadRecord&& fnReadRecord, TFN_CommitRecord&& fnCommitRecord)
        {
            if (!IsInitialized())
            {
                return;
            }
            m_genericObserver.ConsumeCircularBufferRecords(readEndpoints, fnReadRecord, fnCommitRecord);
        }

        void SetObservationStart(const ReadEndpoints& observationStart)
        {
            m_genericObserver.SetObservationStart(observationStart);
        }

    protected:
        DataBlobReader* FindDataBlobReader(uint32_t sectionTag)
        {
            GenericObserver::Section* const pSection = m_genericObserver.FindSection(sectionTag);
            return pSection ? pSection->AsDataBlob() : nullptr;
        }

        CircularBufferReader* FindCircularBufferReader(uint32_t sectionTag)
        {
            GenericObserver::Section* const pSection = m_genericObserver.FindSection(sectionTag);
            return pSection ? pSection->AsCircularBuffer() : nullptr;
        }

    private:
        GenericObserver m_genericObserver;
        // Non-owning cache for fast API reads
        std::vector<uint8_t> m_providerConfigurationStorage;
        const FB::Provider::GpuPmCounters::ProviderConfiguration* m_pFbProviderConfiguration = nullptr;
    };

}; // namespace nv::trace
