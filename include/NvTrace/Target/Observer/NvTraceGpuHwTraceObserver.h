/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/DataFormats/nvtrace_gpu_hw_trace.h>
#include <NvTrace/Target/NvTraceClientLogger.h>
#include <NvTrace/Target/NvTraceScopeExitGuard.h>
#include <NvTrace/Target/NvTraceUuidUtilities.h>
#include <NvTrace/Target/Observer/NvTraceGenericObserver.h>

#include <limits>
#include <utility>

namespace nv::trace {

    /// \brief HW Trace Observer
    ///
    /// Thin typed wrapper over GenericObserver. GenericObserver owns provider-buffer layout validation and
    /// record traversal; this wrapper only validates the provider identity and exposes GPU HW Trace-specific
    /// section accessors.
    class GpuHwTraceObserver
    {
    public:
        ~GpuHwTraceObserver() = default;
        GpuHwTraceObserver() noexcept = default;

        GpuHwTraceObserver(const GpuHwTraceObserver&) = delete;
        GpuHwTraceObserver(GpuHwTraceObserver&& rhs) noexcept
        {
            *this = std::move(rhs);
        }

        GpuHwTraceObserver& operator=(const GpuHwTraceObserver&) = delete;
        GpuHwTraceObserver& operator=(GpuHwTraceObserver&& rhs) noexcept
        {
            if (this != &rhs)
            {
                m_genericObserver = std::move(rhs.m_genericObserver);
            }
            return *this;
        }

        void Reset() noexcept
        {
            m_genericObserver.Reset();
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
                NVTRACE_LOG_ERR(20, "GpuHwTraceObserver::Initialize failed (provider buffer=%p providerBufferSize=%zu).", static_cast<const void*>(pMappedProviderBufferMemory), providerBufferSize);
                Reset();
            });

            if (!m_genericObserver.Initialize(pMappedProviderBufferMemory, providerBufferSize))
            {
                NVTRACE_LOG_ERR(20, "GenericObserver initialization failed.");
                return false;
            }

            // UUID
            const NVTRACE_UUID providerTypeUuid = m_genericObserver.ProviderTypeUuid();
            NVTRACE_LOG_ERR_RETURN_IF(providerTypeUuid != NvTrace_GPU_HwTrace_ProviderTypeUuid, false, 20, "Invalid provider type UUID for GPU HW Trace observer (got %s, expected %s).", ToString(providerTypeUuid).c_str(), ToString(NvTrace_GPU_HwTrace_ProviderTypeUuid).c_str());

            // Record version info
            DataBlobReader* const pRecordVersionInfoReader = FindDataBlobReader(NvTrace_GPU_HwTrace_SectionTag_RecordVersionInfo);
            NVTRACE_LOG_ERR_RETURN_IF(!pRecordVersionInfoReader, false, 20, "GPU HW Trace record version info section not found or invalid.");

            // Provider configuration
            DataBlobReader* const pConfigurationReader = FindDataBlobReader(NvTrace_GPU_HwTrace_SectionTag_ProviderConfiguration);
            NVTRACE_LOG_ERR_RETURN_IF(!pConfigurationReader, false, 20, "GPU HW Trace provider configuration section not found or invalid.");

            // Provider diagnostic event circular buffer
            CircularBufferReader* const pProviderDiagnosticEventCbReader = FindCircularBufferReader(NvTrace_GPU_HwTrace_SectionTag_ProviderDiagnosticEventCircularBuffer);
            NVTRACE_LOG_ERR_RETURN_IF(!pProviderDiagnosticEventCbReader, false, 20, "GPU HW Trace provider diagnostic circular buffer section not found or invalid.");

            // A diagnostic-only faulted provider may omit the following circular buffers; log the error and continue.
            CircularBufferReader* const pHwTraceEventCbReader = FindCircularBufferReader(NvTrace_GPU_HwTrace_SectionTag_HwTraceEventCircularBuffer);
            NVTRACE_LOG_WRN_IF(!pHwTraceEventCbReader, 20, "GPU HW Trace event circular buffer section not found or invalid.");

            CircularBufferReader* const pLifecycleEventCbReader = FindCircularBufferReader(NvTrace_GPU_HwTrace_SectionTag_LifecycleEventCircularBuffer);
            NVTRACE_LOG_WRN_IF(!pLifecycleEventCbReader, 20, "GPU HW Trace lifecycle event circular buffer section not found or invalid.");

            errorGuard.Dismiss();
            NVTRACE_LOG_INF(90, "GpuHwTraceObserver initialized successfully.");
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

        uint32_t GetOnAttachRecordsTotalCount() const
        {
            return m_genericObserver.GetOnAttachRecordsTotalCount();
        }

        template <typename TFN_ReadRecord>
        void ConsumeOnAttachDataBlobRecords(TFN_ReadRecord&& fnReadRecord)
        {
            if (IsInitialized())
            {
                m_genericObserver.ConsumeOnAttachDataBlobRecords(fnReadRecord);
            }
        }

        template <typename TFN_ReadRecord>
        void ConsumeOnAttachDataBlobRecords_IgnoringConsumedState(TFN_ReadRecord&& fnReadRecord)
        {
            if (IsInitialized())
            {
                m_genericObserver.ConsumeOnAttachDataBlobRecords_IgnoringConsumedState(fnReadRecord);
            }
        }

        template <typename TFN_ReadRecord, typename TFN_CommitRecord>
        void ConsumeCircularBufferRecords(const ReadEndpoints& readEndpoints, TFN_ReadRecord&& fnReadRecord, TFN_CommitRecord&& fnCommitRecord)
        {
            if (IsInitialized())
            {
                m_genericObserver.ConsumeCircularBufferRecords(readEndpoints, fnReadRecord, fnCommitRecord);
            }
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
    };

} // namespace nv::trace
