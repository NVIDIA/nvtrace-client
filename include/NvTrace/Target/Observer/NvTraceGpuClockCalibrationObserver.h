/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/DataFormats/nvtrace_gpu_clock_calibration.h>
#include <NvTrace/Target/NvTraceClientLogger.h>
#include <NvTrace/Target/NvTraceGpuCalibrationPointTypes.h>
#include <NvTrace/Target/NvTraceGpuClockCalibrationProviderConfiguration_generated.h>
#include <NvTrace/Target/NvTraceScopeExitGuard.h>
#include <NvTrace/Target/NvTraceUuidUtilities.h>
#include <NvTrace/Target/Observer/NvTraceGenericObserver.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

namespace nv::trace {

    static_assert(sizeof(TscGpuCalibrationPoint) == sizeof(NvTrace_GPU_ClockCalibration_TscGpu_CalibrationPoint), "TscGpuCalibrationPoint must not add storage over the C ABI calibration point");

    /// \brief TSC/PTIMER Clock Calibration Observer
    ///
    /// Thin typed wrapper over GenericObserver. It retains only the provider-specific state needed by
    /// GetUpdateIntervalNsec() and GetLatestCalibrationPoint(); records are otherwise forwarded through
    /// the generic APIs.
    class GpuClockCalibrationObserver
    {
    public:
        ~GpuClockCalibrationObserver() = default;
        GpuClockCalibrationObserver() noexcept = default;

        GpuClockCalibrationObserver(const GpuClockCalibrationObserver&) = delete;
        GpuClockCalibrationObserver(GpuClockCalibrationObserver&& rhs) noexcept
        {
            *this = std::move(rhs);
        }

        GpuClockCalibrationObserver& operator=(const GpuClockCalibrationObserver&) noexcept = delete;
        GpuClockCalibrationObserver& operator=(GpuClockCalibrationObserver&& rhs) noexcept
        {
            if (this != &rhs)
            {
                // Reader element addresses remain stable when GenericObserver's reserved vectors are moved.
                m_genericObserver = std::move(rhs.m_genericObserver);
                m_configurationStorage = std::move(rhs.m_configurationStorage);
                m_pFbProviderConfiguration = m_configurationStorage.empty()
                                               ? nullptr
                                               : flatbuffers::GetRoot<FB::Provider::GpuClockCalibration::ProviderConfiguration>(m_configurationStorage.data());
                rhs.m_pFbProviderConfiguration = nullptr;
                m_pCalibrationPointsReader = std::exchange(rhs.m_pCalibrationPointsReader, nullptr);
            }
            return *this;
        }

        void Reset() noexcept
        {
            NVTRACE_LOG_INF(90, "Resetting ...");
            m_pCalibrationPointsReader = nullptr;
            m_pFbProviderConfiguration = nullptr;
            m_configurationStorage.clear();
            m_genericObserver.Reset();
        }

        // Size-less entry point: SIZE_MAX degrades the validator's containment check to overflow-only; prefer the size-aware overload.
        bool Initialize(const uint8_t* pMappedProviderBufferMemory)
        {
            return Initialize(pMappedProviderBufferMemory, std::numeric_limits<size_t>::max());
        }

        bool Initialize(const uint8_t* pMappedProviderBufferMemory, size_t providerBufferSize)
        {
            auto errorGuard = ScopeExitGuard([&]() {
                NVTRACE_LOG_ERR(20, "GpuClockCalibrationObserver::Initialize failed (provider buffer=%p providerBufferSize=%zu).", static_cast<const void*>(pMappedProviderBufferMemory), providerBufferSize);
                Reset();
            });

            if (!m_genericObserver.Initialize(pMappedProviderBufferMemory, providerBufferSize))
            {
                return false;
            }

            // UUID
            const NVTRACE_UUID providerTypeUuid = m_genericObserver.ProviderTypeUuid();
            NVTRACE_LOG_ERR_RETURN_IF(providerTypeUuid != NvTrace_GPU_ClockCalibration_TscGpu_ProviderTypeUuid, false, 20, "Invalid provider type UUID for GPU Clock Calibration observer (got %s, expected %s).", ToString(providerTypeUuid).c_str(), ToString(NvTrace_GPU_ClockCalibration_TscGpu_ProviderTypeUuid).c_str());

            // Record version info
            DataBlobReader* const pRecordVersionInfoReader = FindDataBlobReader(NvTrace_GPU_ClockCalibration_TscGpu_SectionTag_RecordVersionInfo);
            NVTRACE_LOG_ERR_RETURN_IF(!pRecordVersionInfoReader, false, 20, "GPU Clock Calibration record version info section not found or invalid.");

            // Provider configuration
            DataBlobReader* const pConfigurationReader = FindDataBlobReader(NvTrace_GPU_ClockCalibration_TscGpu_SectionTag_ProviderConfiguration);
            NVTRACE_LOG_ERR_RETURN_IF(!pConfigurationReader, false, 20, "GPU Clock Calibration configuration section not found or invalid.");
            // Copy out of the concurrently-writable mapped buffer before verify/parse, to avoid a verify-then-use TOCTOU.
            pConfigurationReader->ReadRecord([&](uint32_t /* recordType */, const uint8_t* pPayload, size_t payloadSize) {
                m_configurationStorage.assign(pPayload, pPayload + payloadSize);
            });
            if (!flatbuffers::Verifier(m_configurationStorage.data(), m_configurationStorage.size()).VerifyBuffer<FB::Provider::GpuClockCalibration::ProviderConfiguration>())
            {
                NVTRACE_LOG_ERR(20, "Provider configuration FlatBuffer verification failed for GPU Clock Calibration (configurationSize=%zu flatbuffer=%p).", m_configurationStorage.size(), static_cast<const void*>(m_configurationStorage.data()));
                return false;
            }
            m_pFbProviderConfiguration = flatbuffers::GetRoot<FB::Provider::GpuClockCalibration::ProviderConfiguration>(m_configurationStorage.data());

            // Calibration point circular buffer
            m_pCalibrationPointsReader = FindCircularBufferReader(NvTrace_GPU_ClockCalibration_TscGpu_SectionTag_CalibrationPointCircularBuffer);
            NVTRACE_LOG_ERR_RETURN_IF(!m_pCalibrationPointsReader, false, 20, "GPU Clock Calibration circular buffer section not found or invalid.");

            errorGuard.Dismiss();
            NVTRACE_LOG_INF(90, "Initialized successfully.");
            return true;
        }

        bool IsInitialized() const
        {
            return m_genericObserver.IsInitialized() && m_pFbProviderConfiguration;
        }

        bool IsProviderMeasuring() const
        {
            return IsInitialized() && m_genericObserver.ProviderStatus() == NvTrace_ProviderStatus_measuring;
        }

        uint64_t GetUpdateIntervalNsec() const
        {
            if (!IsInitialized())
            {
                return 0;
            }

            return m_pFbProviderConfiguration->updateIntervalNsec();
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
            m_genericObserver.SetObservationStart(observationStart, true);
        }

        // Snapshots the most recent calibration point without disturbing the recurring-record read cursor. Returns
        // false if the buffer is empty, every attempt is lapped by the writer, or the latest record is not a
        // calibration point. retryCount is the number of retries after the initial attempt and is clamped to 64.
        bool GetLatestCalibrationPoint(TscGpuCalibrationPoint& calibrationPointOut, uint32_t retryCount = 3) const
        {
            if (!IsInitialized())
            {
                return false;
            }

            const uint32_t clampedRetryCount = std::min(retryCount, 64u);
            for (uint32_t attempt = 0; attempt <= clampedRetryCount; ++attempt)
            {
                TscGpuCalibrationPoint calibrationPoint = {};
                bool isCalibrationPoint = false;
                const bool isConsistent = m_pCalibrationPointsReader->PeekLatestRecord(
                    [&](uint32_t recordType, const uint8_t* pPayload, size_t payloadSize) {
                        if (IsCalibrationPointRecord(recordType, payloadSize))
                        {
                            memcpy(&calibrationPoint, pPayload, sizeof(calibrationPoint));
                            isCalibrationPoint = true;
                        }
                    });
                if (isConsistent && isCalibrationPoint)
                {
                    calibrationPointOut = calibrationPoint;
                    return true;
                }
            }
            return false;
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
        bool IsCalibrationPointRecord(uint32_t recordType, size_t payloadSize) const
        {
            return NvTrace_GPU_ClockCalibration_TscGpu_RecordType_CalibrationPoint == recordType && sizeof(TscGpuCalibrationPoint) <= payloadSize;
        }

        GenericObserver m_genericObserver;
        std::vector<uint8_t> m_configurationStorage;
        const FB::Provider::GpuClockCalibration::ProviderConfiguration* m_pFbProviderConfiguration = nullptr;
        CircularBufferReader* m_pCalibrationPointsReader = nullptr;
    };

} // namespace nv::trace
