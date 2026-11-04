/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/DataFormats/nvtrace_provider_buffer.h>
#include <NvTrace/Target/NvTraceClientLogger.h>
#include <NvTrace/Target/NvTraceScopeExitGuard.h>
#include <NvTrace/Target/NvTraceUuidUtilities.h>
#include <NvTrace/Target/Observer/NvTraceAdaptiveBufferReader.h>
#include <NvTrace/Target/Observer/NvTraceCircularBufferReader.h>
#include <NvTrace/Target/Observer/NvTraceDataBlobReader.h>
#include <NvTrace/Target/Observer/NvTraceGpuReportBufferReader.h>
#include <NvTrace/Target/Observer/NvTraceSectionLayoutValidator.h>
#include <NvTrace/Target/Observer/NvTraceStateArenaReader.h>
#include <NvTrace/Target/Observer/NvTraceStateTableReader.h>

#include <cinttypes>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

namespace nv::trace {

    struct ReadEndpoints
    {
        uint64_t tsc;
        uint64_t gpuNsec;
    };

    // Generic infrastructure for reading a mapped provider buffer: section lookup, attach-data blobs,
    // and circular-buffer / adaptive-buffer record consumption. Typed observers (e.g. CpuClockCalibrationObserver)
    // sit on top.
    class GenericObserver
    {
    public:
        static constexpr uint32_t INVALID_SECTION_INDEX = UINT32_MAX;

        ~GenericObserver() = default;
        GenericObserver() = default;

        GenericObserver(const GenericObserver&) = delete;
        GenericObserver(GenericObserver&& rhs) noexcept
            : m_pProviderBufferHeader(std::exchange(rhs.m_pProviderBufferHeader, nullptr))
            , m_dataBlobReaders(std::move(rhs.m_dataBlobReaders))
            , m_onAttachRecordsTotalCount(std::exchange(rhs.m_onAttachRecordsTotalCount, 0))
            , m_circularBufferReaders(std::move(rhs.m_circularBufferReaders))
            , m_adaptiveBufferReaders(std::move(rhs.m_adaptiveBufferReaders))
            , m_stateTableReaders(std::move(rhs.m_stateTableReaders))
            , m_stateArenaReaders(std::move(rhs.m_stateArenaReaders))
            , m_gpuReportBufferReaders(std::move(rhs.m_gpuReportBufferReaders))
            , m_sections(std::move(rhs.m_sections))
        {
        }

        GenericObserver& operator=(const GenericObserver&) = delete;
        GenericObserver& operator=(GenericObserver&& rhs) noexcept
        {
            if (this != &rhs)
            {
                m_pProviderBufferHeader = std::exchange(rhs.m_pProviderBufferHeader, nullptr);
                m_dataBlobReaders = std::move(rhs.m_dataBlobReaders);
                m_onAttachRecordsTotalCount = std::exchange(rhs.m_onAttachRecordsTotalCount, 0);
                m_circularBufferReaders = std::move(rhs.m_circularBufferReaders);
                m_adaptiveBufferReaders = std::move(rhs.m_adaptiveBufferReaders);
                m_stateTableReaders = std::move(rhs.m_stateTableReaders);
                m_stateArenaReaders = std::move(rhs.m_stateArenaReaders);
                m_gpuReportBufferReaders = std::move(rhs.m_gpuReportBufferReaders);
                m_sections = std::move(rhs.m_sections);
            }
            return *this;
        }

        void Reset() noexcept
        {
            m_sections.clear();
            m_gpuReportBufferReaders.clear();
            m_stateArenaReaders.clear();
            m_stateTableReaders.clear();
            m_adaptiveBufferReaders.clear();
            m_circularBufferReaders.clear();
            m_dataBlobReaders.clear();
            m_onAttachRecordsTotalCount = 0;
            m_pProviderBufferHeader = nullptr;
        }

        bool IsInitialized() const
        {
            return m_pProviderBufferHeader != nullptr;
        }

        NVTRACE_UUID ProviderTypeUuid() const
        {
            return m_pProviderBufferHeader ? m_pProviderBufferHeader->providerTypeUuid : NVTRACE_UUID{};
        }

        const char* GetProviderName() const
        {
            return m_pProviderBufferHeader ? m_pProviderBufferHeader->providerName : "";
        }

        NvTrace_ProviderStatus ProviderStatus() const
        {
            return m_pProviderBufferHeader ? m_pProviderBufferHeader->providerStatus : NvTrace_ProviderStatus_uninitialized;
        }

        uint32_t ThreadEventSourceProviderIndex() const
        {
            return m_pProviderBufferHeader ? m_pProviderBufferHeader->threadEventSourceProviderIndex : 0;
        }

        bool Initialize(const uint8_t* pProviderBuffer, size_t providerBufferSize)
        {
            NVTRACE_LOG_ERR_RETURN_IF(providerBufferSize < sizeof(NvTrace_ProviderBufferVersion), false, 20, "Provider buffer is too small to contain the header version information (size=%zu).", providerBufferSize);

            // Header/version/descriptors are read in place; require the base to meet the strictest alignment.
            NVTRACE_LOG_ERR_RETURN_IF((reinterpret_cast<uintptr_t>(pProviderBuffer) % alignof(NvTrace_ProviderBufferHeader)) != 0, false, 20, "Provider buffer base must be %zu-byte aligned (pProviderBuffer=%p).", alignof(NvTrace_ProviderBufferHeader), static_cast<const void*>(pProviderBuffer));

            const NvTrace_ProviderBufferVersion* pProviderBufferVersion = reinterpret_cast<const NvTrace_ProviderBufferVersion*>(pProviderBuffer);
            NVTRACE_LOG_ERR_RETURN_IF(pProviderBufferVersion->majorVersion != NvTrace_ProviderBuffer_MAJOR_VERSION_2, false, 20, "Unsupported provider buffer major version %u.", pProviderBufferVersion->majorVersion);

            SectionLayoutValidator layoutValidator(providerBufferSize);

            bool succeeded = layoutValidator.AddAndValidateSection(0, pProviderBufferVersion->providerBufferHeaderSize, pProviderBufferVersion->providerBufferHeaderMinorVersion, NvTrace_ProviderBufferHeader_MINOR_VERSION_A);
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 20, "Provider buffer header layout validation failed (providerBufferHeaderSize=%u providerBufferHeaderMinorVersion=%u).", pProviderBufferVersion->providerBufferHeaderSize, pProviderBufferVersion->providerBufferHeaderMinorVersion);

            const NvTrace_ProviderBufferHeader* pProviderBufferHeader = reinterpret_cast<const NvTrace_ProviderBufferHeader*>(pProviderBuffer);

            NVTRACE_LOG_ERR_RETURN_IF(pProviderBufferHeader->recordReferenceMinorVersion < NvTrace_RecordReference_MINOR_VERSION_A, false, 20, "Record reference layout is incompatible (minorVersion=%u expected=%u).", pProviderBufferHeader->recordReferenceMinorVersion, NvTrace_RecordReference_MINOR_VERSION_A);

            succeeded = layoutValidator.AddAndValidateSection(pProviderBufferHeader->sectionDescsOffset, pProviderBufferHeader->sectionDescCount, pProviderBufferHeader->sectionDescSize, pProviderBufferHeader->sectionDescMinorVersion, NvTrace_ProviderBufferSectionDesc_MINOR_VERSION_A);
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 20, "Provider buffer section descriptor table layout validation failed (sectionDescsOffset=%" PRIu64 " sectionDescCount=%u sectionDescSize=%u).", pProviderBufferHeader->sectionDescsOffset, pProviderBufferHeader->sectionDescCount, pProviderBufferHeader->sectionDescSize);

            const uint8_t* pSectionDescs = pProviderBuffer + pProviderBufferHeader->sectionDescsOffset;

            // Descriptor-aligned base + descriptor-multiple stride keeps every in-place descriptor aligned.
            NVTRACE_LOG_ERR_RETURN_IF((reinterpret_cast<uintptr_t>(pSectionDescs) % alignof(NvTrace_ProviderBufferSectionDesc)) != 0 || (pProviderBufferHeader->sectionDescSize % alignof(NvTrace_ProviderBufferSectionDesc)) != 0, false, 20, "Provider buffer section descriptor table is misaligned (sectionDescsOffset=%" PRIu64 " sectionDescSize=%u alignment=%zu).", pProviderBufferHeader->sectionDescsOffset, pProviderBufferHeader->sectionDescSize, alignof(NvTrace_ProviderBufferSectionDesc));

            struct ResolvedSection
            {
                const uint8_t* pSection = nullptr;
                uint64_t sectionSize = 0;
                uint32_t sectionHeaderSize = 0;
                uint32_t sectionHeaderMinorVersion = 0;
            };
            // A dangling or wrong-typed cross-section reference is buffer corruption, not version skew,
            // so unlike any other reader-init failure it fails the whole attach.
            bool sectionResolutionFailed = false;
            auto fnResolveSection = [pProviderBuffer, pProviderBufferHeader, pSectionDescs, &sectionResolutionFailed](uint32_t sectionIndex, NvTrace_ProviderBufferSectionType expectedSectionType) -> ResolvedSection {
                const bool sectionIndexOutOfRange = pProviderBufferHeader->sectionDescCount <= sectionIndex;
                sectionResolutionFailed |= sectionIndexOutOfRange;
                NVTRACE_LOG_ERR_RETURN_IF(sectionIndexOutOfRange, {}, 20, "Section resolution failed (sectionIndex=%u sectionCount=%u).", sectionIndex, pProviderBufferHeader->sectionDescCount);

                const NvTrace_ProviderBufferSectionDesc* pSectionDesc = reinterpret_cast<const NvTrace_ProviderBufferSectionDesc*>(pSectionDescs + static_cast<uint64_t>(sectionIndex) * pProviderBufferHeader->sectionDescSize);
                const bool sectionTypeMismatched = pSectionDesc->sectionType != expectedSectionType;
                sectionResolutionFailed |= sectionTypeMismatched;
                NVTRACE_LOG_ERR_RETURN_IF(sectionTypeMismatched, {}, 20, "Section resolution type mismatch (sectionIndex=%u sectionType=%u expectedSectionType=%u).", sectionIndex, static_cast<uint32_t>(pSectionDesc->sectionType), static_cast<uint32_t>(expectedSectionType));

                return {
                    pProviderBuffer + pSectionDesc->sectionOffset,
                    pSectionDesc->sectionSize,
                    pSectionDesc->sectionHeaderSize,
                    pSectionDesc->sectionHeaderMinorVersion};
            };

            // Validate every section's placement before any reader is constructed, since a reader may
            // resolve a section appearing later in the array. Counting here also lets the reader vectors
            // be reserved up front: the Section handles point into them, so they must never reallocate.
            uint32_t dataBlobCount = 0;
            uint32_t circularBufferCount = 0;
            uint32_t adaptiveBufferCount = 0;
            uint32_t stateTableCount = 0;
            uint32_t stateArenaCount = 0;
            uint32_t gpuReportBufferCount = 0;
            for (uint32_t sectionDescIndex = 0; sectionDescIndex < pProviderBufferHeader->sectionDescCount; ++sectionDescIndex)
            {
                const NvTrace_ProviderBufferSectionDesc* pSectionDesc = reinterpret_cast<const NvTrace_ProviderBufferSectionDesc*>(pSectionDescs + static_cast<uint64_t>(sectionDescIndex) * pProviderBufferHeader->sectionDescSize);
                succeeded = layoutValidator.AddAndValidateSection(pSectionDesc->sectionOffset, pSectionDesc->sectionSize);
                NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 20, "Provider buffer section layout validation failed (sectionIndex=%u sectionOffset=%" PRIu64 " sectionSize=%" PRIu64 ").", sectionDescIndex, pSectionDesc->sectionOffset, pSectionDesc->sectionSize);

                switch (pSectionDesc->sectionType)
                {
                    case NvTrace_ProviderBufferSectionType_DataBlob: ++dataBlobCount; break;
                    case NvTrace_ProviderBufferSectionType_CircularBuffer: ++circularBufferCount; break;
                    case NvTrace_ProviderBufferSectionType_AdaptiveBuffer: ++adaptiveBufferCount; break;
                    case NvTrace_ProviderBufferSectionType_StateTable: ++stateTableCount; break;
                    case NvTrace_ProviderBufferSectionType_StateArena: ++stateArenaCount; break;
                    case NvTrace_ProviderBufferSectionType_GpuReportBuffer: ++gpuReportBufferCount; break;
                    default: break;
                }
            }

            std::vector<DataBlobReader> dataBlobReaders;
            std::vector<CircularBufferReader> circularBufferReaders;
            std::vector<AdaptiveBufferReader> adaptiveBufferReaders;
            std::vector<StateTableReader> stateTableReaders;
            std::vector<StateArenaReader> stateArenaReaders;
            std::vector<GpuReportBufferReader> gpuReportBufferReaders;
            dataBlobReaders.reserve(dataBlobCount);
            circularBufferReaders.reserve(circularBufferCount);
            adaptiveBufferReaders.reserve(adaptiveBufferCount);
            stateTableReaders.reserve(stateTableCount);
            stateArenaReaders.reserve(stateArenaCount);
            gpuReportBufferReaders.reserve(gpuReportBufferCount);

            std::vector<Section> sections;
            sections.reserve(pProviderBufferHeader->sectionDescCount);

            for (uint32_t sectionDescIndex = 0; sectionDescIndex < pProviderBufferHeader->sectionDescCount; ++sectionDescIndex)
            {
                const NvTrace_ProviderBufferSectionDesc* pSectionDesc = reinterpret_cast<const NvTrace_ProviderBufferSectionDesc*>(pSectionDescs + static_cast<uint64_t>(sectionDescIndex) * pProviderBufferHeader->sectionDescSize);

                void* pReader = nullptr;

                // Version skew in one section must not fail the whole attach: the reader is discarded
                // (pop_back keeps the reserved vectors dense) and the section exposed reader-less.
                sectionResolutionFailed = false;
                switch (pSectionDesc->sectionType)
                {
                    case NvTrace_ProviderBufferSectionType_DataBlob:
                    {
                        DataBlobReader& reader = dataBlobReaders.emplace_back();
                        succeeded = reader.Initialize(sectionDescIndex, fnResolveSection);
                        if (succeeded)
                        {
                            pReader = &reader;
                        }
                        else
                        {
                            dataBlobReaders.pop_back();
                            NVTRACE_LOG_WRN(20, "Data blob section reader initialization failed; section left reader-less (sectionIndex=%u sectionTag=%u).", sectionDescIndex, pSectionDesc->sectionTag);
                        }
                        break;
                    }
                    case NvTrace_ProviderBufferSectionType_CircularBuffer:
                    {
                        CircularBufferReader& reader = circularBufferReaders.emplace_back();
                        succeeded = reader.Initialize(sectionDescIndex, fnResolveSection);
                        if (succeeded)
                        {
                            pReader = &reader;
                        }
                        else
                        {
                            circularBufferReaders.pop_back();
                            NVTRACE_LOG_WRN(20, "Circular buffer section reader initialization failed; section left reader-less (sectionIndex=%u sectionTag=%u).", sectionDescIndex, pSectionDesc->sectionTag);
                        }
                        break;
                    }
                    case NvTrace_ProviderBufferSectionType_AdaptiveBuffer:
                    {
                        AdaptiveBufferReader& reader = adaptiveBufferReaders.emplace_back();
                        succeeded = reader.Initialize(sectionDescIndex, fnResolveSection);
                        if (succeeded)
                        {
                            pReader = &reader;
                        }
                        else
                        {
                            adaptiveBufferReaders.pop_back();
                            NVTRACE_LOG_WRN(20, "Adaptive buffer section reader initialization failed; section left reader-less (sectionIndex=%u sectionTag=%u).", sectionDescIndex, pSectionDesc->sectionTag);
                        }
                        break;
                    }
                    case NvTrace_ProviderBufferSectionType_StateTable:
                    {
                        StateTableReader& reader = stateTableReaders.emplace_back();
                        succeeded = reader.Initialize(sectionDescIndex, fnResolveSection);
                        if (succeeded)
                        {
                            pReader = &reader;
                        }
                        else
                        {
                            stateTableReaders.pop_back();
                            NVTRACE_LOG_WRN(20, "State table section reader initialization failed; section left reader-less (sectionIndex=%u sectionTag=%u).", sectionDescIndex, pSectionDesc->sectionTag);
                        }
                        break;
                    }
                    case NvTrace_ProviderBufferSectionType_StateArena:
                    {
                        StateArenaReader& reader = stateArenaReaders.emplace_back();
                        succeeded = reader.Initialize(sectionDescIndex, fnResolveSection);
                        if (succeeded)
                        {
                            pReader = &reader;
                        }
                        else
                        {
                            stateArenaReaders.pop_back();
                            NVTRACE_LOG_WRN(20, "State arena section reader initialization failed; section left reader-less (sectionIndex=%u sectionTag=%u).", sectionDescIndex, pSectionDesc->sectionTag);
                        }
                        break;
                    }
                    case NvTrace_ProviderBufferSectionType_GpuReportBuffer:
                    {
                        GpuReportBufferReader& reader = gpuReportBufferReaders.emplace_back();
                        succeeded = reader.Initialize(sectionDescIndex, fnResolveSection);
                        if (succeeded)
                        {
                            pReader = &reader;
                        }
                        else
                        {
                            gpuReportBufferReaders.pop_back();
                            NVTRACE_LOG_WRN(20, "GPU report buffer section reader initialization failed; section left reader-less (sectionIndex=%u sectionTag=%u).", sectionDescIndex, pSectionDesc->sectionTag);
                        }
                        break;
                    }
                    case NvTrace_ProviderBufferSectionType_AdaptiveChunkPool:
                    case NvTrace_ProviderBufferSectionType_GpuReportChunkPool:
                        break;

                    case NvTrace_ProviderBufferSectionType_Invalid:
                    case NvTrace_ProviderBufferSectionType_FORCE_INT:
                    default:
                        NVTRACE_LOG_WRN(20, "Unrecognized provider buffer section type (sectionIndex=%u sectionType=%u sectionTag=%u).", sectionDescIndex, static_cast<uint32_t>(pSectionDesc->sectionType), pSectionDesc->sectionTag);
                        break;
                }

                NVTRACE_LOG_ERR_RETURN_IF(sectionResolutionFailed, false, 20, "Provider buffer section carries a dangling or wrong-typed cross-section reference (sectionIndex=%u sectionTag=%u).", sectionDescIndex, pSectionDesc->sectionTag);

                sections.emplace_back(pSectionDesc->sectionType, pSectionDesc->sectionTag, sectionDescIndex, pReader);
            }

            uint32_t onAttachRecordsTotalCount = 0;
            for (const DataBlobReader& reader : dataBlobReaders)
            {
                if ((reader.DrainFlags() & NvTrace_DataBlobDrainFlag_OnAttach) != 0)
                {
                    onAttachRecordsTotalCount++;
                }
            }
            for (const StateTableReader& reader : stateTableReaders)
            {
                onAttachRecordsTotalCount += reader.EntryCount();
            }
            for (const StateArenaReader& reader : stateArenaReaders)
            {
                onAttachRecordsTotalCount += reader.GetRecordCount();
            }

            // Commit observer state only now, after all validation has succeeded. The moves preserve the
            // reader addresses the Section handles hold.
            m_pProviderBufferHeader = pProviderBufferHeader;
            m_dataBlobReaders = std::move(dataBlobReaders);
            m_onAttachRecordsTotalCount = onAttachRecordsTotalCount;
            m_circularBufferReaders = std::move(circularBufferReaders);
            m_adaptiveBufferReaders = std::move(adaptiveBufferReaders);
            m_stateTableReaders = std::move(stateTableReaders);
            m_stateArenaReaders = std::move(stateArenaReaders);
            m_gpuReportBufferReaders = std::move(gpuReportBufferReaders);
            m_sections = std::move(sections);

            return true;
        }

        // The reader pointer is non-owning -- the GenericObserver's per-type vectors own it -- and null
        // on a reader-less section.
        class Section
        {
        public:
            Section(NvTrace_ProviderBufferSectionType sectionType, uint32_t sectionTag, uint32_t sectionIndex, void* pReader)
                : m_sectionType(sectionType)
                , m_sectionTag(sectionTag)
                , m_sectionIndex(sectionIndex)
                , m_pReader(pReader)
            {
            }

            NvTrace_ProviderBufferSectionType SectionType() const
            {
                return m_sectionType;
            }

            uint32_t SectionTag() const
            {
                return m_sectionTag;
            }

            uint32_t SectionIndex() const
            {
                return m_sectionIndex;
            }

            DataBlobReader* AsDataBlob() const
            {
                return m_sectionType == NvTrace_ProviderBufferSectionType_DataBlob ? static_cast<DataBlobReader*>(m_pReader) : nullptr;
            }

            CircularBufferReader* AsCircularBuffer() const
            {
                return m_sectionType == NvTrace_ProviderBufferSectionType_CircularBuffer ? static_cast<CircularBufferReader*>(m_pReader) : nullptr;
            }

            AdaptiveBufferReader* AsAdaptiveBuffer() const
            {
                return m_sectionType == NvTrace_ProviderBufferSectionType_AdaptiveBuffer ? static_cast<AdaptiveBufferReader*>(m_pReader) : nullptr;
            }

            StateTableReader* AsStateTable() const
            {
                return m_sectionType == NvTrace_ProviderBufferSectionType_StateTable ? static_cast<StateTableReader*>(m_pReader) : nullptr;
            }

            StateArenaReader* AsStateArena() const
            {
                return m_sectionType == NvTrace_ProviderBufferSectionType_StateArena ? static_cast<StateArenaReader*>(m_pReader) : nullptr;
            }

            // GPU report buffer sections have no generic drain call: their records gate on
            // GPU-derived report refs, so the typed observer that owns them drains them through here.
            GpuReportBufferReader* AsGpuReportBuffer() const
            {
                return m_sectionType == NvTrace_ProviderBufferSectionType_GpuReportBuffer ? static_cast<GpuReportBufferReader*>(m_pReader) : nullptr;
            }

        private:
            NvTrace_ProviderBufferSectionType m_sectionType;
            uint32_t m_sectionTag;
            uint32_t m_sectionIndex;
            void* m_pReader;
        };

        std::vector<Section*> RecordSections()
        {
            std::vector<Section*> sections;
            sections.reserve(m_sections.size());
            for (Section& section : m_sections)
            {
                if (section.SectionType() != NvTrace_ProviderBufferSectionType_AdaptiveChunkPool && section.SectionType() != NvTrace_ProviderBufferSectionType_GpuReportChunkPool && section.SectionType() != NvTrace_ProviderBufferSectionType_Invalid)
                {
                    sections.push_back(&section);
                }
            }
            return sections;
        }

        uint32_t SectionCount() const
        {
            return static_cast<uint32_t>(m_sections.size());
        }

        Section* FindSection(uint32_t sectionTag)
        {
            return const_cast<Section*>(std::as_const(*this).FindSection(sectionTag));
        }

        const Section* FindSection(uint32_t sectionTag) const
        {
            for (const Section& section : m_sections)
            {
                if (section.SectionTag() == sectionTag)
                {
                    return &section;
                }
            }

            return nullptr;
        }

        Section& GetSection(uint32_t sectionIndex)
        {
            return m_sections[sectionIndex];
        }

        const Section& GetSection(uint32_t sectionIndex) const
        {
            return m_sections[sectionIndex];
        }

        uint32_t GetOnAttachRecordsTotalCount() const
        {
            return m_onAttachRecordsTotalCount;
        }

        // Pending DataBlobs remain available for later calls.
        //
        //   void fnReadRecord(uint32_t recordType, const uint8_t* pPayload, size_t payloadSize)
        //     Accepts the DataBlob and marks it consumed.
        //
        //   bool fnReadRecord(uint32_t recordType, const uint8_t* pPayload, size_t payloadSize)
        //     Return true to accept the DataBlob and mark it consumed.
        //     Return false to leave the DataBlob available for a later call.
        template <typename TFN_ReadRecord>
        void ConsumeOnAttachDataBlobRecords(TFN_ReadRecord&& fnReadRecord)
        {
            for (DataBlobReader& reader : m_dataBlobReaders)
            {
                if (reader.IsConsumed())
                {
                    continue;
                }

                if ((reader.DrainFlags() & NvTrace_DataBlobDrainFlag_OnAttach) == 0)
                {
                    continue;
                }

                // Perf: LoadAcquire the status flags; this must be the last check
                if (reader.StatusFlags() != NvTrace_DataBlobStatusFlag_Ready)
                {
                    continue;
                }

                bool recordAccepted = true;
                reader.ReadRecord([&](uint32_t recordType, const uint8_t* pPayload, size_t payloadSize) {
                    if constexpr (std::is_same_v<std::invoke_result_t<TFN_ReadRecord, uint32_t, const uint8_t*, size_t>, void>)
                    {
                        fnReadRecord(recordType, pPayload, payloadSize);
                    }
                    else
                    {
                        recordAccepted = static_cast<bool>(fnReadRecord(recordType, pPayload, payloadSize));
                    }
                });
                if (recordAccepted)
                {
                    reader.MarkConsumed();
                }
            }
        }

        // SnapshotConsumer reuses the same observer while rebuilding the complete provider attachment state for
        // every snapshot. Re-read every ready on-attach DataBlob even if it was consumed by an earlier snapshot;
        // this operation does not inspect or modify the reader's consumed state.
        template <typename TFN_ReadRecord>
        void ConsumeOnAttachDataBlobRecords_IgnoringConsumedState(TFN_ReadRecord&& fnReadRecord)
        {
            for (DataBlobReader& reader : m_dataBlobReaders)
            {
                if ((reader.DrainFlags() & NvTrace_DataBlobDrainFlag_OnAttach) == 0)
                {
                    continue;
                }

                // Perf: LoadAcquire the status flags; this must be the last check
                if (reader.StatusFlags() != NvTrace_DataBlobStatusFlag_Ready)
                {
                    continue;
                }

                reader.ReadRecord(fnReadRecord);
            }
        }

        // Call once, on the final poll. Data blobs carry no consistency protocol, so there is no commit
        // channel; a detach-drained blob may race the producer and is best-effort.
        template <typename TFN_ReadRecord>
        void ConsumeDetachRecords(TFN_ReadRecord&& fnReadRecord)
        {
            for (DataBlobReader& reader : m_dataBlobReaders)
            {
                if ((reader.DrainFlags() & NvTrace_DataBlobDrainFlag_OnDetach) != 0)
                {
                    reader.ReadRecord(fnReadRecord);
                }
            }
        }

        // Call once after attach: every record of every state section, tables and arenas alike, in
        // section order, each followed by fnCommitRecord(bool). A section's own type decides what it
        // yields -- a table entry leads a chunk chain, an arena record stands alone -- so no caller
        // needs to know which record type continues which.
        // fnReadRecord(NvTrace_CorrelationId recordCorrelationId, uint32_t recordType, const uint8_t* pPayload, size_t payloadSize)
        //   -- recordCorrelationId is the record's own address; what it belongs to is in its payload.
        template <typename TFN_ReadRecord, typename TFN_CommitRecord>
        void ConsumeAttachStateRecords(TFN_ReadRecord&& fnReadRecord, TFN_CommitRecord&& fnCommitRecord)
        {
            for (const Section& section : m_sections)
            {
                if (const StateTableReader* const pStateTable = section.AsStateTable())
                {
                    pStateTable->ConsumeAttachRecords(fnReadRecord, fnCommitRecord);
                }
                else if (const StateArenaReader* const pStateArena = section.AsStateArena())
                {
                    pStateArena->ConsumeAttachRecords(fnReadRecord, fnCommitRecord);
                }
            }
        }

        // Call every poll for providers whose recurring records all use circular buffers. Unlike adaptive-buffer
        // records, circular-buffer callbacks may receive the source sequence number.
        void SetObservationStart(const ReadEndpoints& observationStart, bool retainAtLeastOneRecord = false)
        {
            for (CircularBufferReader& reader : m_circularBufferReaders)
            {
                switch (reader.RecordTimestampType())
                {
                    case NvTrace_CircularBufferRecordTimestampType_Tsc:
                        reader.SetObservationStartTimestamp(observationStart.tsc, retainAtLeastOneRecord);
                        break;

                    case NvTrace_CircularBufferRecordTimestampType_GpuPtimer:
                        reader.SetObservationStartTimestamp(observationStart.gpuNsec, retainAtLeastOneRecord);
                        break;

                    default:
                        NVTRACE_LOG_ERR(20, "Unsupported circular-buffer record timestamp type %u.", static_cast<uint32_t>(reader.RecordTimestampType()));
                        break;
                }
            }
        }

        template <typename TFN_ReadRecord, typename TFN_CommitRecord>
        void ConsumeCircularBufferRecords(const ReadEndpoints& readEndpoints, TFN_ReadRecord&& fnReadRecord, TFN_CommitRecord&& fnCommitRecord)
        {
            for (CircularBufferReader& reader : m_circularBufferReaders)
            {
                switch (reader.RecordTimestampType())
                {
                    case NvTrace_CircularBufferRecordTimestampType_Tsc:
                        reader.ConsumeRecords(readEndpoints.tsc, fnReadRecord, fnCommitRecord);
                        break;

                    case NvTrace_CircularBufferRecordTimestampType_GpuPtimer:
                        reader.ConsumeRecords(readEndpoints.gpuNsec, fnReadRecord, fnCommitRecord);
                        break;

                    default:
                        NVTRACE_LOG_ERR(20, "Unsupported circular-buffer record timestamp type %u.", static_cast<uint32_t>(reader.RecordTimestampType()));
                        break;
                }
            }
        }

        // Call every poll for providers whose recurring records use adaptive buffers. Records newer than
        // readEndpointTsc are left for a later call.
        template <typename TFN_ReadRecord, typename TFN_CommitRecord>
        void ConsumeAdaptiveBufferRecords(uint64_t readEndpointTsc, TFN_ReadRecord&& fnReadRecord, TFN_CommitRecord&& fnCommitRecord)
        {
            for (AdaptiveBufferReader& reader : m_adaptiveBufferReaders)
            {
                reader.ConsumeRecords(readEndpointTsc, fnReadRecord, fnCommitRecord);
            }
        }

    private:
        const NvTrace_ProviderBufferHeader* m_pProviderBufferHeader = nullptr;
        std::vector<DataBlobReader> m_dataBlobReaders;
        uint32_t m_onAttachRecordsTotalCount = 0;
        std::vector<CircularBufferReader> m_circularBufferReaders;
        std::vector<AdaptiveBufferReader> m_adaptiveBufferReaders;
        std::vector<StateTableReader> m_stateTableReaders;
        std::vector<StateArenaReader> m_stateArenaReaders;
        std::vector<GpuReportBufferReader> m_gpuReportBufferReaders;
        std::vector<Section> m_sections;
    };

} // namespace nv::trace
