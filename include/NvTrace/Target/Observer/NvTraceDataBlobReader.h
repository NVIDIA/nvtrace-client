/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/DataFormats/nvtrace_provider_buffer.h>
#include <NvTrace/Target/NvTraceClientLogger.h>
#include <NvTrace/Target/NvTraceMemoryFence.h>
#include <NvTrace/Target/Observer/NvTraceSectionLayoutValidator.h>

#include <cinttypes>
#include <cstdint>

namespace nv::trace {

    class DataBlobReader
    {
    public:
        ~DataBlobReader() = default;
        DataBlobReader() noexcept = default;

        DataBlobReader(const DataBlobReader&) = delete;
        DataBlobReader(DataBlobReader&& rhs) noexcept
            : m_recordType(std::exchange(rhs.m_recordType, 0))
            , m_drainFlags(std::exchange(rhs.m_drainFlags, 0))
            , m_pStatusFlags(std::exchange(rhs.m_pStatusFlags, nullptr))
            , m_isConsumed(std::exchange(rhs.m_isConsumed, false))
            , m_pPayload(std::exchange(rhs.m_pPayload, nullptr))
            , m_payloadSize(std::exchange(rhs.m_payloadSize, 0))
        {
        }

        DataBlobReader& operator=(const DataBlobReader&) = delete;
        DataBlobReader& operator=(DataBlobReader&& rhs) noexcept
        {
            if (this != &rhs)
            {
                m_recordType = std::exchange(rhs.m_recordType, 0);
                m_drainFlags = std::exchange(rhs.m_drainFlags, 0);
                m_pStatusFlags = std::exchange(rhs.m_pStatusFlags, nullptr);
                m_isConsumed = std::exchange(rhs.m_isConsumed, false);
                m_pPayload = std::exchange(rhs.m_pPayload, nullptr);
                m_payloadSize = std::exchange(rhs.m_payloadSize, 0);
            }
            return *this;
        }

        void Reset() noexcept
        {
            m_payloadSize = 0;
            m_pPayload = nullptr;
            m_isConsumed = false;
            m_pStatusFlags = nullptr;
            m_drainFlags = 0;
            m_recordType = 0;
        }

        template <typename TFN_ResolveSection>
        bool Initialize(uint32_t dataBlobSectionIndex, TFN_ResolveSection&& fnResolveSection)
        {
            const auto [pSection, sectionSize, sectionHeaderSize, sectionHeaderMinorVersion] = fnResolveSection(dataBlobSectionIndex, NvTrace_ProviderBufferSectionType_DataBlob);
            NVTRACE_LOG_ERR_RETURN_IF(!pSection, false, 50, "DataBlobReader: failed to resolve data blob section (dataBlobSectionIndex %u)", dataBlobSectionIndex);

            SectionLayoutValidator layoutValidator(sectionSize);

            bool succeeded = layoutValidator.AddAndValidateSection(0, sectionHeaderSize, sectionHeaderMinorVersion, NvTrace_DataBlobSectionHeader_MINOR_VERSION_A);
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "DataBlobReader: invalid section header layout (size %u, minorVersion %u)", sectionHeaderSize, sectionHeaderMinorVersion);

            const NvTrace_DataBlobSectionHeader* pSectionHeader = reinterpret_cast<const NvTrace_DataBlobSectionHeader*>(pSection);
            NVTRACE_LOG_ERR_RETURN_IF((reinterpret_cast<uintptr_t>(pSectionHeader) % alignof(NvTrace_DataBlobSectionHeader)) != 0, false, 50, "DataBlobReader: section header must be %zu-byte aligned (pSectionHeader %p)", alignof(NvTrace_DataBlobSectionHeader), pSectionHeader);

            succeeded = layoutValidator.AddAndValidateSection(pSectionHeader->recordDescOffsetInSection, pSectionHeader->recordDescSize, pSectionHeader->recordDescMinorVersion, NvTrace_RecordDesc_MINOR_VERSION_A);
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "DataBlobReader: invalid record desc layout (offset %" PRIu64 ", size %u, minorVersion %u)", pSectionHeader->recordDescOffsetInSection, pSectionHeader->recordDescSize, pSectionHeader->recordDescMinorVersion);

            const NvTrace_RecordDesc* const pRecordDesc = reinterpret_cast<const NvTrace_RecordDesc*>(pSection + pSectionHeader->recordDescOffsetInSection);
            NVTRACE_LOG_ERR_RETURN_IF((reinterpret_cast<uintptr_t>(pRecordDesc) % alignof(NvTrace_RecordDesc)) != 0, false, 50, "DataBlobReader: record desc must be %zu-byte aligned (pRecordDesc %p)", alignof(NvTrace_RecordDesc), pRecordDesc);

            const size_t payloadSize = static_cast<size_t>(pRecordDesc->payloadEndOffsetDiv8) * sizeof(uint64_t);
            succeeded = layoutValidator.AddAndValidateSection(pSectionHeader->payloadOffsetInSection, payloadSize);
            NVTRACE_LOG_ERR_RETURN_IF(!succeeded, false, 50, "DataBlobReader: invalid payload layout (offset %" PRIu64 ", size %zu)", pSectionHeader->payloadOffsetInSection, payloadSize);

            // Section base is header-aligned (>= 8 bytes), so an 8-byte-multiple payload offset keeps pPayload 8-byte aligned.
            NVTRACE_LOG_ERR_RETURN_IF((pSectionHeader->payloadOffsetInSection % alignof(uint64_t)) != 0, false, 50, "DataBlobReader: payload must be %zu-byte aligned (payloadOffsetInSection %" PRIu64 ")", alignof(uint64_t), pSectionHeader->payloadOffsetInSection);

            NVTRACE_LOG_ERR_IF(pSectionHeader->drainFlags == 0, 50, "DataBlobReader: published drain flags name no drain phase; the blob will never be drained (sectionIndex %u, recordType %u)", dataBlobSectionIndex, pRecordDesc->recordType);

            m_recordType = pRecordDesc->recordType;
            m_drainFlags = pSectionHeader->drainFlags;
            m_pStatusFlags = &pSectionHeader->statusFlags;
            m_pPayload = pSection + pSectionHeader->payloadOffsetInSection;
            m_payloadSize = payloadSize;

            return true;
        }

        // NvTrace_DataBlobDrainFlag bits; 0 means a malformed header published no phase (logged at
        // Initialize) and the blob is never consumed.
        uint32_t DrainFlags() const
        {
            return m_drainFlags;
        }

        uint32_t StatusFlags() const
        {
            return LoadAcquire(m_pStatusFlags);
        }

        bool IsConsumed() const
        {
            return m_isConsumed;
        }

        void MarkConsumed()
        {
            m_isConsumed = true;
        }

        // Invokes fnReadRecord(recordType, pPayload, payloadSize). pPayload points into the mapped buffer and
        // is bounds-validated and 8-byte aligned (enforced in Initialize), and payloadSize is a whole number
        // of words.
        template <typename TFN_ReadRecord>
        void ReadRecord(TFN_ReadRecord&& fnReadRecord)
        {
            fnReadRecord(m_recordType, m_pPayload, m_payloadSize);
        }

    private:
        uint32_t m_recordType = 0;
        uint32_t m_drainFlags = 0;

        const volatile uint32_t* m_pStatusFlags = nullptr;
        bool m_isConsumed = false;

        const uint8_t* m_pPayload = nullptr;
        size_t m_payloadSize = 0;
    };

} // namespace nv::trace
