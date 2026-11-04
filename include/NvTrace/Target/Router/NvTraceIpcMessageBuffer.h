/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/DataFormats/nvtrace_common_types.h>
#include <NvTrace/Target/NvTraceFlatBufferUtilities.h>

#include <cstddef>
#include <cstdint>

namespace nv::trace {

    ////////////////////////////////////////////////////////////
    // IpcMessageHeader
    ////////////////////////////////////////////////////////////

    // The layout of a message is platform dependent, although the header is the same on both platforms.
    // On Linux, each message starts with IpcMessageHeader, followed by the message itself. The memory identifiers are sent separately as Unix ancillary data.
    struct IpcMessageHeader
    {
        uint32_t structSize;
        uint32_t messageType;
        uint32_t messageSize;
        uint32_t messageObjectIdentifierCount;
    };


    ////////////////////////////////////////////////////////////
    // IpcMessageBuffer
    ////////////////////////////////////////////////////////////

    // The purpose of this class is to store a single IPC message that is either sent or received, on its way to/from the socket's internal buffer.
    // It's not meant to serve as a backpressure buffer, hence why it can only store a single message. That's the socket's internal buffer's job.
    // This class is shared between both Server and Client.
    class IpcMessageBuffer
    {
    public:
        ~IpcMessageBuffer();
        IpcMessageBuffer();

        IpcMessageBuffer(const IpcMessageBuffer&) = delete;
        IpcMessageBuffer& operator=(const IpcMessageBuffer&) = delete;

        void Reset() noexcept;

        static size_t CalculateRequiredBufferMemorySize(size_t maxMessageSize, uint32_t maxMessageObjectIdentifierCount);
        void Initialize(uint8_t* pBufferMemory, size_t maxMessageSize, uint32_t maxMessageObjectIdentifierCount);

        void Clear() noexcept;

        bool IsEmpty() const
        {
            return m_messageDataSizeInBuffer == 0;
        }

    protected:
#if defined(__linux__)
        void FreeOwnedAncillaryData();
#endif

    protected:
        size_t m_maxMessageSize = 0;
        uint32_t m_maxMessageObjectIdentifierCount = 0;

        uint8_t* m_pMessageBuffer = nullptr;
        size_t m_messageBufferSize = 0;
        size_t m_messageDataSizeInBuffer = 0;

#if defined(__linux__)
        uint8_t* m_pAncillaryBuffer = nullptr;
        size_t m_ancillaryBufferSize = 0;
        size_t m_ownedAncillaryDataSizeInBuffer = 0;
#endif
    };


    ////////////////////////////////////////////////////////////
    // IpcMessageSendBuffer
    ////////////////////////////////////////////////////////////

    using PFN_IpcWriteMessageCallback = void (*)(void* pUserData, uint32_t& messageType, flatbuffers::FlatBufferBuilder& messageBuilder, NvTrace_OsObjectIdentifier* pMessageObjectIdentifiers, uint32_t& messageObjectIdentifierCount);

    // This class provides additional functionality on top of IpcMessageBuffer to encode and send an IPC message.
    class IpcMessageSendBuffer : public IpcMessageBuffer
    {
    public:
        using IpcMessageBuffer::IpcMessageBuffer;

        void Reset() noexcept;
        using IpcMessageBuffer::Initialize;

        void Clear() noexcept;

        // Must only be called when the buffer is empty
        bool WriteMessage(PFN_IpcWriteMessageCallback pfnWriteMessageCallback, void* pUserData);

        const uint8_t* GetUnsentMessageDataPtr() const
        {
            return m_pMessageBuffer + m_messageDataOffsetInBuffer;
        }
        size_t GetUnsentMessageDataSize() const
        {
            return m_messageDataSizeInBuffer;
        }
#if defined(__linux__)
        const uint8_t* GetUnsentAncillaryDataPtr() const
        {
            return m_pAncillaryBuffer;
        }
        size_t GetUnsentAncillaryDataSize() const
        {
            return m_ancillaryDataSizeInBuffer;
        }
#endif

#if defined(__linux__)
        // On failure, the ancillary data is left in an inconsistent state, so the message must be cancelled.
        // Even so, Clear() will free all of the fds that were successfully duplicated, and either way no fds will be leaked
        bool DuplicateAncillaryFds();
#endif

        void SetDataSent(size_t messageDataSizeSent)
        {
            m_messageDataOffsetInBuffer += messageDataSizeSent;
            m_messageDataSizeInBuffer -= messageDataSizeSent;
#if defined(__linux__)
            // Ancillary data is guaranteed to be sent in full on the first successful sendmsg
            FreeOwnedAncillaryData();
            m_ancillaryDataSizeInBuffer = 0;
#endif
        }

        bool IsSendComplete() const
        {
            return m_messageDataSizeInBuffer == 0;
        }

    private:
        size_t m_messageDataOffsetInBuffer = 0;
#if defined(__linux__)
        size_t m_ancillaryDataSizeInBuffer = 0;
#endif
    };


    ////////////////////////////////////////////////////////////
    // IpcMessageReceiveBuffer
    ////////////////////////////////////////////////////////////

    using PFN_IpcReadMessageCallback = bool (*)(void* pUserData, uint32_t messageType, const uint8_t* pMessage, size_t messageSize, const NvTrace_OsObjectIdentifier* pMessageObjectIdentifiers, uint32_t messageObjectIdentifierCount);
    using PFN_IpcReadMessageCallback2 = void (*)(void* pUserData1, void* pUserData2, uint32_t messageType, const uint8_t* pMessage, size_t messageSize, const NvTrace_OsObjectIdentifier* pMessageObjectIdentifiers, uint32_t messageObjectIdentifierCount);

    // This class provides additional functionality on top of IpcMessageBuffer to receive and decode an IPC message.
    class IpcMessageReceiveBuffer : public IpcMessageBuffer
    {
    public:
        using IpcMessageBuffer::IpcMessageBuffer;

        void Reset() noexcept;
        using IpcMessageBuffer::Initialize;

        void Clear() noexcept;

        uint8_t* GetReceiveMessageWindowPtr()
        {
            return m_pMessageBuffer + m_messageDataSizeInBuffer;
        }
        size_t GetReceiveMessageWindowSize()
        {
            return m_messageBufferSize - m_messageDataSizeInBuffer;
        }
#if defined(__linux__)
        uint8_t* GetReceiveAncillaryWindowPtr()
        {
            return m_pAncillaryBuffer;
        }
        size_t GetReceiveAncillaryWindowSize()
        {
            // Ancillary data may only be received in the first packet of the message
            return m_messageDataSizeInBuffer == 0 ? m_ancillaryBufferSize : 0;
        }
#endif

#if defined(__linux__)
        void SetDataReceived(size_t receivedMessageDataSize, size_t receivedAncillaryDataSize, int receivedFlags);
#else
        void SetDataReceived(size_t receivedMessageDataSize);
#endif

        bool IsReceiveComplete() const
        {
            return m_receiveStatus != ReceiveStatus::InProgress;
        }
        bool IsReceiveSuccessful() const
        {
            return m_receiveStatus == ReceiveStatus::Success;
        }

        // Must only be called for a successful receive
        bool ReadMessage(PFN_IpcReadMessageCallback pfnReadMessageCallback, void* pUserData);
        void ReadMessage(PFN_IpcReadMessageCallback2 pfnReadMessageCallback, void* pUserData1, void* pUserData2);

    private:
        enum class ReceiveStatus
        {
            InProgress,
            Success,
            Error
        };
        ReceiveStatus m_receiveStatus = ReceiveStatus::InProgress;
    };

} // namespace nv::trace

#if defined(__linux__)
#include <NvTrace/Target/Router/NvTraceIpcMessageBufferLinux.h>
#endif
