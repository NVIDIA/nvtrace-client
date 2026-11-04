/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/Target/NvTraceClientLogger.h>
#include <NvTrace/Target/NvTraceMathUtilities.h>
#include <NvTrace/Target/Router/NvTraceIpcMessageBuffer.h>

#include <fcntl.h>
#include <sys/socket.h>
#include <sys/unistd.h>

namespace nv::trace {

    ////////////////////////////////////////////////////////////
    // IpcMessageBuffer
    ////////////////////////////////////////////////////////////

    inline IpcMessageBuffer::~IpcMessageBuffer()
    {
        FreeOwnedAncillaryData();
    }

    inline IpcMessageBuffer::IpcMessageBuffer() = default;

    inline void IpcMessageBuffer::Reset() noexcept
    {
        FreeOwnedAncillaryData();
        m_ancillaryBufferSize = 0;
        m_pAncillaryBuffer = nullptr;

        m_messageDataSizeInBuffer = 0;
        m_messageBufferSize = 0;
        m_pMessageBuffer = nullptr;

        m_maxMessageObjectIdentifierCount = 0;
        m_maxMessageSize = 0;
    }

    inline size_t IpcMessageBuffer::CalculateRequiredBufferMemorySize(size_t maxMessageSize, uint32_t maxMessageObjectIdentifierCount)
    {
        // Store both message and ancillary data in the same buffer
        size_t requiredMessageBufferSize = sizeof(IpcMessageHeader) + AlignUpP2(maxMessageSize, alignof(uintptr_t));
        size_t requiredAncillaryBufferSize = maxMessageObjectIdentifierCount != 0 ? CMSG_SPACE(maxMessageObjectIdentifierCount * sizeof(int)) : 0;

        return CMSG_ALIGN(requiredMessageBufferSize) + requiredAncillaryBufferSize;
    }

    inline void IpcMessageBuffer::Initialize(uint8_t* pBufferMemory, size_t maxMessageSize, uint32_t maxMessageObjectIdentifierCount)
    {
        static_assert(sizeof(IpcMessageHeader) % alignof(uintptr_t) == 0);

        m_maxMessageSize = AlignUpP2(maxMessageSize, alignof(uintptr_t));
        m_maxMessageObjectIdentifierCount = maxMessageObjectIdentifierCount;
        m_pMessageBuffer = pBufferMemory;
        m_messageBufferSize = sizeof(IpcMessageHeader) + m_maxMessageSize;

        if (m_maxMessageObjectIdentifierCount != 0)
        {
            m_pAncillaryBuffer = pBufferMemory + CMSG_ALIGN(m_messageBufferSize);
            m_ancillaryBufferSize = CMSG_SPACE(maxMessageObjectIdentifierCount * sizeof(int));
        }
    }

    inline void IpcMessageBuffer::Clear() noexcept
    {
        m_messageDataSizeInBuffer = 0;
        FreeOwnedAncillaryData();
    }

    inline static uint32_t GetAncillaryMessageFdCount(const cmsghdr* pAncillaryMessage)
    {
        return static_cast<uint32_t>((pAncillaryMessage->cmsg_len - CMSG_LEN(0)) / sizeof(int));
    }

    inline static cmsghdr* GetFirstAncillaryMesssage(uint8_t* pAncillaryBuffer, size_t ancillaryDataSize)
    {
        return sizeof(cmsghdr) <= ancillaryDataSize ? reinterpret_cast<cmsghdr*>(pAncillaryBuffer) : nullptr;
    }

    inline static cmsghdr* GetNextAncillaryMesssage(uint8_t* pAncillaryBuffer, size_t ancillaryDataSize, cmsghdr* pAncillaryMessage)
    {
        cmsghdr* pNextAncillaryMessage = reinterpret_cast<cmsghdr*>(reinterpret_cast<uint8_t*>(pAncillaryMessage) + CMSG_ALIGN(pAncillaryMessage->cmsg_len));

        return reinterpret_cast<uint8_t*>(pNextAncillaryMessage + 1) <= pAncillaryBuffer + ancillaryDataSize ? pNextAncillaryMessage : nullptr;
    }

    inline void IpcMessageBuffer::FreeOwnedAncillaryData()
    {
        if (m_ownedAncillaryDataSizeInBuffer != 0)
        {
            // Free file descriptors
            cmsghdr* pAncillaryMessage = GetFirstAncillaryMesssage(m_pAncillaryBuffer, m_ownedAncillaryDataSizeInBuffer);
            while (pAncillaryMessage)
            {
                if (pAncillaryMessage->cmsg_type == SCM_RIGHTS)
                {
                    int* const pFds = reinterpret_cast<int*>(CMSG_DATA(pAncillaryMessage));
                    const uint32_t fdCount = GetAncillaryMessageFdCount(pAncillaryMessage);
                    for (uint32_t fdIndex = 0; fdIndex < fdCount; ++fdIndex)
                    {
                        close(pFds[fdIndex]);
                    }
                }

                pAncillaryMessage = GetNextAncillaryMesssage(m_pAncillaryBuffer, m_ownedAncillaryDataSizeInBuffer, pAncillaryMessage);
            }

            m_ownedAncillaryDataSizeInBuffer = 0;
        }
    }

    ////////////////////////////////////////////////////////////
    // IpcMessageSendBuffer
    ////////////////////////////////////////////////////////////

    inline void IpcMessageSendBuffer::Reset() noexcept
    {
        m_ancillaryDataSizeInBuffer = 0;
        m_messageDataOffsetInBuffer = 0;
        IpcMessageBuffer::Reset();
    }

    inline void IpcMessageSendBuffer::Clear() noexcept
    {
        IpcMessageBuffer::Clear();
        m_messageDataOffsetInBuffer = 0;
        m_ancillaryDataSizeInBuffer = 0;
    }

    inline bool IpcMessageSendBuffer::WriteMessage(PFN_IpcWriteMessageCallback pfnWriteMessageCallback, void* pUserData)
    {
        NVTRACE_LOG_ERR_RETURN_IF(m_messageDataSizeInBuffer != 0, false, 50, "WriteMessage called when data from the previous message is still in the buffer");

        cmsghdr* pScmRightsMessage = reinterpret_cast<cmsghdr*>(m_pAncillaryBuffer);

        uint32_t messageType = 0;
        size_t messageSize = 0;
        uint32_t messageObjectIdentifierCount = 0;
        {
            FlatBufferTryContext tryContext;
            if (tryContext.Try())
            {
                // Call the callback to write the message
                FlatBufferFixedAllocator allocator(tryContext, m_pMessageBuffer + sizeof(IpcMessageHeader), m_maxMessageSize);
                flatbuffers::FlatBufferBuilder messageBuilder(m_maxMessageSize, &allocator);
                NvTrace_OsObjectIdentifier* const pMessageObjectIdentifiers = pScmRightsMessage ? reinterpret_cast<int*>(CMSG_DATA(pScmRightsMessage)) : nullptr;

                pfnWriteMessageCallback(pUserData, messageType, messageBuilder, pMessageObjectIdentifiers, messageObjectIdentifierCount);
                messageSize = messageBuilder.GetSize();
            }
            else
            {
                // Arrived here via longjmp - an error occurred
                return false;
            }
        }

        NVTRACE_LOG_ERR_RETURN_IF(m_maxMessageObjectIdentifierCount < messageObjectIdentifierCount, false, 50, "WriteMessage callback wrote %u object identifiers, past the %u the buffer was initialized for", messageObjectIdentifierCount, m_maxMessageObjectIdentifierCount);

        // Write message header
        const size_t messageHeaderOffset = m_maxMessageSize - messageSize;
        IpcMessageHeader* const pMessageHeader = reinterpret_cast<IpcMessageHeader*>(m_pMessageBuffer + messageHeaderOffset);
        pMessageHeader->structSize = sizeof(IpcMessageHeader);
        pMessageHeader->messageType = messageType;
        pMessageHeader->messageSize = static_cast<uint32_t>(messageSize);
        pMessageHeader->messageObjectIdentifierCount = messageObjectIdentifierCount;

        // Mark that there is data in the message buffer
        m_messageDataOffsetInBuffer = messageHeaderOffset;
        m_messageDataSizeInBuffer = sizeof(IpcMessageHeader) + messageSize;

        if (messageObjectIdentifierCount != 0)
        {
            // Write the cmsg header
            pScmRightsMessage->cmsg_len = CMSG_LEN(messageObjectIdentifierCount * sizeof(int));
            pScmRightsMessage->cmsg_level = SOL_SOCKET;
            pScmRightsMessage->cmsg_type = SCM_RIGHTS;

            // Mark that there is data in the ancillary buffer
            m_ancillaryDataSizeInBuffer = CMSG_SPACE(messageObjectIdentifierCount * sizeof(int));
        }

        return true;
    }

    inline bool IpcMessageSendBuffer::DuplicateAncillaryFds()
    {
        if (m_ancillaryDataSizeInBuffer != 0)
        {
            cmsghdr* pScmRightsMessage = reinterpret_cast<cmsghdr*>(m_pAncillaryBuffer);
            int* const pFds = reinterpret_cast<int*>(CMSG_DATA(pScmRightsMessage));
            const uint32_t fdCount = GetAncillaryMessageFdCount(pScmRightsMessage);

            for (uint32_t fdIndex = 0; fdIndex < fdCount; ++fdIndex)
            {
                int ownedFd = fcntl(pFds[fdIndex], F_DUPFD_CLOEXEC, 0);
                if (ownedFd == -1)
                {
                    return false;
                }

                pFds[fdIndex] = ownedFd;
                pScmRightsMessage->cmsg_len = CMSG_LEN((fdIndex + 1) * sizeof(int));
                m_ownedAncillaryDataSizeInBuffer = CMSG_SPACE((fdIndex + 1) * sizeof(int));
            }
        }

        return true;
    }


    ////////////////////////////////////////////////////////////
    // IpcMessageReceiveBuffer
    ////////////////////////////////////////////////////////////

    inline void IpcMessageReceiveBuffer::Reset() noexcept
    {
        m_receiveStatus = ReceiveStatus::InProgress;
        IpcMessageBuffer::Reset();
    }

    inline void IpcMessageReceiveBuffer::Clear() noexcept
    {
        IpcMessageBuffer::Clear();
        m_receiveStatus = ReceiveStatus::InProgress;
    }

    inline void IpcMessageReceiveBuffer::SetDataReceived(size_t receivedMessageDataSize, size_t receivedAncillaryDataSize, int receivedFlags)
    {
        bool validateMessageHeaderAndAncillaryData = m_messageDataSizeInBuffer < sizeof(IpcMessageHeader);
        m_messageDataSizeInBuffer += receivedMessageDataSize;
        m_ownedAncillaryDataSizeInBuffer += receivedAncillaryDataSize;

        // Update receive status
        if (receivedFlags & (MSG_TRUNC | MSG_CTRUNC))
        {
            NVTRACE_LOG_ERR(50, "Received message was truncated");
            m_receiveStatus = ReceiveStatus::Error;
            return;
        }

        if (m_messageDataSizeInBuffer >= sizeof(IpcMessageHeader))
        {
            const IpcMessageHeader* pMessageHeader = reinterpret_cast<const IpcMessageHeader*>(m_pMessageBuffer);
            if (validateMessageHeaderAndAncillaryData)
            {
                // clang-format off
                if (pMessageHeader->structSize != sizeof(IpcMessageHeader) ||
                    pMessageHeader->messageSize > m_maxMessageSize ||
                    pMessageHeader->messageObjectIdentifierCount > m_maxMessageObjectIdentifierCount)
                // clang-format on
                {
                    NVTRACE_LOG_ERR(50, "Received message header is corrupted or the message is too long");
                    m_receiveStatus = ReceiveStatus::Error;
                    return;
                }

                cmsghdr* pAncillaryMessage = GetFirstAncillaryMesssage(m_pAncillaryBuffer, m_ownedAncillaryDataSizeInBuffer);
                if (pMessageHeader->messageObjectIdentifierCount != 0)
                {
                    // clang-format off
                    if (!pAncillaryMessage ||
                        pAncillaryMessage->cmsg_len != CMSG_LEN(pMessageHeader->messageObjectIdentifierCount * sizeof(int)) ||
                        pAncillaryMessage->cmsg_level != SOL_SOCKET ||
                        pAncillaryMessage->cmsg_type != SCM_RIGHTS)
                    // clang-format on
                    {
                        NVTRACE_LOG_ERR(50, "Received a mismatching or unexpected number of memory identifiers");
                        m_receiveStatus = ReceiveStatus::Error;
                        return;
                    }

                    pAncillaryMessage = GetNextAncillaryMesssage(m_pAncillaryBuffer, m_ownedAncillaryDataSizeInBuffer, pAncillaryMessage);
                }
                if (pAncillaryMessage)
                {
                    NVTRACE_LOG_ERR(50, "Received unexpected ancillary data");
                    m_receiveStatus = ReceiveStatus::Error;
                    return;
                }
            }

            size_t totalMessageSize = sizeof(IpcMessageHeader) + pMessageHeader->messageSize;
            if (m_messageDataSizeInBuffer > totalMessageSize)
            {
                NVTRACE_LOG_ERR(50, "Received message is longer than expected");
                m_receiveStatus = ReceiveStatus::Error;
                return;
            }

            if (m_messageDataSizeInBuffer == totalMessageSize)
            {
                m_receiveStatus = ReceiveStatus::Success;
            }
        }
    }

    inline bool IpcMessageReceiveBuffer::ReadMessage(PFN_IpcReadMessageCallback pfnReadMessageCallback, void* pUserData)
    {
        NVTRACE_LOG_ERR_RETURN_IF(m_receiveStatus != ReceiveStatus::Success, false, 50, "ReadMessage called for incomplete or failed receive operation");

        const IpcMessageHeader* pMessageHeader = reinterpret_cast<const IpcMessageHeader*>(m_pMessageBuffer);
        const uint8_t* pMessage = reinterpret_cast<const uint8_t*>(pMessageHeader + 1);
        const NvTrace_OsObjectIdentifier* pMessageObjectIdentifiers = pMessageHeader->messageObjectIdentifierCount != 0 ? reinterpret_cast<const int*>(CMSG_DATA(GetFirstAncillaryMesssage(m_pAncillaryBuffer, m_ownedAncillaryDataSizeInBuffer))) : nullptr;

        return pfnReadMessageCallback(pUserData, pMessageHeader->messageType, pMessage, pMessageHeader->messageSize, pMessageObjectIdentifiers, pMessageHeader->messageObjectIdentifierCount);
    }

    inline void IpcMessageReceiveBuffer::ReadMessage(PFN_IpcReadMessageCallback2 pfnReadMessageCallback, void* pUserData1, void* pUserData2)
    {
        NVTRACE_LOG_ERR_RETURN_IF(m_receiveStatus != ReceiveStatus::Success, /*void*/, 50, "ReadMessage called for incomplete or failed receive operation");

        const IpcMessageHeader* pMessageHeader = reinterpret_cast<const IpcMessageHeader*>(m_pMessageBuffer);
        const uint8_t* pMessage = reinterpret_cast<const uint8_t*>(pMessageHeader + 1);
        const NvTrace_OsObjectIdentifier* pMessageObjectIdentifiers = pMessageHeader->messageObjectIdentifierCount != 0 ? reinterpret_cast<const int*>(CMSG_DATA(GetFirstAncillaryMesssage(m_pAncillaryBuffer, m_ownedAncillaryDataSizeInBuffer))) : nullptr;

        return pfnReadMessageCallback(pUserData1, pUserData2, pMessageHeader->messageType, pMessage, pMessageHeader->messageSize, pMessageObjectIdentifiers, pMessageHeader->messageObjectIdentifierCount);
    }

} // namespace nv::trace
