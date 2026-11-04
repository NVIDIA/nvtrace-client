/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/DataFormats/nvtrace_cpu_clock_calibration.h>
#include <NvTrace/Target/NvTraceClientLogger.h>
#include <NvTrace/Target/NvTraceScopeExitGuard.h>
#include <NvTrace/Target/Router/NvTraceOsGpuAccess.h>
#include <NvTrace/Target/Router/NvTraceRouterClient.h>
#include <NvTrace/Target/Router/NvTraceRouterObserverClientTypes.h>
#include <NvTrace/Target/Router/NvTraceRouterObserverInterface_generated.h>

#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace nv::trace {

    class RouterObserverClient : private RouterClient
    {
    public:
        ~RouterObserverClient() = default;
        RouterObserverClient() = default;

        NvTrace_WaitableOsHandle GetWaitForPendingNotificationHandle() const
        {
            return IpcClient::GetWaitForPendingReceiveHandle();
        }

        template <typename TFN_NotificationCallback>
        bool ReceiveNotification(TFN_NotificationCallback&& fnNotificationCallback)
        {
            using NotificationCallback = std::remove_reference_t<TFN_NotificationCallback>;
            return ReceiveNotification([](void* pUserData, enum NvTrace_RouterObserverClient_NotificationType notificationType, const void* pNotification) { return (*reinterpret_cast<NotificationCallback*>(pUserData))(notificationType, pNotification); }, &fnNotificationCallback);
        }

        template <typename TFN_NotificationCallback>
        bool ReceiveNotificationIfPending(TFN_NotificationCallback&& fnNotificationCallback)
        {
            using NotificationCallback = std::remove_reference_t<TFN_NotificationCallback>;
            return ReceiveNotificationIfPending([](void* pUserData, enum NvTrace_RouterObserverClient_NotificationType notificationType, const void* pNotification) { return (*reinterpret_cast<NotificationCallback*>(pUserData))(notificationType, pNotification); }, &fnNotificationCallback);
        }

        bool HasPendingNotification() const
        {
            return IpcClient::HasPendingMessage();
        }

        bool Initialize(const NvTrace_RouterObserverClient_ProviderTypeFilter* pProviderTypeFilter, const NvTrace_RouterObserverClient_GpuPartitionFilter* pGpuPartitionFilter, const NvTrace_RouterObserverClient_ProcessFilter* pProcessFilter, uint32_t timeoutMsec, NvTrace_CPU_ClockCalibration_TscRtc_CalibrationPoint* out_pInitialTscRtcCalibrationPoint, NvTrace_RouterObserverClient_RouterConfig* out_pRouterConfig = nullptr)
        {
            using namespace FB;
            using namespace FB::RouterObserverInterface;

            auto resetGuard = ScopeExitGuard([this]() { Reset(); });

            if (!RouterClient::Initialize(ROUTER_OBSERVER_ENDPOINT_DESC, timeoutMsec))
            {
                return false;
            }

#if defined(__linux__)
            UniqueLinuxFd accessibleGpus[MAX_ACCESSIBLE_GPU_COUNT];
            const uint32_t accessibleGpuCount = OpenAccessibleGpuIdentifiers(ROUTER_OBSERVER_ENDPOINT_DESC.maxMessageObjectIdentifierCount, accessibleGpus);
#endif

            bool succeeded = IpcClient::SendMessage([&](uint32_t& messageType, flatbuffers::FlatBufferBuilder& messageBuilder, NvTrace_OsObjectIdentifier* pMessageObjectIdentifiers, uint32_t& messageObjectIdentifierCount) {
                messageType = RequestType_BeginObserverClientSession;

                flatbuffers::Offset<ProviderTypeFilter> providerTypeFilterOffset = 0;
                if (pProviderTypeFilter)
                {
                    auto providerTypeUuids = messageBuilder.CreateVectorOfStructs<UUID>(reinterpret_cast<const UUID*>(pProviderTypeFilter->pProviderTypeUuids), pProviderTypeFilter->providerTypeUuidCount);
                    providerTypeFilterOffset = CreateProviderTypeFilter(messageBuilder, providerTypeUuids);
                }

                flatbuffers::Offset<GpuPartitionFilter> gpuPartitionFilterOffset = 0;
                if (pGpuPartitionFilter)
                {
                    auto gpuPartitionUuids = messageBuilder.CreateVectorOfStructs<UUID>(reinterpret_cast<const UUID*>(pGpuPartitionFilter->pGpuPartitionUuids), pGpuPartitionFilter->gpuPartitionUuidCount);
                    gpuPartitionFilterOffset = CreateGpuPartitionFilter(messageBuilder, gpuPartitionUuids);
                }

                flatbuffers::Offset<ProcessFilter> processFilterOffset = 0;
                if (pProcessFilter)
                {
                    auto pids = messageBuilder.CreateVectorOfStructs<PID>(reinterpret_cast<const PID*>(pProcessFilter->pPids), pProcessFilter->pidCount);
                    processFilterOffset = CreateProcessFilter(messageBuilder, pids);
                }

                flatbuffers::Offset<DeclareAccessibleGpusRequest> declareAccessibleGpusRequestOffset = 0;
#if defined(__linux__)
                declareAccessibleGpusRequestOffset = WriteAccessibleGpuDeclaration(messageBuilder, accessibleGpus, accessibleGpuCount, pMessageObjectIdentifiers, messageObjectIdentifierCount);
#endif

                auto messageOffset = CreateBeginObserverClientSessionRequest(messageBuilder, providerTypeFilterOffset, gpuPartitionFilterOffset, processFilterOffset, declareAccessibleGpusRequestOffset);
                messageBuilder.Finish(messageOffset);
            });
            if (!succeeded)
            {
                NVTRACE_LOG_ERR(50, "Failed to send BeginObserverClientSessionRequest");
                return false;
            }

            bool beginObserverClientSessionSucceeded = false;
            succeeded = IpcClient::ReceiveMessage([&beginObserverClientSessionSucceeded, out_pInitialTscRtcCalibrationPoint, out_pRouterConfig](uint32_t messageType, const uint8_t* pMessage, size_t messageSize, const NvTrace_OsObjectIdentifier* pMessageObjectIdentifiers, uint32_t messageObjectIdentifierCount) {
                if (messageType != ResponseType_BeginObserverClientSession || !flatbuffers::Verifier(pMessage, messageSize).VerifyBuffer<BeginObserverClientSessionResponse>())
                {
                    NVTRACE_LOG_ERR(50, "Received unexpected message of type %u while waiting for BeginObserverClientSessionResponse", messageType);
                    return false;
                }
                const BeginObserverClientSessionResponse* pResponse = flatbuffers::GetRoot<BeginObserverClientSessionResponse>(pMessage);

                beginObserverClientSessionSucceeded = pResponse->success();

                if (out_pRouterConfig)
                {
                    out_pRouterConfig->maxGpuPartitionCount = pResponse->maxGpuPartitionCount();
                    out_pRouterConfig->maxProcessCount = pResponse->maxProcessCount();
                    out_pRouterConfig->maxProviderCount = pResponse->maxProviderCount();
                }
                if (out_pInitialTscRtcCalibrationPoint)
                {
                    const FB::TscRtcCalibrationPoint* initialTscRtcCalibrationPoint = pResponse->initialTscRtcCalibrationPoint();
                    if (initialTscRtcCalibrationPoint)
                    {
                        out_pInitialTscRtcCalibrationPoint->tscTick = initialTscRtcCalibrationPoint->tscTick();
                        out_pInitialTscRtcCalibrationPoint->utcNsec = initialTscRtcCalibrationPoint->utcNsec();
                        out_pInitialTscRtcCalibrationPoint->tscToUtcFactor = initialTscRtcCalibrationPoint->tscToUtcFactor();
                        out_pInitialTscRtcCalibrationPoint->utcToTscFactor = initialTscRtcCalibrationPoint->utcToTscFactor();
                    }
                }

                return true;
            });
            if (!succeeded)
            {
                NVTRACE_LOG_ERR(50, "Failed to receive BeginObserverClientSessionResponse");
                return false;
            }

            if (!beginObserverClientSessionSucceeded)
            {
                NVTRACE_LOG_ERR(50, "BeginObserverClientSessionRequest rejected by the router");
                return false;
            }

            NVTRACE_LOG_INF(50, "Observer client session begun with the router");
            resetGuard.Dismiss();
            return true;
        }

        // Proves which GPUs this process can reach at the OS level; may be called again mid-session to prove more.
        bool DeclareAccessibleGpus()
        {
#if defined(__linux__)
            using namespace FB::RouterObserverInterface;

            UniqueLinuxFd accessibleGpus[MAX_ACCESSIBLE_GPU_COUNT];
            const uint32_t accessibleGpuCount = OpenAccessibleGpuIdentifiers(ROUTER_OBSERVER_ENDPOINT_DESC.maxMessageObjectIdentifierCount, accessibleGpus);

            bool succeeded = IpcClient::SendMessage([&accessibleGpus, accessibleGpuCount](uint32_t& messageType, flatbuffers::FlatBufferBuilder& messageBuilder, NvTrace_OsObjectIdentifier* pMessageObjectIdentifiers, uint32_t& messageObjectIdentifierCount) {
                messageType = RequestType_DeclareAccessibleGpus;

                auto messageOffset = WriteAccessibleGpuDeclaration(messageBuilder, accessibleGpus, accessibleGpuCount, pMessageObjectIdentifiers, messageObjectIdentifierCount);
                messageBuilder.Finish(messageOffset);
            });
            if (!succeeded)
            {
                NVTRACE_LOG_ERR(50, "Failed to send DeclareAccessibleGpusRequest");
                return false;
            }

            NVTRACE_LOG_INF(60, "Declared %u OS-reachable GPUs to the router", accessibleGpuCount);
#endif
            return true;
        }

        // A notification this client cannot read is logged and skipped, so only a lost connection reports false
        bool ReceiveNotification(NvTrace_PFN_RouterObserverClient_NotificationCallback pfnNotificationCallback, void* pCallbackUserData)
        {
            return IpcClient::ReceiveMessage([this, pfnNotificationCallback, pCallbackUserData](uint32_t messageType, const uint8_t* pMessage, size_t messageSize, const NvTrace_OsObjectIdentifier* pMessageObjectIdentifiers, uint32_t messageObjectIdentifierCount) {
                ReadMessage(messageType, pMessage, messageSize, pMessageObjectIdentifiers, messageObjectIdentifierCount, pfnNotificationCallback, pCallbackUserData);
                return true;
            });
        }

        bool ReceiveNotificationIfPending(NvTrace_PFN_RouterObserverClient_NotificationCallback pfnNotificationCallback, void* pCallbackUserData)
        {
            return IpcClient::ReceiveMessageIfPending([this, pfnNotificationCallback, pCallbackUserData](uint32_t messageType, const uint8_t* pMessage, size_t messageSize, const NvTrace_OsObjectIdentifier* pMessageObjectIdentifiers, uint32_t messageObjectIdentifierCount) {
                ReadMessage(messageType, pMessage, messageSize, pMessageObjectIdentifiers, messageObjectIdentifierCount, pfnNotificationCallback, pCallbackUserData);
                return true;
            });
        }

    private:
#if defined(__linux__)
        // The Router gets the GPUs from the message's object identifiers, so this table only records their range.
        // Appends after any existing identifiers, omitting GPUs that do not fit the endpoint's identifier capacity.
        static flatbuffers::Offset<FB::RouterObserverInterface::DeclareAccessibleGpusRequest> WriteAccessibleGpuDeclaration(flatbuffers::FlatBufferBuilder& messageBuilder, const UniqueLinuxFd* pGpuIdentifiers, uint32_t gpuIdentifierCount, NvTrace_OsObjectIdentifier* pMessageObjectIdentifiers, uint32_t& messageObjectIdentifierCount)
        {
            using namespace FB::RouterObserverInterface;

            constexpr uint32_t identifierCapacity = ROUTER_OBSERVER_ENDPOINT_DESC.maxMessageObjectIdentifierCount;
            const uint32_t firstGpuIdentifierIndex = messageObjectIdentifierCount;
            const uint32_t freeIdentifierCount = firstGpuIdentifierIndex < identifierCapacity ? identifierCapacity - firstGpuIdentifierIndex : 0;
            if (freeIdentifierCount < gpuIdentifierCount)
            {
                NVTRACE_LOG_ERR(50, "Only %u of the %u accessible GPUs fit in the message, so the rest cannot be declared", freeIdentifierCount, gpuIdentifierCount);
                gpuIdentifierCount = freeIdentifierCount;
            }

            for (uint32_t gpuIndex = 0; gpuIndex < gpuIdentifierCount; ++gpuIndex)
            {
                pMessageObjectIdentifiers[firstGpuIdentifierIndex + gpuIndex] = pGpuIdentifiers[gpuIndex];
            }
            messageObjectIdentifierCount = firstGpuIdentifierIndex + gpuIdentifierCount;

            return CreateDeclareAccessibleGpusRequest(messageBuilder, firstGpuIdentifierIndex, gpuIdentifierCount);
        }
#endif

        bool ReadMessage(uint32_t messageType, const uint8_t* pMessage, size_t messageSize, const NvTrace_OsObjectIdentifier* pMessageObjectIdentifiers, uint32_t messageObjectIdentifierCount, NvTrace_PFN_RouterObserverClient_NotificationCallback pfnNotificationCallback, void* pCallbackUserData)
        {
            using namespace FB::RouterObserverInterface;

            switch (messageType)
            {
                case NotificationType_InitialStateSent:
                {
                    if (!flatbuffers::Verifier(pMessage, messageSize).VerifyBuffer<InitialStateSentNotification>())
                    {
                        break;
                    }

                    NvTrace_RouterObserverClient_InitialStateSentNotification notification = {sizeof(NvTrace_RouterObserverClient_InitialStateSentNotification)};

                    return pfnNotificationCallback(pCallbackUserData, NvTrace_RouterObserverClient_NotificationType_InitialStateSent, &notification);
                }

                case NotificationType_SystemCreated:
                {
                    if (!flatbuffers::Verifier(pMessage, messageSize).VerifyBuffer<SystemCreatedNotification>())
                    {
                        break;
                    }
                    const SystemCreatedNotification* pFbNotification = flatbuffers::GetRoot<SystemCreatedNotification>(pMessage);

                    NvTrace_OsObjectIdentifier infoMemoryIdentifier;
                    if (uint32_t identifierIndex = pFbNotification->infoMemoryIdentifierIndex(); identifierIndex < messageObjectIdentifierCount)
                    {
                        infoMemoryIdentifier = pMessageObjectIdentifiers[identifierIndex];
                    }
                    else
                    {
                        NVTRACE_LOG_ERR(50, "Received SystemCreatedNotification with no payload memory identifier");
                        break;
                    }

                    NvTrace_RouterObserverClient_SystemCreatedNotification notification = {sizeof(NvTrace_RouterObserverClient_SystemCreatedNotification)};
                    notification.infoMemoryIdentifier = infoMemoryIdentifier;
                    notification.infoSize = pFbNotification->infoSize();

                    return pfnNotificationCallback(pCallbackUserData, NvTrace_RouterObserverClient_NotificationType_SystemCreated, &notification);
                }

                case NotificationType_GpuPartitionCreated:
                {
                    if (!flatbuffers::Verifier(pMessage, messageSize).VerifyBuffer<GpuPartitionCreatedNotification>())
                    {
                        break;
                    }
                    const GpuPartitionCreatedNotification* pFbNotification = flatbuffers::GetRoot<GpuPartitionCreatedNotification>(pMessage);

                    NvTrace_OsObjectIdentifier infoMemoryIdentifier;
                    if (uint32_t identifierIndex = pFbNotification->infoMemoryIdentifierIndex(); identifierIndex < messageObjectIdentifierCount)
                    {
                        infoMemoryIdentifier = pMessageObjectIdentifiers[identifierIndex];
                    }
                    else
                    {
                        NVTRACE_LOG_ERR(50, "Received GpuPartitionCreatedNotification with no payload memory identifier");
                        break;
                    }

                    NvTrace_RouterObserverClient_GpuPartitionCreatedNotification notification = {sizeof(NvTrace_RouterObserverClient_GpuPartitionCreatedNotification)};
                    notification.gpuPartitionLuid = *reinterpret_cast<const NVTRACE_LUID*>(pFbNotification->gpuPartitionLuid());
                    notification.gpuUuid = *reinterpret_cast<const NVTRACE_UUID*>(pFbNotification->gpuUuid());
                    notification.creationTimestampTsc = pFbNotification->creationTimestampTsc();
                    notification.infoMemoryIdentifier = infoMemoryIdentifier;
                    notification.infoSize = pFbNotification->infoSize();

                    return pfnNotificationCallback(pCallbackUserData, NvTrace_RouterObserverClient_NotificationType_GpuPartitionCreated, &notification);
                }

                case NotificationType_GpuPartitionDestroyed:
                {
                    if (!flatbuffers::Verifier(pMessage, messageSize).VerifyBuffer<GpuPartitionDestroyedNotification>())
                    {
                        break;
                    }
                    const GpuPartitionDestroyedNotification* pFbNotification = flatbuffers::GetRoot<GpuPartitionDestroyedNotification>(pMessage);

                    NvTrace_RouterObserverClient_GpuPartitionDestroyedNotification notification = {sizeof(NvTrace_RouterObserverClient_GpuPartitionDestroyedNotification)};
                    notification.gpuPartitionLuid = *reinterpret_cast<const NVTRACE_LUID*>(pFbNotification->gpuPartitionLuid());
                    notification.destructionTimestampTsc = pFbNotification->destructionTimestampTsc();

                    return pfnNotificationCallback(pCallbackUserData, NvTrace_RouterObserverClient_NotificationType_GpuPartitionDestroyed, &notification);
                }

                case NotificationType_ProcessCreated:
                {
                    if (!flatbuffers::Verifier(pMessage, messageSize).VerifyBuffer<ProcessCreatedNotification>())
                    {
                        break;
                    }
                    const ProcessCreatedNotification* pFbNotification = flatbuffers::GetRoot<ProcessCreatedNotification>(pMessage);

                    NvTrace_RouterObserverClient_ProcessCreatedNotification notification = {sizeof(NvTrace_RouterObserverClient_ProcessCreatedNotification)};
                    notification.processLuid = *reinterpret_cast<const NVTRACE_LUID*>(pFbNotification->processLuid());
                    notification.namespacePid = *reinterpret_cast<const NVTRACE_PID*>(pFbNotification->namespacePid());
                    // globalPid is optional; null (unobtainable) => 0, consumer falls back to namespacePid.
                    const auto* pFbGlobalPid = pFbNotification->globalPid();
                    notification.globalPid = pFbGlobalPid ? *reinterpret_cast<const NVTRACE_PID*>(pFbGlobalPid) : 0;
                    // Optional struct field; null (a buffer that omitted it) => {0,0}.
                    const auto* pFbPidNamespaceId = pFbNotification->pidNamespaceId();
                    notification.pidNamespaceId = pFbPidNamespaceId ? NvTrace_LinuxNamespaceId{pFbPidNamespaceId->inode(), pFbPidNamespaceId->device()} : NvTrace_LinuxNamespaceId{};
                    notification.creationTimestampTsc = pFbNotification->creationTimestampTsc();

                    return pfnNotificationCallback(pCallbackUserData, NvTrace_RouterObserverClient_NotificationType_ProcessCreated, &notification);
                }

                case NotificationType_ProcessDestroyed:
                {
                    if (!flatbuffers::Verifier(pMessage, messageSize).VerifyBuffer<ProcessDestroyedNotification>())
                    {
                        break;
                    }
                    const ProcessDestroyedNotification* pFbNotification = flatbuffers::GetRoot<ProcessDestroyedNotification>(pMessage);

                    NvTrace_RouterObserverClient_ProcessDestroyedNotification notification = {sizeof(NvTrace_RouterObserverClient_ProcessDestroyedNotification)};
                    notification.processLuid = *reinterpret_cast<const NVTRACE_LUID*>(pFbNotification->processLuid());
                    notification.destructionTimestampTsc = pFbNotification->destructionTimestampTsc();

                    return pfnNotificationCallback(pCallbackUserData, NvTrace_RouterObserverClient_NotificationType_ProcessDestroyed, &notification);
                }

                case NotificationType_ProviderCreated:
                {
                    if (!flatbuffers::Verifier(pMessage, messageSize).VerifyBuffer<ProviderCreatedNotification>())
                    {
                        break;
                    }
                    const ProviderCreatedNotification* pFbNotification = flatbuffers::GetRoot<ProviderCreatedNotification>(pMessage);

                    NvTrace_OsObjectIdentifier providerBufferMemoryIdentifier;
                    if (uint32_t identifierIndex = pFbNotification->providerBufferMemoryIdentifierIndex(); identifierIndex < messageObjectIdentifierCount)
                    {
                        providerBufferMemoryIdentifier = pMessageObjectIdentifiers[identifierIndex];
                    }
                    else
                    {
                        NVTRACE_LOG_ERR(50, "Received ProviderCreatedNotification with no provider buffer memory identifier");
                        break;
                    }

                    NvTrace_RouterObserverClient_ProviderCreatedNotification notification = {sizeof(NvTrace_RouterObserverClient_ProviderCreatedNotification)};
                    notification.providerLuid = *reinterpret_cast<const NVTRACE_LUID*>(pFbNotification->providerLuid());
                    notification.parentLuid = pFbNotification->parentLuid() ? *reinterpret_cast<const NVTRACE_LUID*>(pFbNotification->parentLuid()) : NVTRACE_NIL_LUID;
                    notification.providerTypeUuid = *reinterpret_cast<const NVTRACE_UUID*>(pFbNotification->providerTypeUuid());
                    notification.pProviderName = pFbNotification->providerName() ? pFbNotification->providerName()->c_str() : "";
                    notification.providerBufferMemoryIdentifier = providerBufferMemoryIdentifier;
                    notification.providerBufferSize = pFbNotification->providerBufferSize();

                    return pfnNotificationCallback(pCallbackUserData, NvTrace_RouterObserverClient_NotificationType_ProviderCreated, &notification);
                }

                case NotificationType_ProviderDestroyed:
                {
                    if (!flatbuffers::Verifier(pMessage, messageSize).VerifyBuffer<ProviderDestroyedNotification>())
                    {
                        break;
                    }
                    const ProviderDestroyedNotification* pFbNotification = flatbuffers::GetRoot<ProviderDestroyedNotification>(pMessage);

                    NvTrace_RouterObserverClient_ProviderDestroyedNotification notification = {sizeof(NvTrace_RouterObserverClient_ProviderDestroyedNotification)};
                    notification.providerLuid = *reinterpret_cast<const NVTRACE_LUID*>(pFbNotification->providerLuid());
                    notification.parentLuid = pFbNotification->parentLuid() ? *reinterpret_cast<const NVTRACE_LUID*>(pFbNotification->parentLuid()) : NVTRACE_NIL_LUID;
                    notification.providerTypeUuid = *reinterpret_cast<const NVTRACE_UUID*>(pFbNotification->providerTypeUuid());

                    return pfnNotificationCallback(pCallbackUserData, NvTrace_RouterObserverClient_NotificationType_ProviderDestroyed, &notification);
                }

                default:
                    break;
            }

            NVTRACE_LOG_ERR(50, "Ignoring unknown router notification message of type %" PRIu32, messageType);
            return false;
        }
    };

} // namespace nv::trace
