/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/DataFormats/nvtrace_common_types.h>

struct NvTrace_RouterObserverClient_ProviderTypeFilter
{
    const NVTRACE_UUID* pProviderTypeUuids;
    uint32_t providerTypeUuidCount;
};

struct NvTrace_RouterObserverClient_GpuPartitionFilter
{
    const NVTRACE_UUID* pGpuPartitionUuids;
    uint32_t gpuPartitionUuidCount;
};

struct NvTrace_RouterObserverClient_ProcessFilter
{
    const NVTRACE_PID* pPids;
    uint32_t pidCount;
};

struct NvTrace_RouterObserverClient_RouterConfig
{
    uint32_t maxGpuPartitionCount;
    uint32_t maxProcessCount;
    uint32_t maxProviderCount;
};

enum NvTrace_RouterObserverClient_NotificationType
{
    NvTrace_RouterObserverClient_NotificationType_Invalid = 0,
    NvTrace_RouterObserverClient_NotificationType_InitialStateSent,
    NvTrace_RouterObserverClient_NotificationType_SystemCreated,
    NvTrace_RouterObserverClient_NotificationType_GpuPartitionCreated,
    NvTrace_RouterObserverClient_NotificationType_GpuPartitionDestroyed,
    NvTrace_RouterObserverClient_NotificationType_ProcessCreated,
    NvTrace_RouterObserverClient_NotificationType_ProcessDestroyed,
    NvTrace_RouterObserverClient_NotificationType_ProviderCreated,
    NvTrace_RouterObserverClient_NotificationType_ProviderDestroyed
};

struct NvTrace_RouterObserverClient_InitialStateSentNotification
{
    ///
    size_t structSize;
};

struct NvTrace_RouterObserverClient_SystemCreatedNotification
{
    ///
    size_t structSize;
    /// Info shared memory identifier.
    NvTrace_OsObjectIdentifier infoMemoryIdentifier;
    /// Size of the info buffer.
    size_t infoSize;
};

struct NvTrace_RouterObserverClient_GpuPartitionCreatedNotification
{
    ///
    size_t structSize;
    /// Uniquely identifies the instance of the GPU partition across the system.
    NVTRACE_LUID gpuPartitionLuid;
    /// The driver's GPU or MIG UUID.
    NVTRACE_UUID gpuUuid;
    /// Timestamp of GPU partition creation event
    uint64_t creationTimestampTsc;
    /// Info shared memory identifier.
    NvTrace_OsObjectIdentifier infoMemoryIdentifier;
    /// Size of the info buffer.
    size_t infoSize;
};

struct NvTrace_RouterObserverClient_GpuPartitionDestroyedNotification
{
    ///
    size_t structSize;
    /// Uniquely identifies the instance of the GPU partition across the system.
    NVTRACE_LUID gpuPartitionLuid;
    /// Timestamp of GPU partition destruction event
    uint64_t destructionTimestampTsc;
};

struct NvTrace_RouterObserverClient_ProcessCreatedNotification
{
    ///
    size_t structSize;
    /// Uniquely identifies the instance of the process across the system.
    NVTRACE_LUID processLuid;
    /// Namespace PID (virtualized inside a container).
    NVTRACE_PID namespacePid;
    /// Host PID resolved by the Router. 0 => fall back to namespacePid.
    NVTRACE_PID globalPid;
    /// PID namespace identity: the (device, inode) of /proc/<pid>/ns/pid. Both 0 if unavailable.
    /// Display only, not for correlation.
    NvTrace_LinuxNamespaceId pidNamespaceId;
    /// Timestamp of process creation event.
    uint64_t creationTimestampTsc;
};

struct NvTrace_RouterObserverClient_ProcessDestroyedNotification
{
    ///
    size_t structSize;
    /// Uniquely identifies the instance of the process across the system.
    NVTRACE_LUID processLuid;
    /// Timestamp of process destruction event.
    uint64_t destructionTimestampTsc;
};

struct NvTrace_RouterObserverClient_ProviderCreatedNotification
{
    ///
    size_t structSize;
    /// Uniquely identifies the instance of the provider across the system.
    NVTRACE_LUID providerLuid;
    /// Identifies the parent object that the provider is a part of (e.g. a GPU partition or a process).
    /// If NVTRACE_NIL_LUID, the provider is not associated with any parent object (e.g. a system provider).
    NVTRACE_LUID parentLuid;
    /// Identifies the type of the provider.
    NVTRACE_UUID providerTypeUuid;
    /// Short name for the provider.
    const char* pProviderName;
    /// Provider buffer shared memory identifier.
    NvTrace_OsObjectIdentifier providerBufferMemoryIdentifier;
    /// Size of the provider buffer.
    size_t providerBufferSize;
};

struct NvTrace_RouterObserverClient_ProviderDestroyedNotification
{
    ///
    size_t structSize;
    /// Uniquely identifies the instance of the provider across the system.
    NVTRACE_LUID providerLuid;
    /// Identifies the parent object that the provider is a part of (e.g. a GPU partition or a process).
    /// If NVTRACE_NIL_LUID, the provider is not associated with any parent object (e.g. a system provider).
    NVTRACE_LUID parentLuid;
    /// Identifies the type of the provider.
    NVTRACE_UUID providerTypeUuid;
};

typedef bool (*NvTrace_PFN_RouterObserverClient_NotificationCallback)(void* pUserData, enum NvTrace_RouterObserverClient_NotificationType notificationType, const void* pNotification);
