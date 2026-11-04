/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/DataFormats/nvtrace_cuda_api_trace.h>
#include <NvTrace/Target/NvTraceCudaApiTraceRecordUtilities.h>

namespace nv::trace {

    struct CudaApiTraceGraphNodeRecord
    {
        NvTrace_CUDA_ApiTrace_GraphNodeCreated common = {};
        CudaApiTraceGraphNodeParams params = {};
        uint32_t paramsMinorVersion = 0;
    };

    union CudaApiTraceReferencedCreatedRecord
    {
        NvTrace_CUDA_ApiTrace_ContextCreated contextCreated;
        NvTrace_CUDA_ApiTrace_StreamCreated streamCreated;
        NvTrace_CUDA_ApiTrace_ModuleLoaded moduleLoaded;
        NvTrace_CUDA_ApiTrace_GraphExecCreated graphExecCreated;
        NvTrace_CUDA_ApiTrace_ChannelCreated channelCreated;
    };

} // namespace nv::trace
