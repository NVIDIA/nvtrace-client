/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <charconv>
#include <cstdlib>
#include <cstring>
#include <optional>

#include <NvTrace/Target/NvTraceClientLogger.h>

namespace nv::trace {

    inline std::optional<size_t> GetEnvironmentVariableAsSizeT(const char* variableName)
    {
        const char* pValueCstr = std::getenv(variableName);

        if (!pValueCstr)
        {
            return std::nullopt;
        }
        size_t value;
        std::from_chars_result result = std::from_chars(pValueCstr, pValueCstr + strlen(pValueCstr), value);
        if (result.ec == std::errc()) // No error
        {
            // Check if the entire string was consumed (no trailing non-numeric characters)
            if (result.ptr == pValueCstr + strlen(pValueCstr))
            {
                return value;
            }
            else
            {
                // The string contained valid digits followed by non-digits (e.g., "123abc")
                NVTRACE_LOG_WRN(20, "Env variable %s has trailing non-numeric characters: '%s'", variableName, pValueCstr);
                return std::nullopt;
            }
        }
        else if (result.ec == std::errc::invalid_argument)
        {
            // No conversion could be performed (e.g., "hello")
            NVTRACE_LOG_WRN(20, "Env variable %s is not a valid number (invalid argument): '%s'", variableName, pValueCstr);
            return std::nullopt;
        }
        else if (result.ec == std::errc::result_out_of_range)
        {
            // The value is out of range for size_t
            NVTRACE_LOG_WRN(20, "Env variable %s value is out of range for size_t: '%s'", variableName, pValueCstr);
            return std::nullopt;
        }
        else
        {
            // Other unexpected errors
            NVTRACE_LOG_WRN(20, "Env variable %s conversion failed with unexpected error: '%s'", variableName, pValueCstr);
            return std::nullopt;
        }
    }

    inline std::optional<bool> GetEnvironmentVariableAsBool(const char* variableName)
    {
        const char* pValueCstr = std::getenv(variableName);
        if (!pValueCstr)
        {
            return std::nullopt;
        }
        if (strcmp(pValueCstr, "0") == 0)
        {
            return false;
        }
        else
        {
            return true;
        }
    }

} // namespace nv::trace
