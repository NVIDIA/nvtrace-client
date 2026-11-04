/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cinttypes>
#include <cstdint>
#include <ctime>
#include <stdarg.h>
#include <string>

namespace nv::trace {

    inline std::string SprintfToString(const char* pFormat, ...)
    {
        va_list vl;

        va_start(vl, pFormat);
        const int length = 1 + vsnprintf(NULL, 0, pFormat, vl);
        va_end(vl);

        std::string str;
        str.resize(length);
        va_start(vl, pFormat);
        vsnprintf(&str[0], length, pFormat, vl);
        va_end(vl);
        str.erase(length - 1);

        return str;
    }

    inline std::string FormatBytesToHumanReadable(uint64_t bytes)
    {
        constexpr double KB = uint64_t(1) << 10;
        constexpr double MB = uint64_t(1) << 20;
        constexpr double GB = uint64_t(1) << 30;
        constexpr double TB = uint64_t(1) << 40;
        double fBytes = static_cast<double>(bytes);
        const char* suffix = "B";
        if (fBytes >= TB)
        {
            suffix = "TiB";
            fBytes /= TB;
        }
        else if (fBytes >= GB)
        {
            suffix = "GiB";
            fBytes /= GB;
        }
        else if (fBytes >= MB)
        {
            suffix = "MiB";
            fBytes /= MB;
        }
        else if (fBytes >= KB)
        {
            suffix = "KiB";
            fBytes /= KB;
        }
        return SprintfToString("%.2f %s", fBytes, suffix);
    }

    inline std::string FormatUtcNsecToHumanReadable(uint64_t utcNsec)
    {
        constexpr uint64_t NsecPerSecond = 1'000'000'000;
        const std::time_t utcSeconds = static_cast<std::time_t>(utcNsec / NsecPerSecond);
        std::tm utcTime = {};
#if defined(__linux__)
        if (!gmtime_r(&utcSeconds, &utcTime))
#endif
        {
            return {};
        }

        char dateTime[32] = {};
        if (std::strftime(dateTime, sizeof(dateTime), "%Y-%m-%d %H:%M:%S", &utcTime) == 0)
        {
            return {};
        }
        return SprintfToString("%s.%09" PRIu64, dateTime, utcNsec % NsecPerSecond);
    }

} // namespace nv::trace
