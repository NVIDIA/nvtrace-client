/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/DataFormats/nvtrace_common_types.h>

#include <cstring>
#include <string>
#include <string_view>

inline bool operator==(const NVTRACE_UUID& lhs, const NVTRACE_UUID& rhs)
{
    return std::memcmp(&lhs, &rhs, sizeof(NVTRACE_UUID)) == 0;
}
inline bool operator!=(const NVTRACE_UUID& lhs, const NVTRACE_UUID& rhs)
{
    return std::memcmp(&lhs, &rhs, sizeof(NVTRACE_UUID)) != 0;
}
inline bool operator<(const NVTRACE_UUID& lhs, const NVTRACE_UUID& rhs)
{
    return std::memcmp(&lhs, &rhs, sizeof(NVTRACE_UUID)) < 0;
}

namespace nv::trace {

    inline bool ConvertStringToUuid(std::string_view str, NVTRACE_UUID& outUuid)
    {
        auto fnConvert2HexCharsToByte = [](const char* pHexChars, uint8_t& outByte) {
            uint8_t tmp;

            // Convert most significant nibble
            uint8_t hexChar = static_cast<uint8_t>(pHexChars[0]);
            if ((tmp = hexChar - '0') < 10)
            {
                outByte = tmp;
            }
            else if ((tmp = hexChar - 'a') < 6 || (tmp = hexChar - 'A') < 6)
            {
                outByte = tmp + 10;
            }
            else
            {
                return false;
            }

            outByte <<= 4;

            // Convert least significant nibble
            hexChar = static_cast<uint8_t>(pHexChars[1]);
            if ((tmp = hexChar - '0') < 10)
            {
                outByte |= tmp;
            }
            else if ((tmp = hexChar - 'a') < 6 || (tmp = hexChar - 'A') < 6)
            {
                outByte |= tmp + 10;
            }
            else
            {
                return false;
            }

            return true;
        };

        // clang-format off
        return str.size() == 36 && // 32 characters for UUID plus four dashes
            fnConvert2HexCharsToByte(&str[0], outUuid.bytes[0]) &&
            fnConvert2HexCharsToByte(&str[2], outUuid.bytes[1]) &&
            fnConvert2HexCharsToByte(&str[4], outUuid.bytes[2]) &&
            fnConvert2HexCharsToByte(&str[6], outUuid.bytes[3]) &&
            str[8] == '-' &&
            fnConvert2HexCharsToByte(&str[9], outUuid.bytes[4]) &&
            fnConvert2HexCharsToByte(&str[11], outUuid.bytes[5]) &&
            str[13] == '-' &&
            fnConvert2HexCharsToByte(&str[14], outUuid.bytes[6]) &&
            fnConvert2HexCharsToByte(&str[16], outUuid.bytes[7]) &&
            str[18] == '-' &&
            fnConvert2HexCharsToByte(&str[19], outUuid.bytes[8]) &&
            fnConvert2HexCharsToByte(&str[21], outUuid.bytes[9]) &&
            str[23] == '-' &&
            fnConvert2HexCharsToByte(&str[24], outUuid.bytes[10]) &&
            fnConvert2HexCharsToByte(&str[26], outUuid.bytes[11]) &&
            fnConvert2HexCharsToByte(&str[28], outUuid.bytes[12]) &&
            fnConvert2HexCharsToByte(&str[30], outUuid.bytes[13]) &&
            fnConvert2HexCharsToByte(&str[32], outUuid.bytes[14]) &&
            fnConvert2HexCharsToByte(&str[34], outUuid.bytes[15]);
        // clang-format on
    }

    inline void ConvertUuidToString(const NVTRACE_UUID& uuid, char outStr[36])
    {
        auto fnConvertByteTo2HexChars = [](uint8_t byte, char outHexChars[2]) {
            // Convert most significant nibble
            outHexChars[0] = "0123456789ABCDEF"[byte >> 4];

            // Convert least significant nibble
            outHexChars[1] = "0123456789ABCDEF"[byte & 0xF];
        };

        fnConvertByteTo2HexChars(uuid.bytes[0], &outStr[0]);
        fnConvertByteTo2HexChars(uuid.bytes[1], &outStr[2]);
        fnConvertByteTo2HexChars(uuid.bytes[2], &outStr[4]);
        fnConvertByteTo2HexChars(uuid.bytes[3], &outStr[6]);
        outStr[8] = '-';
        fnConvertByteTo2HexChars(uuid.bytes[4], &outStr[9]);
        fnConvertByteTo2HexChars(uuid.bytes[5], &outStr[11]);
        outStr[13] = '-';
        fnConvertByteTo2HexChars(uuid.bytes[6], &outStr[14]);
        fnConvertByteTo2HexChars(uuid.bytes[7], &outStr[16]);
        outStr[18] = '-';
        fnConvertByteTo2HexChars(uuid.bytes[8], &outStr[19]);
        fnConvertByteTo2HexChars(uuid.bytes[9], &outStr[21]);
        outStr[23] = '-';
        fnConvertByteTo2HexChars(uuid.bytes[10], &outStr[24]);
        fnConvertByteTo2HexChars(uuid.bytes[11], &outStr[26]);
        fnConvertByteTo2HexChars(uuid.bytes[12], &outStr[28]);
        fnConvertByteTo2HexChars(uuid.bytes[13], &outStr[30]);
        fnConvertByteTo2HexChars(uuid.bytes[14], &outStr[32]);
        fnConvertByteTo2HexChars(uuid.bytes[15], &outStr[34]);
    }

    inline std::string ToString(const NVTRACE_UUID& uuid)
    {
        std::string str(36, '\0');
        ConvertUuidToString(uuid, str.data());
        return str;
    }

} //namespace nv::trace
