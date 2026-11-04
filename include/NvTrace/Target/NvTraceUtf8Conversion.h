/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

namespace nv::trace {

    // Returns the number of UTF-8 bytes needed to encode a single Unicode codepoint (1-4).
    // Invalid codepoints (surrogates, > U+10FFFF) are treated as U+FFFD (3 bytes).
    inline size_t CodepointSizeAsUtf8(uint32_t codepoint)
    {
        if (codepoint > 0x10FFFF || (0xD800 <= codepoint && codepoint <= 0xDFFF))
        {
            codepoint = 0xFFFD;
        }
        if (codepoint <= 0x7F)
        {
            return 1;
        }
        if (codepoint <= 0x7FF)
        {
            return 2;
        }
        if (codepoint <= 0xFFFF)
        {
            return 3;
        }
        return 4;
    }

    // Writes a single Unicode codepoint as UTF-8 at pDst. Returns bytes written (1-4).
    // Invalid codepoints (surrogates, > U+10FFFF) are replaced with U+FFFD.
    inline size_t WriteUtf8Codepoint(char* pDst, uint32_t codepoint)
    {
        if (codepoint > 0x10FFFF || (0xD800 <= codepoint && codepoint <= 0xDFFF))
        {
            codepoint = 0xFFFD;
        }

        if (codepoint <= 0x7F)
        {
            pDst[0] = static_cast<char>(codepoint);
            return 1;
        }
        if (codepoint <= 0x7FF)
        {
            pDst[0] = static_cast<char>(0xC0 | (codepoint >> 6));
            pDst[1] = static_cast<char>(0x80 | (codepoint & 0x3F));
            return 2;
        }
        if (codepoint <= 0xFFFF)
        {
            pDst[0] = static_cast<char>(0xE0 | (codepoint >> 12));
            pDst[1] = static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
            pDst[2] = static_cast<char>(0x80 | (codepoint & 0x3F));
            return 3;
        }
        pDst[0] = static_cast<char>(0xF0 | (codepoint >> 18));
        pDst[1] = static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F));
        pDst[2] = static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
        pDst[3] = static_cast<char>(0x80 | (codepoint & 0x3F));
        return 4;
    }

    // Returns the size in bytes (including null terminator) of a null-terminated string as UTF-8.
    // Returns 0 if pInput is null.
    inline size_t StringSizeAsUtf8(const char* pInput)
    {
        if (!pInput)
        {
            return 0;
        }
        return strlen(pInput) + 1;
    }

    inline size_t StringSizeAsUtf8(const char16_t* pInput)
    {
        if (!pInput)
        {
            return 0;
        }

        size_t size = 0;
        for (size_t i = 0; pInput[i] != 0; ++i)
        {
            uint32_t codepoint = pInput[i];

            if (0xD800 <= codepoint && codepoint <= 0xDBFF)
            {
                if (0xDC00 <= pInput[i + 1] && pInput[i + 1] <= 0xDFFF)
                {
                    uint32_t highSurrogate = codepoint;
                    uint32_t lowSurrogate = pInput[++i];
                    codepoint = 0x10000 + ((highSurrogate - 0xD800) << 10) + (lowSurrogate - 0xDC00);
                }
                else
                {
                    codepoint = 0xFFFD;
                }
            }
            else if (0xDC00 <= codepoint && codepoint <= 0xDFFF)
            {
                codepoint = 0xFFFD;
            }

            size += CodepointSizeAsUtf8(codepoint);
        }

        return size + 1;
    }

    inline size_t StringSizeAsUtf8(const char32_t* pInput)
    {
        if (!pInput)
        {
            return 0;
        }

        size_t size = 0;
        for (size_t i = 0; pInput[i] != 0; ++i)
        {
            size += CodepointSizeAsUtf8(static_cast<uint32_t>(pInput[i]));
        }

        return size + 1;
    }

    // Converts a null-terminated UTF-16 string to UTF-8.
    // Writes to pOutput (must be at least StringSizeAsUtf8(pInput) bytes).
    // Returns total bytes written including null terminator. Returns 0 if pInput is null.
    inline size_t ConvertUtf16ToUtf8(const char16_t* pInput, char* pOutput)
    {
        if (!pInput)
        {
            return 0;
        }

        char* pDst = pOutput;
        for (size_t i = 0; pInput[i] != 0; ++i)
        {
            uint32_t codepoint = pInput[i];

            if (0xD800 <= codepoint && codepoint <= 0xDBFF)
            {
                if (0xDC00 <= pInput[i + 1] && pInput[i + 1] <= 0xDFFF)
                {
                    uint32_t highSurrogate = codepoint;
                    uint32_t lowSurrogate = pInput[++i];
                    codepoint = 0x10000 + ((highSurrogate - 0xD800) << 10) + (lowSurrogate - 0xDC00);
                }
                else
                {
                    codepoint = 0xFFFD;
                }
            }
            else if (0xDC00 <= codepoint && codepoint <= 0xDFFF)
            {
                codepoint = 0xFFFD;
            }

            pDst += WriteUtf8Codepoint(pDst, codepoint);
        }

        *pDst++ = '\0';
        return static_cast<size_t>(pDst - pOutput);
    }

    // Converts a null-terminated UTF-32 string to UTF-8.
    // Writes to pOutput (must be at least StringSizeAsUtf8(pInput) bytes).
    // Returns total bytes written including null terminator. Returns 0 if pInput is null.
    inline size_t ConvertUtf32ToUtf8(const char32_t* pInput, char* pOutput)
    {
        if (!pInput)
        {
            return 0;
        }

        char* pDst = pOutput;
        for (size_t i = 0; pInput[i] != 0; ++i)
        {
            pDst += WriteUtf8Codepoint(pDst, static_cast<uint32_t>(pInput[i]));
        }

        *pDst++ = '\0';
        return static_cast<size_t>(pDst - pOutput);
    }

} // namespace nv::trace

namespace nv::trace::Codec {

    inline void AppendUtf8Codepoint(std::string& output, uint32_t codepoint)
    {
        char utf8Bytes[4];
        size_t utf8Size = WriteUtf8Codepoint(utf8Bytes, codepoint);
        output.append(utf8Bytes, utf8Size);
    }

    inline void ConvertUtf16ToUtf8(const char16_t* pInput, std::string& output)
    {
        output.clear();
        size_t stringSize = nv::trace::StringSizeAsUtf8(pInput);
        if (stringSize <= 1)
        {
            return;
        }
        output.resize(stringSize - 1);
        nv::trace::ConvertUtf16ToUtf8(pInput, output.data());
    }

    inline void ConvertUtf32ToUtf8(const char32_t* pInput, std::string& output)
    {
        output.clear();
        size_t stringSize = nv::trace::StringSizeAsUtf8(pInput);
        if (stringSize <= 1)
        {
            return;
        }
        output.resize(stringSize - 1);
        nv::trace::ConvertUtf32ToUtf8(pInput, output.data());
    }

} // namespace nv::trace::Codec
