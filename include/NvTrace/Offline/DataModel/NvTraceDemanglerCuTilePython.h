/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cinttypes>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <utility>

namespace nv::trace {

    /// Dependency-free demangler for the cuTile Python kernel-signature ABI.
    /// Reference encoding:
    /// https://github.com/NVIDIA/cutile-python/blob/main/src/cuda/tile/compilation/_name_mangling.py
    ///
    /// Lifecycle:
    ///
    ///   BindScratchSpace --> BeginDemangling --> ParseInput --> Complete
    ///          ^                                  |
    ///          +---------- NeedMoreScratch -------+
    ///
    ///   Complete --> RenderDemangledName --> OutputLimitReached --> resume render
    ///          |
    ///          +----> FinishCountingRemainingOutputBytes (destructive)
    ///
    /// BeginDemangling records input without parsing. ParseInput restarts from the
    /// beginning; rendering resumes at the exact token and UTF-8 byte offset.
    class CuTilePythonDemangler
    {
    public:
        enum class Status : uint8_t
        {
            Complete,
            NeedMoreScratch,
            OutputLimitReached,
            InvalidInput,
            NotParsed,
        };

        CuTilePythonDemangler() = default;
        CuTilePythonDemangler(const CuTilePythonDemangler&) = delete;
        CuTilePythonDemangler& operator=(const CuTilePythonDemangler&) = delete;
        CuTilePythonDemangler(CuTilePythonDemangler&&) = delete;
        CuTilePythonDemangler& operator=(CuTilePythonDemangler&&) = delete;

        void BindScratchSpace(uint8_t* pScratchSpace, size_t scratchSpaceSize)
        {
            m_pScratchSpace = pScratchSpace;
            m_scratchSpaceSize = scratchSpaceSize;
            ResetPlan();
            ResetRender();
            m_parseStatus = Status::NotParsed;
        }

        void BeginDemangling(const char* pMangledName, size_t mangledNameLength)
        {
            m_pInput = pMangledName;
            m_inputLength = mangledNameLength;
            m_inputIndex = 0;
            m_requiredScratchSpaceSize = CalculateRequiredScratchSpaceSize(pMangledName, mangledNameLength);
            m_parseStatus = Status::NotParsed;
            ResetPlan();
            ResetRender();
        }

        Status ParseInput()
        {
            ResetPlan();
            ResetRender();
            m_inputIndex = 0;
            m_lastAliasGroupId = -1;

            if (!m_pInput || m_inputLength == 0 || m_requiredScratchSpaceSize == SIZE_MAX)
            {
                m_parseStatus = Status::InvalidInput;
                return m_parseStatus;
            }
            if (!m_pScratchSpace || m_scratchSpaceSize < m_requiredScratchSpaceSize)
            {
                m_parseStatus = Status::NeedMoreScratch;
                return m_parseStatus;
            }

            size_t signatureOffset = SIZE_MAX;
            for (size_t inputIndex = 0; inputIndex + 1 < m_inputLength; ++inputIndex)
            {
                if (m_pInput[inputIndex] == '_' && m_pInput[inputIndex + 1] == 'K')
                {
                    signatureOffset = inputIndex;
                }
            }
            if (signatureOffset == SIZE_MAX || signatureOffset == 0
                || !IsValidUtf8(m_pInput, signatureOffset))
            {
                m_parseStatus = Status::InvalidInput;
                return m_parseStatus;
            }

            m_inputIndex = signatureOffset + 2;
            if (!ConsumeInput('t'))
            {
                m_parseStatus = Status::InvalidInput;
                return m_parseStatus;
            }
            const size_t versionOffset = m_inputIndex;
            unsigned long long callingConventionVersion = 0;
            if (!ParseUnsignedDecimal(callingConventionVersion))
            {
                m_parseStatus = Status::InvalidInput;
                return m_parseStatus;
            }
            const size_t versionLength = m_inputIndex - versionOffset;

            bool planValid = AddInputToken(0, signatureOffset) && AddStaticToken("(");
            size_t parameterCount = 0;
            while (planValid && HasInput())
            {
                planValid = ConsumeInput('_');
                if (planValid && parameterCount++ != 0)
                {
                    planValid = AddStaticToken(", ");
                }
                ConstraintKind constraintKind = ConstraintKind::Invalid;
                if (planValid)
                {
                    constraintKind = ParseConstraint(callingConventionVersion, planValid);
                }
                planValid = planValid && constraintKind != ConstraintKind::Invalid;
            }
            planValid = planValid
                     && AddStaticToken(") [calling_convention=CallingConvention('cutile_python_v")
                     && AddInputToken(versionOffset, versionLength)
                     && AddStaticToken("', 't")
                     && AddInputToken(versionOffset, versionLength)
                     && AddStaticToken("')]");

            if (!planValid || m_allocationFailed || m_inputIndex != m_inputLength)
            {
                m_parseStatus = m_allocationFailed ? Status::NeedMoreScratch : Status::InvalidInput;
                return m_parseStatus;
            }
            m_parseStatus = Status::Complete;
            return m_parseStatus;
        }

        size_t GetRequiredScratchSpaceSize() const
        {
            return m_requiredScratchSpaceSize;
        }

        size_t GetRenderedOutputLength() const
        {
            return m_renderedOutputLength;
        }

        size_t GetInputBytesConsumed() const
        {
            return m_inputIndex;
        }

        template <typename TAppendOutput>
        Status RenderDemangledName(size_t maxOutputLength, TAppendOutput&& appendOutput)
        {
            if (m_parseStatus != Status::Complete)
            {
                return m_parseStatus;
            }
            if (!m_renderStarted)
            {
                m_pRenderToken = m_pFirstToken;
                m_renderStarted = true;
            }

            while (m_pRenderToken)
            {
                const char* pFragment = nullptr;
                size_t fragmentLength = 0;
                GetTokenFragment(*m_pRenderToken, pFragment, fragmentLength);
                const size_t fragmentBytesRemaining = fragmentLength - m_fragmentOffset;
                const size_t outputBytesAvailable = maxOutputLength > m_renderedOutputLength
                                                      ? maxOutputLength - m_renderedOutputLength
                                                      : 0;
                size_t acceptedByteCount = fragmentBytesRemaining < outputBytesAvailable
                                             ? fragmentBytesRemaining
                                             : outputBytesAvailable;
                acceptedByteCount = ClampToUtf8Boundary(pFragment + m_fragmentOffset, fragmentBytesRemaining, acceptedByteCount);
                if (acceptedByteCount != 0)
                {
                    appendOutput(pFragment + m_fragmentOffset, acceptedByteCount);
                    m_fragmentOffset += acceptedByteCount;
                    m_renderedOutputLength += acceptedByteCount;
                }
                if (m_fragmentOffset != fragmentLength)
                {
                    return Status::OutputLimitReached;
                }
                m_pRenderToken = m_pRenderToken->pNext;
                m_fragmentOffset = 0;
            }
            return Status::Complete;
        }

        Status FinishCountingRemainingOutputBytes(size_t& remainingOutputByteCount)
        {
            remainingOutputByteCount = 0;
            if (m_parseStatus != Status::Complete)
            {
                return m_parseStatus;
            }
            if (!m_renderStarted)
            {
                m_pRenderToken = m_pFirstToken;
                m_renderStarted = true;
            }
            while (m_pRenderToken)
            {
                const char* pFragment = nullptr;
                size_t fragmentLength = 0;
                GetTokenFragment(*m_pRenderToken, pFragment, fragmentLength);
                const size_t fragmentBytesRemaining = fragmentLength - m_fragmentOffset;
                if (remainingOutputByteCount > SIZE_MAX - fragmentBytesRemaining
                    || m_renderedOutputLength > SIZE_MAX - fragmentBytesRemaining)
                {
                    return Status::InvalidInput;
                }
                remainingOutputByteCount += fragmentBytesRemaining;
                m_renderedOutputLength += fragmentBytesRemaining;
                m_pRenderToken = m_pRenderToken->pNext;
                m_fragmentOffset = 0;
            }
            return Status::Complete;
        }

    private:
        // Plan schema
        // -----------
        // cuTile signatures render in encoded order, so the immutable plan is a
        // forward list of output tokens rather than an expression tree. A token is
        // static punctuation, an input-backed string slice, or a compact numeric
        // payload. AxisState and AliasGroup are temporary parse records allocated
        // from the same caller-owned bump arena.
        //
        //   kernel_Kt1_Si32
        //     StaticText "kernel("
        //     StaticText "ScalarConstraint(dtype=<DType '"
        //     DType      input slice "i32"
        //     StaticText "'>)) ..."
        //
        // A longer array constraint adds tokens for shape, divisibility, stride,
        // lower-bound, dtype, and alias tuples in precisely their output order.
        // Rendering therefore advances one PlanToken pointer at a time and never
        // needs to stitch non-adjacent temporary strings.

        enum class ConstraintKind : uint8_t
        {
            Invalid,
            Array,
            List,
            Tuple,
            Scalar,
            Constant,
        };

        enum class TokenKind : uint8_t
        {
            StaticText,
            InputText,
            SignedInteger,
            DType,
            FloatBits,
            AliasGroup,
        };

        struct PlanToken
        {
            PlanToken* pNext;
            const char* pStaticText;
            uint64_t unsignedValue;
            uint32_t inputOffset;
            uint32_t inputLength;
            TokenKind kind;
        };

        struct AxisState
        {
            long long shapeConstant;
            long long shapeDivisibleBy;
            long long strideConstant;
            long long strideDivisibleBy;
            long long strideLowerBound;
            uint8_t hasShapeConstant;
            uint8_t hasStrideConstant;
            uint8_t hasStrideLowerBound;
        };

        struct AliasGroup
        {
            AliasGroup* pNext;
            unsigned long long groupId;
        };

    private:
        // Scratch-space and plan-construction helpers.

        static bool TryMultiply(size_t leftValue, size_t rightValue, size_t& result)
        {
            if (leftValue != 0 && rightValue > SIZE_MAX / leftValue)
            {
                result = SIZE_MAX;
                return false;
            }
            result = leftValue * rightValue;
            return true;
        }

        static bool TryAdd(size_t leftValue, size_t rightValue, size_t& result)
        {
            if (rightValue > SIZE_MAX - leftValue)
            {
                result = SIZE_MAX;
                return false;
            }
            result = leftValue + rightValue;
            return true;
        }

        static size_t CalculateRequiredScratchSpaceSize(const char* pInput, size_t inputLength)
        {
            if (!pInput || inputLength == 0)
            {
                return 0;
            }
            size_t totalDimensionCount = 0;
            for (size_t inputIndex = 0; inputIndex + 1 < inputLength; ++inputIndex)
            {
                if (pInput[inputIndex] != 'A' || pInput[inputIndex + 1] < '0' || pInput[inputIndex + 1] > '9')
                {
                    continue;
                }
                size_t dimensionCount = 0;
                size_t digitIndex = inputIndex + 1;
                while (digitIndex < inputLength && '0' <= pInput[digitIndex] && pInput[digitIndex] <= '9')
                {
                    const size_t digitValue = static_cast<size_t>(pInput[digitIndex] - '0');
                    if (dimensionCount > (SIZE_MAX - digitValue) / 10)
                    {
                        return SIZE_MAX;
                    }
                    dimensionCount = dimensionCount * 10 + digitValue;
                    ++digitIndex;
                }
                if (!TryAdd(totalDimensionCount, dimensionCount, totalDimensionCount))
                {
                    return SIZE_MAX;
                }
            }

            size_t inputBytes = 0;
            size_t dimensionBytes = 0;
            size_t requiredBytes = 0;
            return TryMultiply(inputLength, 512, inputBytes)
                        && TryMultiply(totalDimensionCount, 512, dimensionBytes)
                        && TryAdd(inputBytes, dimensionBytes, requiredBytes)
                        && TryAdd(requiredBytes, 4096, requiredBytes)
                     ? requiredBytes
                     : SIZE_MAX;
        }

        void ResetPlan()
        {
            m_scratchBytesUsed = 0;
            m_pFirstToken = nullptr;
            m_pLastToken = nullptr;
            m_allocationFailed = false;
        }

        void ResetRender()
        {
            m_pRenderToken = nullptr;
            m_renderedOutputLength = 0;
            m_fragmentOffset = 0;
            m_renderStarted = false;
        }

        void* AllocateScratch(size_t byteCount, size_t alignment)
        {
            const uintptr_t scratchAddress = reinterpret_cast<uintptr_t>(m_pScratchSpace);
            const uintptr_t currentAddress = scratchAddress + m_scratchBytesUsed;
            const uintptr_t alignmentMask = alignment - 1;
            const uintptr_t alignedAddress = (currentAddress + alignmentMask) & ~alignmentMask;
            const size_t alignmentBytes = static_cast<size_t>(alignedAddress - currentAddress);
            if (alignmentBytes > m_scratchSpaceSize - m_scratchBytesUsed
                || byteCount > m_scratchSpaceSize - m_scratchBytesUsed - alignmentBytes)
            {
                m_allocationFailed = true;
                return nullptr;
            }
            m_scratchBytesUsed += alignmentBytes + byteCount;
            return reinterpret_cast<void*>(alignedAddress);
        }

        template <typename TObject>
        TObject* AllocateObject()
        {
            return reinterpret_cast<TObject*>(AllocateScratch(sizeof(TObject), alignof(TObject)));
        }

        template <typename TObject>
        TObject* AllocateArray(size_t elementCount)
        {
            size_t byteCount = 0;
            if (!TryMultiply(elementCount, sizeof(TObject), byteCount))
            {
                m_allocationFailed = true;
                return nullptr;
            }
            return reinterpret_cast<TObject*>(AllocateScratch(byteCount, alignof(TObject)));
        }

        PlanToken* AddToken(TokenKind kind)
        {
            PlanToken* pToken = AllocateObject<PlanToken>();
            if (!pToken)
            {
                return nullptr;
            }
            pToken->pNext = nullptr;
            pToken->pStaticText = nullptr;
            pToken->unsignedValue = 0;
            pToken->inputOffset = 0;
            pToken->inputLength = 0;
            pToken->kind = kind;
            if (m_pLastToken)
            {
                m_pLastToken->pNext = pToken;
            }
            else
            {
                m_pFirstToken = pToken;
            }
            m_pLastToken = pToken;
            return pToken;
        }

        bool AddStaticToken(const char* pText)
        {
            PlanToken* pToken = AddToken(TokenKind::StaticText);
            if (pToken)
            {
                pToken->pStaticText = pText;
            }
            return pToken != nullptr;
        }

        bool AddInputToken(size_t inputOffset, size_t inputLength)
        {
            if (inputOffset > UINT32_MAX || inputLength > UINT32_MAX)
            {
                return false;
            }
            PlanToken* pToken = AddToken(TokenKind::InputText);
            if (pToken)
            {
                pToken->inputOffset = static_cast<uint32_t>(inputOffset);
                pToken->inputLength = static_cast<uint32_t>(inputLength);
            }
            return pToken != nullptr;
        }

        bool AddSignedIntegerToken(long long value)
        {
            PlanToken* pToken = AddToken(TokenKind::SignedInteger);
            if (pToken)
            {
                std::memcpy(&pToken->unsignedValue, &value, sizeof(value));
            }
            return pToken != nullptr;
        }

        bool AddDTypeToken(size_t inputOffset, size_t inputLength)
        {
            if (inputOffset > UINT32_MAX || inputLength > UINT32_MAX)
            {
                return false;
            }
            PlanToken* pToken = AddToken(TokenKind::DType);
            if (pToken)
            {
                pToken->inputOffset = static_cast<uint32_t>(inputOffset);
                pToken->inputLength = static_cast<uint32_t>(inputLength);
            }
            return pToken != nullptr;
        }

        bool AddFloatToken(uint64_t valueBits)
        {
            PlanToken* pToken = AddToken(TokenKind::FloatBits);
            if (pToken)
            {
                pToken->unsignedValue = valueBits;
            }
            return pToken != nullptr;
        }

        bool AddAliasGroupToken(unsigned long long groupId)
        {
            PlanToken* pToken = AddToken(TokenKind::AliasGroup);
            if (pToken)
            {
                pToken->unsignedValue = groupId;
            }
            return pToken != nullptr;
        }

        bool HasInput(size_t byteCount = 1) const
        {
            return byteCount <= m_inputLength - m_inputIndex;
        }

        char PeekInput(size_t lookahead = 0) const
        {
            return lookahead < m_inputLength - m_inputIndex ? m_pInput[m_inputIndex + lookahead] : '\0';
        }

        bool ConsumeInput(char expectedCharacter)
        {
            if (PeekInput() != expectedCharacter)
            {
                return false;
            }
            ++m_inputIndex;
            return true;
        }

        static bool IsDecimalDigit(char inputCharacter)
        {
            return '0' <= inputCharacter && inputCharacter <= '9';
        }

        static bool IsLowerHexDigit(char inputCharacter)
        {
            return IsDecimalDigit(inputCharacter) || ('a' <= inputCharacter && inputCharacter <= 'f');
        }

        static int HexValue(char inputCharacter)
        {
            return IsDecimalDigit(inputCharacter)                 ? inputCharacter - '0'
                 : 'a' <= inputCharacter && inputCharacter <= 'f' ? inputCharacter - 'a' + 10
                                                                  : -1;
        }

    private:
        // Parsing is a single forward pass over the signature fields. Helpers
        // validate and append render-ready PlanTokens as each constraint is read.
        // ParseInput resets the bump arena and can be retried after scratch growth.

        bool ParseUnsignedDecimal(unsigned long long& value)
        {
            if (!IsDecimalDigit(PeekInput()))
            {
                return false;
            }
            value = 0;
            while (IsDecimalDigit(PeekInput()))
            {
                const unsigned long long digitValue = static_cast<unsigned long long>(PeekInput() - '0');
                if (value > (std::numeric_limits<unsigned long long>::max() - digitValue) / 10)
                {
                    return false;
                }
                value = value * 10 + digitValue;
                ++m_inputIndex;
            }
            return true;
        }

        bool ParseUnsignedHex(unsigned long long& value, size_t& digitCount)
        {
            if (!IsLowerHexDigit(PeekInput()))
            {
                return false;
            }
            value = 0;
            digitCount = 0;
            while (IsLowerHexDigit(PeekInput()))
            {
                const unsigned long long digitValue = static_cast<unsigned long long>(HexValue(PeekInput()));
                if (value > (std::numeric_limits<unsigned long long>::max() - digitValue) / 16)
                {
                    return false;
                }
                value = value * 16 + digitValue;
                ++m_inputIndex;
                ++digitCount;
            }
            return true;
        }

        bool ParseSignedInteger(long long& value)
        {
            const bool isNegative = ConsumeInput('_');
            unsigned long long magnitude = 0;
            if (!ParseUnsignedDecimal(magnitude))
            {
                return false;
            }
            const unsigned long long maxPositive =
                static_cast<unsigned long long>(std::numeric_limits<long long>::max());
            if ((!isNegative && magnitude > maxPositive) || (isNegative && magnitude > maxPositive + 1))
            {
                return false;
            }
            value = magnitude == maxPositive + 1
                      ? std::numeric_limits<long long>::min()
                  : isNegative ? -static_cast<long long>(magnitude)
                               : static_cast<long long>(magnitude);
            return true;
        }

        bool ParseDivisibility(long long& value)
        {
            return ParseSignedInteger(value) && value > 1;
        }

        bool ParseDType(size_t& dtypeOffset, size_t& dtypeLength)
        {
            dtypeOffset = m_inputIndex;
            while (('a' <= PeekInput() && PeekInput() <= 'z') || IsDecimalDigit(PeekInput()))
            {
                ++m_inputIndex;
            }
            dtypeLength = m_inputIndex - dtypeOffset;
            return dtypeLength != 0 && DTypeName(m_pInput + dtypeOffset, dtypeLength) != nullptr;
        }

        static const char* DTypeName(const char* pCode, size_t codeLength)
        {
            struct DTypeEntry
            {
                const char* pCode;
                const char* pName;
            };
            static constexpr DTypeEntry dtypeEntries[] = {
                {     "b8",          "bool_"},
                {     "u8",          "uint8"},
                {    "u16",         "uint16"},
                {    "u32",         "uint32"},
                {    "u64",         "uint64"},
                {     "i8",           "int8"},
                {    "i16",          "int16"},
                {    "i32",          "int32"},
                {    "i64",          "int64"},
                {    "f16",        "float16"},
                {    "f32",        "float32"},
                {    "f64",        "float64"},
                {   "bf16",       "bfloat16"},
                {   "tf32",       "tfloat32"},
                { "f8m3fn",  "float8_e4m3fn"},
                {   "f8m2",    "float8_e5m2"},
                {"f8m0fnu", "float8_e8m0fnu"},
            };
            for (const DTypeEntry& dtypeEntry : dtypeEntries)
            {
                const size_t knownLength = std::strlen(dtypeEntry.pCode);
                if (knownLength == codeLength && std::memcmp(pCode, dtypeEntry.pCode, codeLength) == 0)
                {
                    return dtypeEntry.pName;
                }
            }
            return nullptr;
        }

        ConstraintKind ParseConstraint(unsigned long long version, bool& planValid)
        {
            const char constraintCode = PeekInput();
            if (constraintCode < 'A' || constraintCode > 'Z')
            {
                planValid = false;
                return ConstraintKind::Invalid;
            }
            ++m_inputIndex;
            switch (constraintCode)
            {
                case 'A': return ParseArrayConstraint(version, planValid);
                case 'L': return ParseListConstraint(version, planValid);
                case 'T': return ParseTupleConstraint(version, planValid);
                case 'S': return ParseScalarConstraint(planValid);
                case 'B': return ParseBooleanConstraint(planValid);
                case 'I': return ParseIntegerConstraint(planValid);
                case 'F': return ParseFloatConstraint(planValid);
                default:
                    planValid = false;
                    return ConstraintKind::Invalid;
            }
        }

        ConstraintKind ParseScalarConstraint(bool& planValid)
        {
            size_t dtypeOffset = 0;
            size_t dtypeLength = 0;
            planValid = ParseDType(dtypeOffset, dtypeLength)
                     && AddStaticToken("ScalarConstraint(dtype=")
                     && AddDTypeToken(dtypeOffset, dtypeLength)
                     && AddStaticToken(")");
            return planValid ? ConstraintKind::Scalar : ConstraintKind::Invalid;
        }

        ConstraintKind ParseBooleanConstraint(bool& planValid)
        {
            const char booleanValue = PeekInput();
            if (booleanValue != '0' && booleanValue != '1')
            {
                planValid = false;
                return ConstraintKind::Invalid;
            }
            ++m_inputIndex;
            planValid = AddStaticToken(booleanValue == '0' ? "ConstantConstraint(value=False)" : "ConstantConstraint(value=True)");
            return planValid ? ConstraintKind::Constant : ConstraintKind::Invalid;
        }

        ConstraintKind ParseIntegerConstraint(bool& planValid)
        {
            long long integerValue = 0;
            planValid = ParseSignedInteger(integerValue)
                     && AddStaticToken("ConstantConstraint(value=")
                     && AddSignedIntegerToken(integerValue)
                     && AddStaticToken(")");
            return planValid ? ConstraintKind::Constant : ConstraintKind::Invalid;
        }

        ConstraintKind ParseFloatConstraint(bool& planValid)
        {
            if (!HasInput(16))
            {
                planValid = false;
                return ConstraintKind::Invalid;
            }
            uint64_t valueBits = 0;
            for (size_t digitIndex = 0; digitIndex < 16; ++digitIndex)
            {
                const int digitValue = HexValue(PeekInput());
                if (digitValue < 0)
                {
                    planValid = false;
                    return ConstraintKind::Invalid;
                }
                valueBits = valueBits * 16 + static_cast<uint64_t>(digitValue);
                ++m_inputIndex;
            }
            planValid = AddStaticToken("ConstantConstraint(value=")
                     && AddFloatToken(valueBits)
                     && AddStaticToken(")");
            return planValid ? ConstraintKind::Constant : ConstraintKind::Invalid;
        }

        ConstraintKind ParseTupleConstraint(unsigned long long version, bool& planValid)
        {
            if (version < 2)
            {
                planValid = false;
                return ConstraintKind::Invalid;
            }
            unsigned long long elementCount = 0;
            planValid = ParseUnsignedDecimal(elementCount)
                     && AddStaticToken("TupleConstraint(elements=(");
            for (unsigned long long elementIndex = 0; planValid && elementIndex < elementCount; ++elementIndex)
            {
                if (elementIndex != 0)
                {
                    planValid = AddStaticToken(", ");
                }
                ConstraintKind elementKind = ConstraintKind::Invalid;
                if (planValid)
                {
                    elementKind = ParseConstraint(version, planValid);
                }
                planValid = planValid && elementKind != ConstraintKind::Invalid;
            }
            if (planValid && elementCount == 1)
            {
                planValid = AddStaticToken(",");
            }
            planValid = planValid && AddStaticToken("))");
            return planValid ? ConstraintKind::Tuple : ConstraintKind::Invalid;
        }

        AliasGroup* ParseAliasGroups(bool& planValid, size_t& aliasCount)
        {
            AliasGroup* pFirstAlias = nullptr;
            AliasGroup* pLastAlias = nullptr;
            aliasCount = 0;
            bool hasPreviousGroupId = false;
            unsigned long long previousGroupId = 0;
            while (planValid && ConsumeInput('g'))
            {
                const size_t groupOffset = m_inputIndex;
                unsigned long long groupId = 0;
                size_t digitCount = 0;
                const unsigned long long nextAllowedGroupId = m_lastAliasGroupId < 0
                                                                ? 0
                                                                : static_cast<unsigned long long>(m_lastAliasGroupId) + 1;
                planValid = ParseUnsignedHex(groupId, digitCount)
                         && !(digitCount > 1 && m_pInput[groupOffset] == '0')
                         && groupId <= static_cast<unsigned long long>(std::numeric_limits<long long>::max())
                         && groupId <= nextAllowedGroupId
                         && (!hasPreviousGroupId || groupId > previousGroupId);
                if (!planValid)
                {
                    break;
                }
                AliasGroup* pAlias = AllocateObject<AliasGroup>();
                if (!pAlias)
                {
                    planValid = false;
                    break;
                }
                pAlias->pNext = nullptr;
                pAlias->groupId = groupId;
                if (pLastAlias)
                {
                    pLastAlias->pNext = pAlias;
                }
                else
                {
                    pFirstAlias = pAlias;
                }
                pLastAlias = pAlias;
                m_lastAliasGroupId = static_cast<long long>(groupId);
                previousGroupId = groupId;
                hasPreviousGroupId = true;
                ++aliasCount;
            }
            return pFirstAlias;
        }

        bool AddAliasTuple(AliasGroup* pAlias, size_t aliasCount)
        {
            bool planValid = AddStaticToken("(");
            size_t aliasIndex = 0;
            while (planValid && pAlias)
            {
                if (aliasIndex++ != 0)
                {
                    planValid = AddStaticToken(", ");
                }
                planValid = planValid && AddAliasGroupToken(pAlias->groupId);
                pAlias = pAlias->pNext;
            }
            if (planValid && aliasCount == 1)
            {
                planValid = AddStaticToken(",");
            }
            return planValid && AddStaticToken(")");
        }

        ConstraintKind ParseListConstraint(unsigned long long version, bool& planValid)
        {
            size_t aliasCount = 0;
            AliasGroup* pAliases = ParseAliasGroups(planValid, aliasCount);
            const bool elementsMayAlias = ConsumeInput('i');
            planValid = planValid && AddStaticToken("ListConstraint(element=");
            ConstraintKind elementKind = ConstraintKind::Invalid;
            if (planValid)
            {
                elementKind = ParseConstraint(version, planValid);
            }
            planValid = planValid && elementKind == ConstraintKind::Array
                     && AddStaticToken(", alias_groups=")
                     && AddAliasTuple(pAliases, aliasCount)
                     && AddStaticToken(", elements_may_alias=")
                     && AddStaticToken(elementsMayAlias ? "True" : "False")
                     && AddStaticToken(")");
            return planValid ? ConstraintKind::List : ConstraintKind::Invalid;
        }

        static size_t HexBitLength(const char* pHexDigits, size_t digitCount)
        {
            for (size_t digitIndex = 0; digitIndex < digitCount; ++digitIndex)
            {
                const int digitValue = HexValue(pHexDigits[digitIndex]);
                if (digitValue == 0)
                {
                    continue;
                }
                const size_t highBitCount = digitValue >= 8 ? 4 : digitValue >= 4 ? 3
                                                            : digitValue >= 2     ? 2
                                                                                  : 1;
                return (digitCount - digitIndex - 1) * 4 + highBitCount;
            }
            return 0;
        }

        static bool HexMaskHasBit(const char* pHexDigits, size_t digitCount, size_t bitIndex)
        {
            const size_t nibbleFromEnd = bitIndex / 4;
            if (nibbleFromEnd >= digitCount)
            {
                return false;
            }
            const size_t digitIndex = digitCount - 1 - nibbleFromEnd;
            return (HexValue(pHexDigits[digitIndex]) & (1 << (bitIndex % 4))) != 0;
        }

        bool AddOptionalAxisTuple(
            const AxisState* pAxes,
            size_t dimensionCount,
            uint8_t AxisState::* pHasValue,
            long long AxisState::* pValue)
        {
            bool planValid = AddStaticToken("(");
            for (size_t axisIndex = 0; planValid && axisIndex < dimensionCount; ++axisIndex)
            {
                if (axisIndex != 0)
                {
                    planValid = AddStaticToken(", ");
                }
                planValid = planValid && (pAxes[axisIndex].*pHasValue ? AddSignedIntegerToken(pAxes[axisIndex].*pValue) : AddStaticToken("None"));
            }
            if (planValid && dimensionCount == 1)
            {
                planValid = AddStaticToken(",");
            }
            return planValid && AddStaticToken(")");
        }

        bool AddAxisTuple(
            const AxisState* pAxes,
            size_t dimensionCount,
            long long AxisState::* pValue)
        {
            bool planValid = AddStaticToken("(");
            for (size_t axisIndex = 0; planValid && axisIndex < dimensionCount; ++axisIndex)
            {
                if (axisIndex != 0)
                {
                    planValid = AddStaticToken(", ");
                }
                planValid = planValid && AddSignedIntegerToken(pAxes[axisIndex].*pValue);
            }
            if (planValid && dimensionCount == 1)
            {
                planValid = AddStaticToken(",");
            }
            return planValid && AddStaticToken(")");
        }

        ConstraintKind ParseArrayConstraint(unsigned long long version, bool& planValid)
        {
            unsigned long long dimensionCountValue = 0;
            if (!ParseUnsignedDecimal(dimensionCountValue)
                || dimensionCountValue > SIZE_MAX
                || dimensionCountValue > static_cast<unsigned long long>(std::numeric_limits<long long>::max()))
            {
                planValid = false;
                return ConstraintKind::Invalid;
            }
            const size_t dimensionCount = static_cast<size_t>(dimensionCountValue);
            size_t dtypeOffset = 0;
            size_t dtypeLength = 0;
            planValid = ParseDType(dtypeOffset, dtypeLength);
            AxisState* pAxes = planValid ? AllocateArray<AxisState>(dimensionCount) : nullptr;
            if (dimensionCount != 0 && !pAxes)
            {
                planValid = false;
            }
            for (size_t axisIndex = 0; planValid && axisIndex < dimensionCount; ++axisIndex)
            {
                pAxes[axisIndex].shapeConstant = 0;
                pAxes[axisIndex].shapeDivisibleBy = 1;
                pAxes[axisIndex].strideConstant = 0;
                pAxes[axisIndex].strideDivisibleBy = 1;
                pAxes[axisIndex].strideLowerBound = 0;
                pAxes[axisIndex].hasShapeConstant = 0;
                pAxes[axisIndex].hasStrideConstant = 0;
                pAxes[axisIndex].hasStrideLowerBound = 0;
            }

            while (planValid && PeekInput() == '_' && IsLowerHexDigit(PeekInput(1)))
            {
                ++m_inputIndex;
                const size_t maskOffset = m_inputIndex;
                while (IsLowerHexDigit(PeekInput()))
                {
                    ++m_inputIndex;
                }
                const size_t maskLength = m_inputIndex - maskOffset;
                const size_t maskBitLength = HexBitLength(m_pInput + maskOffset, maskLength);
                planValid = maskBitLength != 0 && maskBitLength <= dimensionCount;

                bool hasShapeConstant = false;
                long long shapeConstant = 0;
                if (planValid && ConsumeInput('s'))
                {
                    hasShapeConstant = true;
                    planValid = version >= 2 && ParseSignedInteger(shapeConstant);
                }
                long long shapeDivisibleBy = 1;
                if (planValid && ConsumeInput('i'))
                {
                    planValid = ParseDivisibility(shapeDivisibleBy);
                }
                bool hasStrideConstant = false;
                long long strideConstant = 0;
                if (planValid && ConsumeInput('t'))
                {
                    hasStrideConstant = true;
                    planValid = ParseSignedInteger(strideConstant);
                }
                long long strideDivisibleBy = 1;
                if (planValid && ConsumeInput('v'))
                {
                    planValid = ParseDivisibility(strideDivisibleBy);
                }
                bool hasStrideLowerBound = false;
                long long strideLowerBound = 0;
                if (planValid && ConsumeInput('l'))
                {
                    hasStrideLowerBound = true;
                    planValid = ParseSignedInteger(strideLowerBound);
                }

                for (size_t axisIndex = 0; planValid && axisIndex < maskBitLength; ++axisIndex)
                {
                    if (!HexMaskHasBit(m_pInput + maskOffset, maskLength, axisIndex))
                    {
                        continue;
                    }
                    AxisState& axis = pAxes[axisIndex];
                    if (hasShapeConstant)
                    {
                        planValid = axis.hasShapeConstant == 0;
                        axis.hasShapeConstant = 1;
                        axis.shapeConstant = shapeConstant;
                    }
                    if (shapeDivisibleBy != 1)
                    {
                        planValid = planValid && axis.shapeDivisibleBy == 1;
                        axis.shapeDivisibleBy = shapeDivisibleBy;
                    }
                    if (hasStrideConstant)
                    {
                        planValid = planValid && axis.hasStrideConstant == 0;
                        axis.hasStrideConstant = 1;
                        axis.strideConstant = strideConstant;
                    }
                    if (strideDivisibleBy != 1)
                    {
                        planValid = planValid && axis.strideDivisibleBy == 1;
                        axis.strideDivisibleBy = strideDivisibleBy;
                    }
                    if (hasStrideLowerBound)
                    {
                        planValid = planValid && axis.hasStrideLowerBound == 0;
                        axis.hasStrideLowerBound = 1;
                        axis.strideLowerBound = strideLowerBound;
                    }
                }
            }

            for (size_t axisIndex = 0; planValid && axisIndex < dimensionCount; ++axisIndex)
            {
                const AxisState& axis = pAxes[axisIndex];
                planValid = axis.hasStrideConstant == 0
                         || (axis.strideDivisibleBy == 1 && axis.hasStrideLowerBound == 0);
            }

            long long baseAddressDivisibleBy = 1;
            AliasGroup* pAliases = nullptr;
            size_t aliasCount = 0;
            bool mayAliasInternally = false;
            const char* pIndexDTypeCode = "i32";
            size_t indexDTypeLength = 3;
            if (planValid && PeekInput() == '_' && 'a' <= PeekInput(1) && PeekInput(1) <= 'z')
            {
                ++m_inputIndex;
                if (ConsumeInput('p'))
                {
                    planValid = ParseDivisibility(baseAddressDivisibleBy);
                }
                if (planValid)
                {
                    pAliases = ParseAliasGroups(planValid, aliasCount);
                }
                mayAliasInternally = planValid && ConsumeInput('i');
                if (planValid && ConsumeInput('u'))
                {
                    pIndexDTypeCode = "u32";
                }
                else if (planValid && ConsumeInput('w'))
                {
                    pIndexDTypeCode = "i64";
                }
            }

            planValid = planValid
                     && AddStaticToken("ArrayConstraint(dtype=")
                     && AddDTypeToken(dtypeOffset, dtypeLength)
                     && AddStaticToken(", ndim=")
                     && AddSignedIntegerToken(static_cast<long long>(dimensionCount))
                     && AddStaticToken(", index_dtype=<DType '")
                     && AddStaticToken(DTypeName(pIndexDTypeCode, indexDTypeLength))
                     && AddStaticToken("'>, stride_lower_bound_incl=")
                     && AddOptionalAxisTuple(pAxes, dimensionCount, &AxisState::hasStrideLowerBound, &AxisState::strideLowerBound)
                     && AddStaticToken(", alias_groups=")
                     && AddAliasTuple(pAliases, aliasCount)
                     && AddStaticToken(", may_alias_internally=")
                     && AddStaticToken(mayAliasInternally ? "True" : "False")
                     && AddStaticToken(", stride_constant=")
                     && AddOptionalAxisTuple(pAxes, dimensionCount, &AxisState::hasStrideConstant, &AxisState::strideConstant)
                     && AddStaticToken(", shape_constant=")
                     && AddOptionalAxisTuple(pAxes, dimensionCount, &AxisState::hasShapeConstant, &AxisState::shapeConstant)
                     && AddStaticToken(", stride_divisible_by=")
                     && AddAxisTuple(pAxes, dimensionCount, &AxisState::strideDivisibleBy)
                     && AddStaticToken(", shape_divisible_by=")
                     && AddAxisTuple(pAxes, dimensionCount, &AxisState::shapeDivisibleBy)
                     && AddStaticToken(", base_addr_divisible_by=")
                     && AddSignedIntegerToken(baseAddressDivisibleBy)
                     && AddStaticToken(")");
            return planValid ? ConstraintKind::Array : ConstraintKind::Invalid;
        }

    private:
        // Rendering follows the token list linearly. Each fragment is either static
        // storage, an input slice, or the small numeric formatting buffer. Output
        // limits retain the token and UTF-8-safe fragment offset for resumption.

        void GetTokenFragment(const PlanToken& token, const char*& pFragment, size_t& fragmentLength)
        {
            if (token.kind == TokenKind::StaticText)
            {
                pFragment = token.pStaticText;
                fragmentLength = std::strlen(pFragment);
                return;
            }
            if (token.kind == TokenKind::InputText)
            {
                pFragment = m_pInput + token.inputOffset;
                fragmentLength = token.inputLength;
                return;
            }
            if (token.kind == TokenKind::DType)
            {
                const char* pDTypeName = DTypeName(m_pInput + token.inputOffset, token.inputLength);
                const int formattedLength = std::snprintf(m_formatBuffer, sizeof(m_formatBuffer), "<DType '%s'>", pDTypeName ? pDTypeName : "?");
                pFragment = m_formatBuffer;
                fragmentLength = formattedLength > 0 ? static_cast<size_t>(formattedLength) : 0;
                return;
            }
            if (token.kind == TokenKind::SignedInteger)
            {
                long long signedValue = 0;
                std::memcpy(&signedValue, &token.unsignedValue, sizeof(signedValue));
                const int formattedLength = std::snprintf(m_formatBuffer, sizeof(m_formatBuffer), "%lld", signedValue);
                pFragment = m_formatBuffer;
                fragmentLength = formattedLength > 0 ? static_cast<size_t>(formattedLength) : 0;
                return;
            }
            if (token.kind == TokenKind::AliasGroup)
            {
                const int formattedLength = std::snprintf(m_formatBuffer, sizeof(m_formatBuffer), "'group%" PRIu64 "'", token.unsignedValue);
                pFragment = m_formatBuffer;
                fragmentLength = formattedLength > 0 ? static_cast<size_t>(formattedLength) : 0;
                return;
            }

            double floatValue = 0.0;
            std::memcpy(&floatValue, &token.unsignedValue, sizeof(floatValue));
            int formattedLength = 0;
            if (std::isnan(floatValue))
            {
                formattedLength = std::snprintf(m_formatBuffer, sizeof(m_formatBuffer), "nan");
            }
            else if (std::isinf(floatValue))
            {
                formattedLength = std::snprintf(m_formatBuffer, sizeof(m_formatBuffer), "%s", std::signbit(floatValue) ? "-inf" : "inf");
            }
            else
            {
                formattedLength = std::snprintf(m_formatBuffer, sizeof(m_formatBuffer), "%.15g", floatValue);
                if (formattedLength > 0
                    && !std::strchr(m_formatBuffer, '.')
                    && !std::strchr(m_formatBuffer, 'e')
                    && !std::strchr(m_formatBuffer, 'E')
                    && static_cast<size_t>(formattedLength) + 2 < sizeof(m_formatBuffer))
                {
                    m_formatBuffer[formattedLength++] = '.';
                    m_formatBuffer[formattedLength++] = '0';
                    m_formatBuffer[formattedLength] = '\0';
                }
            }
            pFragment = m_formatBuffer;
            fragmentLength = formattedLength > 0 ? static_cast<size_t>(formattedLength) : 0;
        }

        static size_t ClampToUtf8Boundary(const char* pBytes, size_t byteCount, size_t requestedByteCount)
        {
            if (requestedByteCount >= byteCount)
            {
                return byteCount;
            }
            size_t acceptedByteCount = requestedByteCount;
            while (acceptedByteCount != 0
                   && (static_cast<uint8_t>(pBytes[acceptedByteCount]) & 0xc0) == 0x80)
            {
                --acceptedByteCount;
            }
            return acceptedByteCount;
        }

        static bool IsValidUtf8(const char* pBytes, size_t byteCount)
        {
            size_t byteIndex = 0;
            while (byteIndex < byteCount)
            {
                const uint8_t leadingByte = static_cast<uint8_t>(pBytes[byteIndex]);
                if (leadingByte < 0x80)
                {
                    if (leadingByte == 0)
                    {
                        return false;
                    }
                    ++byteIndex;
                    continue;
                }
                size_t sequenceLength = 0;
                uint32_t codePoint = 0;
                uint32_t minimumCodePoint = 0;
                if (0xc2 <= leadingByte && leadingByte <= 0xdf)
                {
                    sequenceLength = 2;
                    codePoint = leadingByte & 0x1f;
                    minimumCodePoint = 0x80;
                }
                else if (0xe0 <= leadingByte && leadingByte <= 0xef)
                {
                    sequenceLength = 3;
                    codePoint = leadingByte & 0x0f;
                    minimumCodePoint = 0x800;
                }
                else if (0xf0 <= leadingByte && leadingByte <= 0xf4)
                {
                    sequenceLength = 4;
                    codePoint = leadingByte & 0x07;
                    minimumCodePoint = 0x10000;
                }
                else
                {
                    return false;
                }
                if (sequenceLength > byteCount - byteIndex)
                {
                    return false;
                }
                for (size_t continuationIndex = 1; continuationIndex < sequenceLength; ++continuationIndex)
                {
                    const uint8_t continuationByte = static_cast<uint8_t>(pBytes[byteIndex + continuationIndex]);
                    if ((continuationByte & 0xc0) != 0x80)
                    {
                        return false;
                    }
                    codePoint = (codePoint << 6) | (continuationByte & 0x3f);
                }
                if (codePoint < minimumCodePoint || codePoint > 0x10ffff
                    || (0xd800 <= codePoint && codePoint <= 0xdfff))
                {
                    return false;
                }
                byteIndex += sequenceLength;
            }
            return true;
        }

    private:
        const char* m_pInput = nullptr;
        size_t m_inputLength = 0;
        size_t m_inputIndex = 0;
        uint8_t* m_pScratchSpace = nullptr;
        size_t m_scratchSpaceSize = 0;
        size_t m_requiredScratchSpaceSize = 0;
        size_t m_scratchBytesUsed = 0;
        PlanToken* m_pFirstToken = nullptr;
        PlanToken* m_pLastToken = nullptr;
        PlanToken* m_pRenderToken = nullptr;
        long long m_lastAliasGroupId = -1;
        Status m_parseStatus = Status::NotParsed;
        size_t m_renderedOutputLength = 0;
        size_t m_fragmentOffset = 0;
        bool m_allocationFailed = false;
        bool m_renderStarted = false;
        char m_formatBuffer[96] = {};
    };

} // namespace nv::trace
