/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#if defined(__GNUC__) || defined(__clang__)
#include <stdalign.h>
#endif

// Helper macros for stringifying
#define NVTRACE_STRINGIZE(x_) #x_
#define NVTRACE_TOSTRING(x_) NVTRACE_STRINGIZE(x_)

// GCC-specific macros
#if defined(__GNUC__)
#define NVTRACE_PRAGMA(x_) _Pragma(NVTRACE_TOSTRING(x_))
#define NVTRACE_SUPPRESS_WARNINGS() \
    NVTRACE_PRAGMA(GCC diagnostic push)


#define NVTRACE_RESTORE_WARNINGS() \
    NVTRACE_PRAGMA(GCC diagnostic pop)

// No-op for other compilers
#else
#define NVTRACE_SUPPRESS_WARNINGS()
#define NVTRACE_RESTORE_WARNINGS()
#endif


// Ability to use alignas() in C11, if the stdlib didn't define the macro.
// https://en.cppreference.com/w/cpp/language/alignas#Notes
#if !defined(__cplusplus) && !defined(alignas)
#if defined(__GNUC__) || defined(__clang__)
#define alignas(bytes_) _Alignas(bytes_)
#endif // compiler
#endif


// Ability to define weak and const weak symbols, for header-only definitions of global constants.
#if defined(__cplusplus) && __cplusplus >= 201703L
#define NVTRACE_WEAK inline
#define NVTRACE_WEAK_CONST constexpr inline
#elif defined(__GNUC__) || defined(__clang__)
#define NVTRACE_WEAK __attribute__((weak))
#define NVTRACE_WEAK_CONST const __attribute__((weak))
#else
#error "NVTRACE_WEAK needs to be ported to this compiler."
#endif

// Export macro for shared libraries.
#if defined(__linux__)
#define NVTRACE_EXPORT __attribute__((visibility("default")))
#else
#error "Unsupported platform."
#endif
