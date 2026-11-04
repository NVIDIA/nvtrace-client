/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__linux__)
#include <sys/types.h>
#endif

namespace nv::trace {

    // A resolved OS user or group identity, used in access rule subjects and target lists.
    // "Sid" is the project-wide shorthand for OsSecurityIdentifier:
    // Linux: uid/gid
    struct OsSecurityIdentifier
    {
#if defined(__linux__)
        enum class Type
        {
            User,
            Group
        } type = Type::User;

        union
        {
            uid_t uid = uid_t(-1);
            gid_t gid;
        };

        bool operator==(const OsSecurityIdentifier& rhs) const
        {
            return type == rhs.type && (type == Type::User ? uid == rhs.uid : gid == rhs.gid);
        }
        bool operator!=(const OsSecurityIdentifier& rhs) const
        {
            return !operator==(rhs);
        }
#endif
    };


    enum class WellKnownSid
    {
        CurrentUser,
        Administrators,
        Everyone
    };
    bool ResolveWellKnownSid(WellKnownSid wellKnown, OsSecurityIdentifier& outSid);
    bool ResolveNamedSid(const char* pName, OsSecurityIdentifier& outSid);

} //namespace nv::trace


#if defined(__linux__)
#include <NvTrace/Target/Router/NvTraceOsSecurityIdentifierLinux.h>
#endif
