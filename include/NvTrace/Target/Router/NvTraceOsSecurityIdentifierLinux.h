/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/Target/Router/NvTraceOsSecurityIdentifier.h>

#include <cstring>
#include <grp.h>
#include <pwd.h>
#include <strings.h>
#include <unistd.h>

namespace nv::trace {

    inline bool ResolveWellKnownSid(WellKnownSid wellKnownSid, OsSecurityIdentifier& outSid)
    {
        switch (wellKnownSid)
        {
            case WellKnownSid::CurrentUser:
                outSid.type = OsSecurityIdentifier::Type::User;
                outSid.uid = geteuid();
                return true;

            case WellKnownSid::Administrators:
                outSid.type = OsSecurityIdentifier::Type::User;
                outSid.uid = 0;
                return true;

            case WellKnownSid::Everyone:
                outSid.type = OsSecurityIdentifier::Type::Group;
                outSid.gid = gid_t(-1);
                return true;
        }

        return false;
    }

    inline bool ResolveNamedSid(const char* pName, OsSecurityIdentifier& outSid)
    {
        // '/everyone' = world-open principal; not a legal user/group name, so no clash.
        if (strcasecmp(pName, "/everyone") == 0)
        {
            outSid.type = OsSecurityIdentifier::Type::Group;
            outSid.gid = gid_t(-1);
            return true;
        }

        char buf[1024];

        // Try OS group name first
        if (struct group grpBuf, *pGrp; getgrnam_r(pName, &grpBuf, buf, sizeof(buf), &pGrp) == 0 && pGrp)
        {
            outSid.type = OsSecurityIdentifier::Type::Group;
            outSid.gid = pGrp->gr_gid;
            return true;
        }

        // Try OS user name
        if (struct passwd pwdBuf, *pPwd; getpwnam_r(pName, &pwdBuf, buf, sizeof(buf), &pPwd) == 0 && pPwd)
        {
            outSid.type = OsSecurityIdentifier::Type::User;
            outSid.uid = pPwd->pw_uid;
            return true;
        }

        return false;
    }

} // namespace nv::trace
