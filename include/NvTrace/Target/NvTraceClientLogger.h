/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <string>
#include <time.h>

#include <sys/time.h>
#define NVTRACE_PATH_SEPARATOR '/'


namespace nv::trace::Logger {

    inline int FormatTimeCommon(char* pBuf, size_t size, uint32_t hour, uint32_t minute, uint32_t second, uint32_t milliSecond)
    {
        const int written = snprintf(pBuf, size, "%02u:%02u:%02u:%03u", hour, minute, second, milliSecond);
        return written;
    }

    inline int FormatDateCommon(char* pBuf, size_t size, uint32_t year, uint32_t month, uint32_t day)
    {
        const char* pMonth = [&]() {
            static const char* s_months[12] = {
                "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"};

            if (1 <= month && month <= 12)
            {
                return s_months[month - 1];
            }
            return "???";
        }();
        const int written = snprintf(pBuf, size, "%4u-%s-%02u", year, pMonth, day);
        return written;
    }

    typedef struct timeval LogTimeStamp;

    inline void UserLogImplPlatform(const char* /* pMessage */)
    {
    }

    inline void GetTimeStamp(LogTimeStamp* pTimestamp)
    {
        gettimeofday(pTimestamp, 0);
    }

    inline size_t FormatTime(LogTimeStamp* pTimestamp, char* pBuf, size_t size)
    {
        const struct tm* ltm = localtime(&pTimestamp->tv_sec);
        int milliseconds = static_cast<int>(pTimestamp->tv_usec / 1000);
        return FormatTimeCommon(pBuf, size, (uint32_t)ltm->tm_hour, (uint32_t)ltm->tm_min, (uint32_t)ltm->tm_sec, (uint32_t)milliseconds);
    }

    inline size_t FormatDate(LogTimeStamp* pTimestamp, char* pBuf, size_t size)
    {
        const struct tm* ltm = localtime(&pTimestamp->tv_sec);
        return FormatDateCommon(pBuf, size, (uint32_t)ltm->tm_year + 1900, (uint32_t)ltm->tm_mon + 1, (uint32_t)ltm->tm_mday);
    }

    inline bool GetEnvVariable(const char* pName, std::string& value)
    {
        auto pValue = std::getenv(pName);
        if (!pValue)
        {
            value = "";
            return false;
        }
        value = pValue;
        return true;
    }

    inline FILE* OpenFile(const char* pFileName, const char* pMode)
    {
        FILE* pFile = std::fopen(pFileName, pMode);
        return pFile;
    }

    enum class LogSeverity
    {
        Inf,
        Wrn,
        Err,
        COUNT
    };

    struct UserLogParameters
    {
        LogSeverity severity;
        uint32_t level;
        const char* pPrefix;
        const char* pDate;
        const char* pTime;
        const char* pFunctionName;
        const char* pFile;
        int line;
        const char* pMessage;
    };

    using UserLoggingFn = void (*)(void* pUserData, const UserLogParameters& params);

    struct LogSettings
    {
        uint32_t volumeLevels[(unsigned)LogSeverity::COUNT] = {50, 50, 50};

#if defined(__linux__)
        bool writePlatform = false;
#endif
        bool writeStderr = false;
        FILE* writeFileFD = nullptr;
        bool appendToFile = true;
        LogSeverity flushFileSeverity = LogSeverity::Err;
        UserLoggingFn pfnWriteUserLog = nullptr;
        void* pUserLogUserData = nullptr;

        bool logDate = true;
        bool logTime = true;
        /// When true, include source file and line before the message on platform/stderr/file sinks.
        bool logFileLine = true;

        LogSettings()
        {
            {
                std::string envValue;
                if (GetEnvVariable("NVTRACE_LOG_ENABLE_STDERR", envValue))
                {
                    char* pEnd = nullptr;
                    writeStderr = !!strtol(envValue.c_str(), &pEnd, 0);
                }
            }
            {
                std::string envValue;
                if (GetEnvVariable("NVTRACE_LOG_ENABLE_FILE", envValue))
                {
                    FILE* fp = OpenFile(envValue.c_str(), appendToFile ? "a" : "w");
                    writeFileFD = fp;
                }
            }
            {
                std::string envValue;
                if (GetEnvVariable("NVTRACE_LOG_FILE_FLUSH_SEVERITY", envValue))
                {
                    char* pEnd = nullptr;
                    int severity = static_cast<int>(strtol(envValue.c_str(), &pEnd, 0));
                    if (0 <= severity && severity < (int)LogSeverity::COUNT)
                    {
                        flushFileSeverity = (LogSeverity)severity;
                    }
                }
            }
            {
                std::string envValue;
                if (GetEnvVariable("NVTRACE_LOG_FILE_LINE", envValue))
                {
                    char* pEnd = nullptr;
                    logFileLine = !!strtol(envValue.c_str(), &pEnd, 0);
                }
            }
        }

        LogSettings(const LogSettings&) = delete;
        LogSettings& operator=(const LogSettings&) = delete;

        ~LogSettings()
        {
            if (writeFileFD)
            {
                fclose(writeFileFD);
            }
        }
    };


    inline LogSettings* GetLogSettingsStorage_()
    {
        static LogSettings settings;
        return &settings;
    }

    inline uint32_t GetLogVolumeLevel(LogSeverity severity)
    {
        LogSettings* pSettings = GetLogSettingsStorage_();
        if ((uint32_t)severity < 3)
        {
            return pSettings->volumeLevels[(uint32_t)severity];
        }
        return 0;
    }

    inline void UserLogImplStderr(const char* pMessage)
    {
        fprintf(stderr, "%s", pMessage);
    }

    inline void UserLogImplFile(const char* pMessage, FILE* fd)
    {
        fprintf(fd, "%s", pMessage);
    }

    inline void UserLogImplFileFlush(FILE* fd)
    {
        fflush(fd);
    }

} // namespace nv::trace::Logger

namespace nv::trace {

    // Higher values produce more log output.  0 <= volumeLevel <= 100
    // Technically it's more like a noise floor (all messages below this level are treated as noise and discarded).
    inline void SetLogVolumeLevel(::nv::trace::Logger::LogSeverity severity, uint32_t volumeLevel)
    {
        using namespace nv::trace::Logger;
        LogSettings* pSettings = GetLogSettingsStorage_();
        if ((uint32_t)severity < 3)
        {
            pSettings->volumeLevels[(uint32_t)severity] = volumeLevel;
        }
    }

    inline void SetLogAppendToFile(bool enable)
    {
        using namespace nv::trace::Logger;
        LogSettings* pSettings = GetLogSettingsStorage_();
        pSettings->appendToFile = enable;
    }

    inline void SetLogFlushSeverity(::nv::trace::Logger::LogSeverity severity)
    {
        using namespace nv::trace::Logger;
        LogSettings* pSettings = GetLogSettingsStorage_();
        if (0 <= (int)severity && (int)severity < (int)LogSeverity::COUNT)
        {
            pSettings->flushFileSeverity = severity;
        }
    }

    inline void SetLogDate(bool enable)
    {
        using namespace nv::trace::Logger;
        LogSettings* pSettings = GetLogSettingsStorage_();
        pSettings->logDate = enable;
    }

    inline void SetLogTime(bool enable)
    {
        using namespace nv::trace::Logger;
        LogSettings* pSettings = GetLogSettingsStorage_();
        pSettings->logTime = enable;
    }

    inline void SetLogFileLine(bool enable)
    {
        using namespace nv::trace::Logger;
        LogSettings* pSettings = GetLogSettingsStorage_();
        pSettings->logFileLine = enable;
    }

    inline bool UserLogEnablePlatform(bool enable)
    {
        using namespace nv::trace::Logger;
        LogSettings* pSettings = GetLogSettingsStorage_();
        pSettings->writePlatform = enable;
        return true;
    }

    inline bool UserLogEnableStderr(bool enable)
    {
        using namespace nv::trace::Logger;
        LogSettings* pSettings = GetLogSettingsStorage_();
        pSettings->writeStderr = enable;
        return true;
    }

    inline bool UserLogEnableFile(const char* filename)
    {
        using namespace nv::trace::Logger;
        LogSettings* pSettings = GetLogSettingsStorage_();
        if (filename)
        {
            FILE* fp = OpenFile(filename, pSettings->appendToFile ? "a" : "w");
            if (!fp)
            {
                return false;
            }
            pSettings->writeFileFD = fp;
        }
        return true;
    }

    inline void UserLogEnableCustom(void* pUserData, ::nv::trace::Logger::UserLoggingFn cb)
    {
        using namespace nv::trace::Logger;
        LogSettings* pSettings = GetLogSettingsStorage_();
        pSettings->pfnWriteUserLog = cb;
        pSettings->pUserLogUserData = pUserData;
    }

    inline void UserLogDisableCustom()
    {
        UserLogEnableCustom(nullptr, nullptr);
    }

    // Emit a log mesage.
    // Check log volume level before calling this function.
    inline void UserLog(::nv::trace::Logger::LogSeverity severity, uint32_t level, const char* pFunctionName, const char* pFile, int line, const char* pFormat, ...)
    {
        using namespace nv::trace::Logger;

        LogSettings& settings = *GetLogSettingsStorage_();

        va_list args;

        va_start(args, pFormat);
        const size_t length = vsnprintf(nullptr, 0, pFormat, args);
        va_end(args);

        std::string str;
        str.append(length + 1, ' ');
        va_start(args, pFormat);
        vsnprintf(&str[0], length + 1, pFormat, args);
        va_end(args);
        str.back() = '\0'; // ensure NULL terminated

        const char* const pPrefix = [&]() {
            switch (severity)
            {
                case (LogSeverity::Inf): return "NVTRACE|INF|";
                case (LogSeverity::Wrn): return "NVTRACE|WRN|";
                case (LogSeverity::Err): return "NVTRACE|ERR|";
                default: return "NVTRACE|???|";
            }
        }();

        char datebuf[16];
        char timebuf[16];
        if (settings.logDate || settings.logTime)
        {
            LogTimeStamp time;
            GetTimeStamp(&time);
            if (settings.logDate)
            {
                FormatDate(&time, datebuf, sizeof(datebuf));
            }
            if (settings.logTime)
            {
                FormatTime(&time, timebuf, sizeof(timebuf));
            }
        }

        if (settings.writePlatform || settings.writeStderr || settings.writeFileFD)
        {
            // TODO: stage the log message in a buffer, to ensure atomicity of writes.
            if (settings.writePlatform)
            {
                UserLogImplPlatform(pPrefix);
                if (settings.logDate)
                {
                    UserLogImplPlatform(datebuf);
                    UserLogImplPlatform("|");
                }
                if (settings.logTime)
                {
                    UserLogImplPlatform(timebuf);
                    UserLogImplPlatform("|");
                }
                if (settings.logFileLine)
                {
                    UserLogImplPlatform(pFile);
                    UserLogImplPlatform(":");
                    UserLogImplPlatform(std::to_string(line).c_str());
                    UserLogImplPlatform(" | ");
                }
                UserLogImplPlatform(pFunctionName);
                UserLogImplPlatform(" || ");
                UserLogImplPlatform(str.c_str());
                UserLogImplPlatform("\n");
            }
            if (settings.writeStderr)
            {
                UserLogImplStderr(pPrefix);
                if (settings.logDate)
                {
                    UserLogImplStderr(datebuf);
                    UserLogImplStderr("|");
                }
                if (settings.logTime)
                {
                    UserLogImplStderr(timebuf);
                    UserLogImplStderr("|");
                }
                if (settings.logFileLine)
                {
                    UserLogImplStderr(pFile);
                    UserLogImplStderr(":");
                    UserLogImplStderr(std::to_string(line).c_str());
                    UserLogImplStderr(" | ");
                }
                UserLogImplStderr(pFunctionName);
                UserLogImplStderr(" || ");
                UserLogImplStderr(str.c_str());
                UserLogImplStderr("\n");
            }
            if (settings.writeFileFD)
            {
                UserLogImplFile(pPrefix, settings.writeFileFD);
                if (settings.logDate)
                {
                    UserLogImplFile(datebuf, settings.writeFileFD);
                    UserLogImplFile("|", settings.writeFileFD);
                }
                if (settings.logTime)
                {
                    UserLogImplFile(timebuf, settings.writeFileFD);
                    UserLogImplFile("|", settings.writeFileFD);
                }
                if (settings.logFileLine)
                {
                    UserLogImplFile(pFile, settings.writeFileFD);
                    UserLogImplFile(":", settings.writeFileFD);
                    UserLogImplFile(std::to_string(line).c_str(), settings.writeFileFD);
                    UserLogImplFile(" | ", settings.writeFileFD);
                }
                UserLogImplFile(pFunctionName, settings.writeFileFD);
                UserLogImplFile(" || ", settings.writeFileFD);
                UserLogImplFile(str.c_str(), settings.writeFileFD);
                UserLogImplFile("\n", settings.writeFileFD);
                if (severity >= settings.flushFileSeverity)
                {
                    UserLogImplFileFlush(settings.writeFileFD);
                }
            }
        }
        if (settings.pfnWriteUserLog)
        {
            UserLogParameters params = {};
            params.severity = severity;
            params.level = level;
            params.pPrefix = pPrefix;
            params.pDate = settings.logDate ? datebuf : nullptr;
            params.pTime = settings.logTime ? timebuf : nullptr;
            params.pFunctionName = pFunctionName;
            params.pFile = pFile;
            params.line = line;
            params.pMessage = str.c_str();
            settings.pfnWriteUserLog(settings.pUserLogUserData, params);
        }
    }

} // namespace nv::trace

#ifndef NVTRACE_LOGGABLE
#define NVTRACE_LOGGABLE(severity_, level_) (::nv::trace::Logger::GetLogVolumeLevel(::nv::trace::Logger::LogSeverity::severity_) >= level_)
#endif

#ifndef NVTRACE_LOG
#define NVTRACE_LOG(severity_, level_, ...)                                                                                       \
    do                                                                                                                            \
    {                                                                                                                             \
        if (NVTRACE_LOGGABLE(severity_, level_))                                                                                  \
        {                                                                                                                         \
            ::nv::trace::UserLog(::nv::trace::Logger::LogSeverity::severity_, level_, __func__, __FILE__, __LINE__, __VA_ARGS__); \
        }                                                                                                                         \
    } while (false)
#endif

#ifndef NVTRACE_LOG_IF
#define NVTRACE_LOG_IF(condition_, severity_, level_, ...) \
    do                                                     \
    {                                                      \
        if (condition_)                                    \
            NVTRACE_LOG(severity_, level_, __VA_ARGS__);   \
    } while (false)
#endif

#ifndef NVTRACE_LOG_DO_IF
#define NVTRACE_LOG_DO_IF(condition_, action_, severity_, level_, ...) \
    if (condition_)                                                    \
    {                                                                  \
        NVTRACE_LOG(severity_, level_, __VA_ARGS__);                   \
        action_                                                        \
    }                                                                  \
    do                                                                 \
    {                                                                  \
    } while (false)
#endif

#ifndef NVTRACE_LOG_INF
#define NVTRACE_LOG_INF(level_, ...) NVTRACE_LOG(Inf, level_, __VA_ARGS__)
#endif
#ifndef NVTRACE_LOG_WRN
#define NVTRACE_LOG_WRN(level_, ...) NVTRACE_LOG(Wrn, level_, __VA_ARGS__)
#endif
#ifndef NVTRACE_LOG_ERR
#define NVTRACE_LOG_ERR(level_, ...) NVTRACE_LOG(Err, level_, __VA_ARGS__)
#endif

#ifndef NVTRACE_LOG_INF_IF
#define NVTRACE_LOG_INF_IF(condition_, level_, ...) NVTRACE_LOG_IF(condition_, Inf, level_, __VA_ARGS__)
#endif
#ifndef NVTRACE_LOG_WRN_IF
#define NVTRACE_LOG_WRN_IF(condition_, level_, ...) NVTRACE_LOG_IF(condition_, Wrn, level_, __VA_ARGS__)
#endif
#ifndef NVTRACE_LOG_ERR_IF
#define NVTRACE_LOG_ERR_IF(condition_, level_, ...) NVTRACE_LOG_IF(condition_, Err, level_, __VA_ARGS__)
#endif

#ifndef NVTRACE_LOG_INF_RETURN_IF
#define NVTRACE_LOG_INF_RETURN_IF(condition_, result_, level_, ...) NVTRACE_LOG_DO_IF(condition_, return result_;, Inf, level_, __VA_ARGS__)
#endif
#ifndef NVTRACE_LOG_WRN_RETURN_IF
#define NVTRACE_LOG_WRN_RETURN_IF(condition_, result_, level_, ...) NVTRACE_LOG_DO_IF(condition_, return result_;, Wrn, level_, __VA_ARGS__)
#endif
#ifndef NVTRACE_LOG_ERR_RETURN_IF
#define NVTRACE_LOG_ERR_RETURN_IF(condition_, result_, level_, ...) NVTRACE_LOG_DO_IF(condition_, return result_;, Err, level_, __VA_ARGS__)
#endif

#ifndef NVTRACE_LOG_INF_BREAK_IF
#define NVTRACE_LOG_INF_BREAK_IF(condition_, level_, ...) NVTRACE_LOG_DO_IF(condition_, break;, Inf, level_, __VA_ARGS__)
#endif
#ifndef NVTRACE_LOG_WRN_BREAK_IF
#define NVTRACE_LOG_WRN_BREAK_IF(condition_, level_, ...) NVTRACE_LOG_DO_IF(condition_, break;, Wrn, level_, __VA_ARGS__)
#endif
#ifndef NVTRACE_LOG_ERR_BREAK_IF
#define NVTRACE_LOG_ERR_BREAK_IF(condition_, level_, ...) NVTRACE_LOG_DO_IF(condition_, break;, Err, level_, __VA_ARGS__)
#endif

#ifndef NVTRACE_LOG_INF_CONTINUE_IF
#define NVTRACE_LOG_INF_CONTINUE_IF(condition_, level_, ...) NVTRACE_LOG_DO_IF(condition_, continue;, Inf, level_, __VA_ARGS__)
#endif
#ifndef NVTRACE_LOG_WRN_CONTINUE_IF
#define NVTRACE_LOG_WRN_CONTINUE_IF(condition_, level_, ...) NVTRACE_LOG_DO_IF(condition_, continue;, Wrn, level_, __VA_ARGS__)
#endif
#ifndef NVTRACE_LOG_ERR_CONTINUE_IF
#define NVTRACE_LOG_ERR_CONTINUE_IF(condition_, level_, ...) NVTRACE_LOG_DO_IF(condition_, continue;, Err, level_, __VA_ARGS__)
#endif
