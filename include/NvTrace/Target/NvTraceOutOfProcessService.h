/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvTrace/Target/NvTraceClientLogger.h>
#include <NvTrace/Target/NvTraceScopeExitGuard.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <climits>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

namespace nv::trace {

    class OutOfProcessService
    {
    public:
        struct Options
        {
            const char* pServiceHostPath = nullptr;
            const char* pPollUntilServiceReadyPath = nullptr;
            uint32_t pollTimeoutSec = 60;
            const char* pNvTraceConfigYaml = nullptr;
            const char* pNvTraceConfigYamlPath = nullptr;
        };

    public:
        ~OutOfProcessService() noexcept
        {
            Reset();
        }

        OutOfProcessService() = default;
        OutOfProcessService(const OutOfProcessService&) = delete;
        OutOfProcessService& operator=(const OutOfProcessService&) = delete;

        void Reset() noexcept
        {
            StopServiceHost();
            CleanupMaterializedConfig();
            m_initialized = false;
        }

        bool Initialize(const Options& options)
        {
            NVTRACE_LOG_INF(50, "Initializing out-of-process service");
            auto resetGuard = ScopeExitGuard([this]() { Reset(); });

            m_serviceHostPath = ResolveToolPath(options.pServiceHostPath, "NVIDIA_NSIGHT_PERF_PATH", "nvidia-nsight-perf");
            m_pollUntilServiceReadyPath = ResolveToolPath(options.pPollUntilServiceReadyPath, "NVTRACE_POLL_UNTIL_SERVICE_READY_PATH", "nvtrace_poll_until_service_ready");

            constexpr uint32_t existingServiceProbeTimeoutSec = 1;
            if (WaitUntilServiceReady(existingServiceProbeTimeoutSec))
            {
                m_initialized = true;
                resetGuard.Dismiss();
                NVTRACE_LOG_INF(50, "Reusing existing out-of-process service");
                return true;
            }

            if (!PrepareServiceHostConfig(options))
            {
                return false;
            }

            if (!StartServiceHost())
            {
                return false;
            }

            m_startedServiceHost = true;

            if (!WaitUntilServiceReady(options.pollTimeoutSec))
            {
                return false;
            }

            m_initialized = true;
            resetGuard.Dismiss();
            NVTRACE_LOG_INF(50, "Out-of-process service is ready");
            return true;
        }

        bool Initialize()
        {
            Options options = {};
            return Initialize(options);
        }

        bool IsInitialized() const
        {
            return m_initialized;
        }

    private:
        static std::string ResolveToolPath(const char* pOptionPath, const char* pEnvironmentVariableName, const char* pToolFileName)
        {
            if (pOptionPath != nullptr && pOptionPath[0] != '\0')
            {
                return pOptionPath;
            }

            const char* pEnvironmentPath = std::getenv(pEnvironmentVariableName);
            if (pEnvironmentPath != nullptr && pEnvironmentPath[0] != '\0')
            {
                return pEnvironmentPath;
            }

            char executablePath[PATH_MAX] = {};
            const ssize_t executablePathLength = readlink("/proc/self/exe", executablePath, sizeof(executablePath) - 1);
            if (executablePathLength <= 0)
            {
                return std::string("./") + pToolFileName;
            }

            executablePath[static_cast<size_t>(executablePathLength)] = '\0';
            char* pLastSlash = strrchr(executablePath, '/');
            if (pLastSlash != nullptr)
            {
                *(pLastSlash + 1) = '\0';
            }
            else
            {
                executablePath[0] = '\0';
            }

            return std::string(executablePath) + pToolFileName;
        }

        void CleanupMaterializedConfig() noexcept
        {
            if (m_materializedConfigPath.empty())
            {
                m_serviceHostConfigPath.clear();
                return;
            }

            unlink(m_materializedConfigPath.c_str());
            m_materializedConfigPath.clear();
            m_serviceHostConfigPath.clear();
        }

        static bool WriteConfigFile(const std::string& path, const char* pNvTraceConfigYaml)
        {
            std::ofstream outFile(path, std::ios::binary);
            if (!outFile)
            {
                NVTRACE_LOG_ERR(50, "Failed to open NVIDIA Nsight Perf config file for write");
                return false;
            }

            outFile.write(pNvTraceConfigYaml, static_cast<std::streamsize>(strlen(pNvTraceConfigYaml)));
            if (!outFile)
            {
                NVTRACE_LOG_ERR(50, "Failed to write NVIDIA Nsight Perf config file");
                return false;
            }

            return true;
        }

        bool PrepareServiceHostConfig(const Options& options)
        {
            CleanupMaterializedConfig();

            if (options.pNvTraceConfigYamlPath != nullptr && options.pNvTraceConfigYamlPath[0] != '\0')
            {
                m_serviceHostConfigPath = options.pNvTraceConfigYamlPath;
                return true;
            }

            if (options.pNvTraceConfigYaml != nullptr && options.pNvTraceConfigYaml[0] != '\0')
            {
                char tempPath[] = "/tmp/nvidia-nsight-perf-XXXXXX";
                const int fd = mkstemp(tempPath);
                if (fd < 0)
                {
                    NVTRACE_LOG_ERR(50, "mkstemp failed for NVIDIA Nsight Perf config file");
                    return false;
                }
                close(fd);
                m_materializedConfigPath = tempPath;
                if (!WriteConfigFile(m_materializedConfigPath, options.pNvTraceConfigYaml))
                {
                    CleanupMaterializedConfig();
                    return false;
                }

                m_serviceHostConfigPath = m_materializedConfigPath;
            }

            return true;
        }

        void BuildServiceHostArgv(std::vector<std::string>& args) const
        {
            args.clear();
            args.push_back(m_serviceHostPath);

#if defined(__linux__)
            if (geteuid() == 0)
            {
                args.emplace_back("--global");
            }
#endif

            if (!m_serviceHostConfigPath.empty())
            {
                args.emplace_back("--config");
                args.push_back(m_serviceHostConfigPath);
            }
        }

        bool StartServiceHost()
        {
            std::vector<std::string> args;
            BuildServiceHostArgv(args);

            std::vector<char*> argv;
            argv.reserve(args.size() + 1);
            for (std::string& arg : args)
            {
                argv.push_back(arg.data());
            }
            argv.push_back(nullptr);

            const pid_t pid = fork();
            if (pid < 0)
            {
                NVTRACE_LOG_ERR(50, "fork() failed while starting nvidia-nsight-perf");
                return false;
            }
            if (pid == 0)
            {
                const char* pServiceHostPath = m_serviceHostPath.c_str();
                execv(pServiceHostPath, argv.data());
                _exit(1);
            }
            m_serviceHostPid = pid;
            return true;
        }

        bool WaitUntilServiceReady(uint32_t pollTimeoutSec) const
        {
            const char* pPollUntilServiceReadyPath = m_pollUntilServiceReadyPath.c_str();
            const std::string timeoutArg = std::to_string(pollTimeoutSec);

            const pid_t pid = fork();
            if (pid < 0)
            {
                NVTRACE_LOG_ERR(50, "fork() failed while starting nvtrace_poll_until_service_ready");
                return false;
            }
            if (pid == 0)
            {
                execl(
                    pPollUntilServiceReadyPath,
                    pPollUntilServiceReadyPath,
                    "--timeout-sec",
                    timeoutArg.c_str(),
                    static_cast<char*>(nullptr));
                _exit(1);
            }

            int status = 0;
            if (waitpid(pid, &status, 0) < 0)
            {
                NVTRACE_LOG_ERR(50, "waitpid() failed for nvtrace_poll_until_service_ready");
                return false;
            }

            if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
            {
                NVTRACE_LOG_ERR(50, "nvtrace_poll_until_service_ready failed");
                return false;
            }

            return true;
        }

        void StopServiceHost() noexcept
        {
            if (!m_startedServiceHost)
            {
                return;
            }

            if (m_serviceHostPid < 0)
            {
                return;
            }

            kill(m_serviceHostPid, SIGINT);
            int status = 0;
            waitpid(m_serviceHostPid, &status, 0);
            m_serviceHostPid = -1;
            m_startedServiceHost = false;
        }

    private:
        bool m_initialized = false;
        bool m_startedServiceHost = false;
        std::string m_serviceHostPath;
        std::string m_pollUntilServiceReadyPath;
        std::string m_serviceHostConfigPath;
        std::string m_materializedConfigPath;
        pid_t m_serviceHostPid = -1;
    };

} // namespace nv::trace
