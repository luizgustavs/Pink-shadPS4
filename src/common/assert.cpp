// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <string>
#include <fmt/format.h>
#include "common/arch.h"
#include "common/assert.h"
#include "common/logging/events.h"
#include "common/thread.h"
#include "core/signals.h"
#include "emulator.h"

#if defined(ARCH_X86_64)
#define Crash() __asm__ __volatile__("int $3")
#elif defined(ARCH_ARM64)
#define Crash() __asm__ __volatile__("brk 0")
#else
#error "Missing Crash() implementation for target CPU architecture."
#endif

// Turns the Critical message the assertion just logged into a Crash event (and a Broken Shader
// event when the thread was compiling a shader)
static void ReportAssertionCrash() {
    static std::atomic<bool> reported{false};
    if (!Common::Log::EventsEnabled() || reported.exchange(true)) {
        return;
    }
    const auto last = Common::Log::LastCriticalMessage();
    if (auto* scope = Common::Log::CurrentErrorScope()) {
        Common::Log::Detail::ErrorScopeCallGuard guard;
        scope->OnFatal(last);
    }
    // "file:line func: Assertion Failed!\n<message>"
    std::string_view location = last;
    std::string_view kind = "Assertion failed";
    std::string message;
    for (const std::string_view marker : {": Assertion Failed!", ": Unreachable code!"}) {
        if (const auto pos = location.find(marker); pos != std::string_view::npos) {
            kind =
                marker == ": Unreachable code!" ? "Unreachable code reached" : "Assertion failed";
            auto rest = location.substr(pos + marker.size());
            while (!rest.empty() && (rest.front() == '\n' || rest.front() == '\r')) {
                rest.remove_prefix(1);
            }
            message = std::string{rest};
            location = location.substr(0, pos);
            break;
        }
    }
    std::ranges::replace(message, '\n', ' ');
    const auto headline =
        message.empty() ? std::string{kind} : fmt::format("{}: {}", kind, message);
    Common::Log::Event(Common::Log::EventKind::Crash, headline,
                       {fmt::format("at {}", location),
                        fmt::format("thread: {}", Common::GetCurrentThreadName()),
                        fmt::format("log: {}", Common::Log::Detail::LogFilePath())});
}

void assert_fail_impl() {
    ReportAssertionCrash();
    Core::Signals::Instance()->RemoveHandlers();
    Common::Singleton<Core::Emulator>::Instance()->Shutdown();
    Crash();
}

[[noreturn]] void unreachable_impl() {
    assert_fail_impl();
    throw std::runtime_error("Unreachable code");
}

void assert_fail_debug_msg(const char* msg) {
    LOG_CRITICAL(Debug, "Assertion Failed!\n{}", msg);
    assert_fail_impl();
}
