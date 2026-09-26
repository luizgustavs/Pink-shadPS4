// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <cstdlib>

#include "common/logging/log.h"

namespace Common {

// Acceptance-harness instrumentation, enabled only by SHADPS4_HARNESS_PROGRESS: progress lines (frames, VRAM,
// images every 10 s), one-time "Harness marker:" lines and applied game patches
inline bool HarnessEnabled() {
    static const bool enabled = std::getenv("SHADPS4_HARNESS_PROGRESS") != nullptr;
    return enabled;
}

} // namespace Common

// Logs a "Harness marker:" line the first time this call site is reached. The message is logged at warning
// level so it survives the reduced log filter used by speed runs
#define LOG_HARNESS_MARKER_ONCE(log_class, ...)                                                    \
    do {                                                                                           \
        static std::atomic_flag harness_marker_once_;                                              \
        if (::Common::HarnessEnabled() && !harness_marker_once_.test_and_set()) {                  \
            LOG_WARNING(log_class, "Harness marker: " __VA_ARGS__);                                \
        }                                                                                          \
    } while (0)
