// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <string>
#include <vector>

#include "common/guest_write_journal.h"
#include "common/logging/log.h"
#include "core/debug_state.h"
#include "core/emulator_settings.h"

namespace Vulkan::DrawTrace {

/// Taken during static initialization, so trace start times count from process launch
inline const std::chrono::steady_clock::time_point process_start =
    std::chrono::steady_clock::now();

/// Diagnostic trace of every draw and dispatch in selected guest frames, enabled only by
/// SHADPS4_DRAW_TRACE=<sec>[,<sec>...][:<frames>]. For each <sec> since launch, the <frames> (default 1)
/// submit frames that start after it are logged at warning level with a "DrawTrace" prefix
/// (tools/harness/draw_trace.py of the SotC fork reads them). Only called from the GPU command processor
/// thread
class Tracer {
public:
    /// Updates the trace window. Call once per draw or dispatch, before logging it
    bool Active() {
        if (!enabled) {
            return false;
        }
        // sceGnmSubmitDone waits for the GPU to go idle before counting a frame, so commands processed while
        // the counter reads N all belong to submit frame N
        const s32 frame = DebugState.GetGnmFrameNum();
        if (first_frame >= 0) {
            if (frame < first_frame) {
                return false;
            }
            if (frame <= last_frame) {
                if (frame != current_frame) {
                    current_frame = frame;
                    index = 0;
                    LOG_WARNING(Render_Vulkan, "DrawTrace frame {} begin", frame);
                }
                return true;
            }
            LOG_WARNING(Render_Vulkan, "DrawTrace window {}..{} end", first_frame, last_frame);
            // With SHADPS4_GUEST_WRITE_JOURNAL set, keep the emulator's write history up to here
            Common::GuestWriteJournal::DumpOnCrash(0, 0);
            first_frame = -1;
        }
        if (next >= starts.size()) {
            return false;
        }
        const double elapsed =
            std::chrono::duration<double>(Clock::now() - process_start).count();
        if (elapsed < starts[next]) {
            return false;
        }
        first_frame = frame + 1;
        last_frame = frame + frames;
        ++next;
        LOG_WARNING(Render_Vulkan, "DrawTrace armed at {:.1f} s: frames {}..{} readbacks_mode={}",
                    elapsed, first_frame, last_frame, EmulatorSettings.GetReadbacksMode());
        return false;
    }

    s32 Frame() const noexcept {
        return current_frame;
    }

    u32 NextIndex() noexcept {
        return index++;
    }

private:
    using Clock = std::chrono::steady_clock;

    std::vector<double> starts;
    s32 frames = 1;
    std::size_t next = 0;
    s32 first_frame = -1;
    s32 last_frame = -1;
    s32 current_frame = -1;
    u32 index = 0;
    bool enabled = Parse(); // Declared last: Parse() fills the members above

    bool Parse() {
        const char* value = std::getenv("SHADPS4_DRAW_TRACE");
        if (!value || !*value) {
            return false;
        }
        std::string spec{value};
        if (const auto colon = spec.find(':'); colon != std::string::npos) {
            frames = std::max(1, std::atoi(spec.c_str() + colon + 1));
            spec.resize(colon);
        }
        std::size_t pos = 0;
        while (pos <= spec.size()) {
            const auto comma = spec.find(',', pos);
            const auto item =
                spec.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
            if (!item.empty()) {
                starts.push_back(std::atof(item.c_str()));
            }
            if (comma == std::string::npos) {
                break;
            }
            pos = comma + 1;
        }
        return !starts.empty();
    }
};

inline Tracer& Instance() {
    static Tracer tracer;
    return tracer;
}

} // namespace Vulkan::DrawTrace
