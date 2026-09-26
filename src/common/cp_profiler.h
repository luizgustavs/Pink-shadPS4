// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

// Sample the GPU command processor on Windows with SHADPS4_CP_PROFILE=<start_s>:<duration_s>[:<hz>]
// After the delay, a sampler suspends the thread, unwinds it and resumes it at the requested rate (default
// 1000 Hz). It writes module+offset stacks to cp_profile.txt for tools/harness/cp_profile.py
namespace Common::CpProfiler {

/// Called by the command processor thread itself; starts the sampler if the variable is set
void RegisterCurrentThread();

/// Samples the first guest thread that submits a completed GNM frame when
/// SHADPS4_GAME_PROFILE=<start_s>:<duration_s>[:<hz>] is set
void RegisterGameRenderThread();

/// Samples the guest's initial Game:Main thread in the same M3 window
void RegisterGameMainThread();

} // namespace Common::CpProfiler
