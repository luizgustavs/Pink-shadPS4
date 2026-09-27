// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

// Sample the GPU command processor on Windows with SHADPS4_CP_PROFILE=<start_s>:<duration_s>[:<hz>]
// After the delay, a sampler suspends the thread, unwinds it and resumes it at the requested rate (default
// 1000 Hz). It writes module+offset stacks to cp_profile.txt for tools/harness/cp_profile.py
namespace Common::CpProfiler {

/// Called by the command processor thread itself; starts the sampler if the variable is set
void RegisterCurrentThread();

/// M3b: samples every guest thread (this one included) with one sampler when
/// SHADPS4_GUEST_PROFILE=<start_s>:<duration_s>[:<hz>] is set (default 250 Hz). One guest_profile_<n>_<name>.txt
/// per thread plus guest_profile_index.txt
void RegisterGuestThread(const char* name);

/// Samples the first guest thread that submits a completed GNM frame when
/// SHADPS4_GAME_PROFILE=<start_s>:<duration_s>[:<hz>] is set
void RegisterGameRenderThread();

/// Samples the guest's initial Game:Main thread in the same M3 window
void RegisterGameMainThread();

} // namespace Common::CpProfiler
