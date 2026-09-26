// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>

#include "common/types.h"

namespace VideoCore {

/// A fault reported by a shader is only worth caching when it points at mapped guest memory
constexpr bool IsFaultAddressCacheable(VAddr fault_address, bool is_mapped) {
    return fault_address != 0 && is_mapped;
}

/// The fault parser keeps counting past the end of an area; entry 0 holds the count, so at most area_entries
/// - 1 addresses follow it
constexpr u32 ClampFaultCount(u64 reported_count, u32 area_entries) {
    const u64 max_faults = area_entries > 0 ? area_entries - 1 : 0;
    return static_cast<u32>(std::min(reported_count, max_faults));
}

} // namespace VideoCore
