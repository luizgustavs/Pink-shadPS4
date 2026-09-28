// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <unordered_map>
#include "common/assert.h"
#include "common/recursive_lock.h"

namespace Common::Detail {

struct RecursiveLockState {
    RecursiveLockType type;
    int count;
};

// Keep common recursive locks inline and spill the rest to the map
struct RecursiveLockEntry {
    void* mutex;
    RecursiveLockState state;
};

constexpr size_t InlineRecursiveLocks = 8;

struct RecursiveLocks {
    std::array<RecursiveLockEntry, InlineRecursiveLocks> entries{};
    std::unordered_map<void*, RecursiveLockState> overflow;

    RecursiveLockState* Find(void* mutex) {
        for (auto& entry : entries) {
            if (entry.mutex == mutex) {
                return &entry.state;
            }
        }
        if (overflow.empty()) {
            return nullptr;
        }
        const auto it = overflow.find(mutex);
        return it == overflow.end() ? nullptr : &it->second;
    }

    void Add(void* mutex, RecursiveLockType type) {
        for (auto& entry : entries) {
            if (!entry.mutex) {
                entry = {mutex, {type, 1}};
                return;
            }
        }
        overflow.emplace(mutex, RecursiveLockState{type, 1});
    }

    void Remove(void* mutex) {
        for (auto& entry : entries) {
            if (entry.mutex == mutex) {
                entry = {};
                return;
            }
        }
        overflow.erase(mutex);
    }
};

thread_local RecursiveLocks g_recursive_locks;

bool IncrementRecursiveLock(void* mutex, RecursiveLockType type) {
    if (auto* state = g_recursive_locks.Find(mutex)) {
        ASSERT(state->type == type);
        ++state->count;
        return false;
    }
    g_recursive_locks.Add(mutex, type);
    return true;
}

bool DecrementRecursiveLock(void* mutex, RecursiveLockType type) {
    auto* state = g_recursive_locks.Find(mutex);
    ASSERT(state && state->type == type && state->count > 0);
    if (--state->count == 0) {
        g_recursive_locks.Remove(mutex);
        return true;
    }
    return false;
}

} // namespace Common::Detail
