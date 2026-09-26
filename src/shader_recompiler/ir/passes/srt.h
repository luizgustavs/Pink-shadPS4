// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <optional>
#include <boost/container/set.hpp>
#include <boost/container/small_vector.hpp>
#include "common/types.h"

namespace Serialization {
struct Archive;
}

namespace Shader {

struct SrtWalkerContext;
/// rdx and rcx are unused; r8 carries the context the generated guest loads check (SrtWalkerContext)
using PFN_SrtWalker = void PS4_SYSV_ABI (*)(const u32* /*user_data*/, u32* /*flat_dst*/,
                                            u64 /*unused*/, u64 /*unused*/,
                                            const SrtWalkerContext* /*context*/);
/// Copies a walker into the executable buffer, sharing identical code. Returns nullptr when the buffer is
/// full
PFN_SrtWalker RegisterWalkerCode(const u8* ptr, size_t size);
/// Bytes used in the walker code buffer and bytes saved by sharing identical walkers
size_t GetSrtCodeUsage();
size_t GetSrtCodeShared();

/// Per-game srt_walker_clean_reads: a faulting SRT walker load, decoded so that the fault handler can
/// complete it from guest memory instead of lifting the page protection
struct SrtLoad {
    u64 address; // Effective address of the memory operand
    u32 size;    // 8 for a pointer load into rdi, 4 for a dword load into r10d
    u32 length;  // Instruction length
};
std::optional<SrtLoad> DecodeSrtLoad(void* context);
/// Writes `value` to the load's destination register and skips the instruction
void CompleteSrtLoad(void* context, const SrtLoad& load, u64 value);

/// srt_walker_clean_reads without exceptions: every guest load of a generated walker first tests its page
/// in `clean_pages`, one bit per 4 KiB page below SrtCleanPageLimit. A page gets its bit when the fault
/// handler serves a clean read from it; later loads from it call `clean_load`, which serves bytes the GPU
/// never wrote without raising an access violation. When it returns false the walker runs the plain load,
/// so GPU-written bytes and unmapped descriptors reach the fault handlers as before. The generated code
/// reads the fields at fixed offsets
struct SrtWalkerContext {
    u64* clean_pages;                                                 // offset 0
    bool PS4_SYSV_ABI (*clean_load)(u64 address, u32 size, u64* out); // offset 8
};
constexpr u64 SrtCleanPageLimit = 1ULL << 40;
const SrtWalkerContext* GetSrtWalkerContext();
/// Sets the clean-load callback; the page manager owns the GPU-written state
void SetSrtCleanLoad(bool PS4_SYSV_ABI (*clean_load)(u64, u32, u64*));
/// Marks the pages of [address, address + size) so that later walker loads skip the fault
void MarkSrtCleanPages(u64 address, u32 size);

struct PersistentSrtInfo {
    // Special case when fetch shader uses step rates.
    struct SrtSharpReservation {
        u32 sgpr_base;
        u32 dword_offset;
        u32 num_dwords;
    };

    PFN_SrtWalker walker_func{};
    size_t walker_func_size{};
    u32 flattened_bufsize_dw = 16; // NumUserDataRegs

    void Serialize(Serialization::Archive& ar) const;
    bool Deserialize(Serialization::Archive& ar);
};

} // namespace Shader
