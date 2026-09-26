// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/arch.h"
#include "common/types.h"

namespace Common {

void* GetXmmPointer(void* ctx, u8 index);

void* GetRip(void* ctx);

void IncrementRip(void* ctx, u64 length);

#ifdef ARCH_X86_64
enum class X64Gpr : u8 {
    Rdi,
    R10,
    R11, // Read only (GetX64Gpr)
};

u64 GetX64Gpr(void* ctx, X64Gpr reg);
void SetX64Gpr(void* ctx, X64Gpr reg, u64 value);
#endif

bool IsWriteError(void* ctx);

} // namespace Common
