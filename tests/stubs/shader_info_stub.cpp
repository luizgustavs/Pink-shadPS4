// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// info.cpp reads sharp tables through the memory manager; the tests build IR without guest memory and with
// dynamic_tsharp_array_size off. The tests run no SRT walker (flatten_extended_userdata_pass.cpp is not
// built), so the walker context is never used

#include "shader_recompiler/info.h"

namespace Shader {

void Info::RefreshSharpTables() {}

u32 DynamicTsharpArraySize() {
    return 0;
}

const SrtWalkerContext* GetSrtWalkerContext() {
    return nullptr;
}

} // namespace Shader
