// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/arch.h"
#include "common/signal_context.h"

#ifdef _WIN32
#include <windows.h>
#elif defined(__FreeBSD__)
#include <machine/npx.h>
#include <sys/ucontext.h>
#else
#include <sys/ucontext.h>
#endif

namespace Common {

void* GetRip(void* ctx) {
#if defined(_WIN32)
    return (void*)((EXCEPTION_POINTERS*)ctx)->ContextRecord->Rip;
#elif defined(__APPLE__) && defined(ARCH_X86_64)
    return (void*)((ucontext_t*)ctx)->uc_mcontext->__ss.__rip;
#elif defined(__APPLE__) && defined(ARCH_ARM64)
    return (void*)((ucontext_t*)ctx)->uc_mcontext->__ss.__pc;
#elif defined(__FreeBSD__)
    return (void*)((ucontext_t*)ctx)->uc_mcontext.mc_rip;
#elif defined(ARCH_X86_64)
    return (void*)((ucontext_t*)ctx)->uc_mcontext.gregs[REG_RIP];
#else
#error "Unsupported architecture"
#endif
}

void IncrementRip(void* ctx, u64 length) {
#if defined(_WIN32)
    ((EXCEPTION_POINTERS*)ctx)->ContextRecord->Rip += length;
#elif defined(__APPLE__) && defined(ARCH_X86_64)
    ((ucontext_t*)ctx)->uc_mcontext->__ss.__rip += length;
#elif defined(__APPLE__) && defined(ARCH_ARM64)
    ((ucontext_t*)ctx)->uc_mcontext->__ss.__pc += length;
#elif defined(__FreeBSD__)
    ((ucontext_t*)ctx)->uc_mcontext.mc_rip += length;
#elif defined(ARCH_X86_64)
    ((ucontext_t*)ctx)->uc_mcontext.gregs[REG_RIP] += length;
#else
#error "Unsupported architecture"
#endif
}

#ifdef ARCH_X86_64
u64 GetX64Gpr(void* ctx, X64Gpr reg) {
#if defined(_WIN32)
    const auto* const context = ((EXCEPTION_POINTERS*)ctx)->ContextRecord;
    return reg == X64Gpr::Rdi ? context->Rdi : reg == X64Gpr::R10 ? context->R10 : context->R11;
#elif defined(__APPLE__)
    const auto& state = ((ucontext_t*)ctx)->uc_mcontext->__ss;
    return reg == X64Gpr::Rdi ? state.__rdi : reg == X64Gpr::R10 ? state.__r10 : state.__r11;
#elif defined(__FreeBSD__)
    const auto& context = ((ucontext_t*)ctx)->uc_mcontext;
    return reg == X64Gpr::Rdi   ? context.mc_rdi
           : reg == X64Gpr::R10 ? context.mc_r10
                                : context.mc_r11;
#else
    const auto& gregs = ((ucontext_t*)ctx)->uc_mcontext.gregs;
    return static_cast<u64>(reg == X64Gpr::Rdi   ? gregs[REG_RDI]
                            : reg == X64Gpr::R10 ? gregs[REG_R10]
                                                 : gregs[REG_R11]);
#endif
}

void SetX64Gpr(void* ctx, X64Gpr reg, u64 value) {
#if defined(_WIN32)
    auto* const context = ((EXCEPTION_POINTERS*)ctx)->ContextRecord;
    switch (reg) {
    case X64Gpr::Rdi:
        context->Rdi = value;
        break;
    case X64Gpr::R10:
        context->R10 = value;
        break;
    default: // R11 is read only
        break;
    }
#elif defined(__APPLE__)
    auto& state = ((ucontext_t*)ctx)->uc_mcontext->__ss;
    switch (reg) {
    case X64Gpr::Rdi:
        state.__rdi = value;
        break;
    case X64Gpr::R10:
        state.__r10 = value;
        break;
    default: // R11 is read only
        break;
    }
#elif defined(__FreeBSD__)
    auto& context = ((ucontext_t*)ctx)->uc_mcontext;
    switch (reg) {
    case X64Gpr::Rdi:
        context.mc_rdi = value;
        break;
    case X64Gpr::R10:
        context.mc_r10 = value;
        break;
    default: // R11 is read only
        break;
    }
#else
    auto& gregs = ((ucontext_t*)ctx)->uc_mcontext.gregs;
    switch (reg) {
    case X64Gpr::Rdi:
        gregs[REG_RDI] = value;
        break;
    case X64Gpr::R10:
        gregs[REG_R10] = value;
        break;
    default: // R11 is read only
        break;
    }
#endif
}
#endif

bool IsWriteError(void* ctx) {
#if defined(_WIN32)
    return ((EXCEPTION_POINTERS*)ctx)->ExceptionRecord->ExceptionInformation[0] == 1;
#elif defined(__APPLE__) && defined(ARCH_X86_64)
    return ((ucontext_t*)ctx)->uc_mcontext->__es.__err & 0x2;
#elif defined(__APPLE__) && defined(ARCH_ARM64)
    return ((ucontext_t*)ctx)->uc_mcontext->__es.__esr & 0x40;
#elif defined(__FreeBSD__) && defined(ARCH_X86_64)
    return ((ucontext_t*)ctx)->uc_mcontext.mc_err & 0x2;
#elif defined(ARCH_X86_64)
    return ((ucontext_t*)ctx)->uc_mcontext.gregs[REG_ERR] & 0x2;
#else
#error "Unsupported architecture"
#endif
}

} // namespace Common
