// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <cstring>
#include <fmt/format.h>
#include "common/arch.h"
#include "common/assert.h"
#include "common/decoder.h"
#include "common/guest_write_journal.h"
#include "common/logging/events.h"
#include "common/memory_patcher.h"
#include "common/signal_context.h"
#include "common/thread.h"
#include "core/cpu_patches.h" // Windows static guest red-zone protection
#include "core/libraries/kernel/kernel.h"
#include "core/libraries/kernel/threads/exception.h"
#include "core/signals.h"
#include "emulator.h"

#ifdef _WIN32
#include <csignal>
#include <windows.h>
static constexpr DWORD MS_VC_EXCEPTION = 0x406D1388;
static constexpr DWORD MSVC_CPP_EXCEPTION = 0xE06D7363;
#else
#include <csignal>
#include <pthread.h>
#ifdef ARCH_X86_64
#include <Zydis/Formatter.h>
#endif
#endif

namespace Core {

#if defined(_WIN32)

// Crash handlers write raw reports through kernel32 because regular logging can overflow a 16 KiB guest fiber
// stack before it prints anything. Static buffers and hex output keep this path small
// Reports go to shadps4_crash_raw.txt and shadps4_crash_stack.bin in the working directory

// Append crash reports, truncating on the first write of each run so handled guest faults do not grow old logs
// Callers are serialized by crash_file_busy
static HANDLE OpenCrashFile(const char* name, volatile LONG& opened) noexcept {
    if (opened == 0) {
        const HANDLE file = CreateFileA(name, GENERIC_WRITE, FILE_SHARE_READ, nullptr,
                                        CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file != INVALID_HANDLE_VALUE) {
            opened = 1;
        }
        return file;
    }
    return CreateFileA(name, FILE_APPEND_DATA, FILE_SHARE_READ, nullptr, OPEN_ALWAYS,
                       FILE_ATTRIBUTE_NORMAL, nullptr);
}

static bool CopyCrashBytes(void* dst, u64 src, u64 size) noexcept {
    // Faulting here would re-enter the vectored handler, so only committed readable pages
    MEMORY_BASIC_INFORMATION info{};
    if (VirtualQuery(reinterpret_cast<const void*>(src), &info, sizeof(info)) == 0 ||
        info.State != MEM_COMMIT || (info.Protect & (PAGE_GUARD | PAGE_NOACCESS)) != 0 ||
        (info.Protect & (PAGE_READONLY | PAGE_READWRITE | PAGE_WRITECOPY | PAGE_EXECUTE_READ |
                         PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY)) == 0) {
        return false;
    }
    __try {
        std::memcpy(dst, reinterpret_cast<const void*>(src), size);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return false;
    }
    return true;
}

// Save a window around rsp for tools/harness/crash_stack.py; guest assert messages may sit there while the
// text report only captures eboot pointers
static void WriteCrashStackWindow(const CONTEXT* ctx) noexcept {
    static constexpr u64 Below = 0x1000;
    static constexpr u64 Above = 0x3000;
    struct Header {
        char magic[8];
        u64 rip;
        u64 rsp;
        u64 window_start;
        u64 window_size;
        u64 eboot_base;
        u64 eboot_size;
        u64 gpr[16]; // rax rcx rdx rbx rsp rbp rsi rdi r8..r15
    };
    static Header header;
    static u8 window[Below + Above];
    static u64 last_rip = 0;
    static u64 last_rsp = 0;
    if (ctx->Rip == last_rip && ctx->Rsp == last_rsp) {
        return;
    }
    last_rip = ctx->Rip;
    last_rsp = ctx->Rsp;

    const u64 start = (ctx->Rsp & ~7ull) - Below;
    for (u64 offset = 0; offset < sizeof(window);) {
        const u64 address = start + offset;
        const u64 chunk = std::min<u64>(sizeof(window) - offset, 0x1000 - (address & 0xFFF));
        if (!CopyCrashBytes(window + offset, address, chunk)) {
            std::memset(window + offset, 0, chunk);
        }
        offset += chunk;
    }
    std::memcpy(header.magic, "SHSTACK1", 8);
    header.rip = ctx->Rip;
    header.rsp = ctx->Rsp;
    header.window_start = start;
    header.window_size = sizeof(window);
    header.eboot_base = MemoryPatcher::g_eboot_address;
    header.eboot_size = MemoryPatcher::g_eboot_image_size;
    const u64 gpr[16] = {ctx->Rax, ctx->Rcx, ctx->Rdx, ctx->Rbx, ctx->Rsp, ctx->Rbp,
                         ctx->Rsi, ctx->Rdi, ctx->R8,  ctx->R9,  ctx->R10, ctx->R11,
                         ctx->R12, ctx->R13, ctx->R14, ctx->R15};
    std::memcpy(header.gpr, gpr, sizeof(gpr));

    static volatile LONG opened = 0;
    HANDLE file = OpenCrashFile("shadps4_crash_stack.bin", opened);
    if (file != INVALID_HANDLE_VALUE) {
        DWORD written = 0;
        WriteFile(file, &header, sizeof(header), &written, nullptr);
        WriteFile(file, window, sizeof(window), &written, nullptr);
        CloseHandle(file);
    }
}

// While writing a report, let its probing faults reach their __except blocks instead of starting another report
static thread_local bool writing_crash_report = false;

static void WriteCrashReportRaw(EXCEPTION_POINTERS* pExp) noexcept {
    static volatile LONG crash_file_busy = 0;
    if (InterlockedExchange(&crash_file_busy, 1) != 0) {
        return;
    }
    writing_crash_report = true;
    static char buf[1024];
    u32 len = 0;
    const auto put = [&](const char* s) {
        while (*s != '\0' && len < sizeof(buf) - 1) {
            buf[len++] = *s++;
        }
    };
    const auto put_hex = [&](u64 v) {
        char tmp[17];
        int n = 0;
        if (v == 0) {
            tmp[n++] = '0';
        }
        while (v != 0 && n < 16) {
            const u32 d = v & 0xF;
            tmp[n++] = d < 10 ? char('0' + d) : char('a' + d - 10);
            v >>= 4;
        }
        if (len + n + 3 < sizeof(buf)) {
            buf[len++] = '0';
            buf[len++] = 'x';
            while (n > 0) {
                buf[len++] = tmp[--n];
            }
        }
    };
    const auto put_reg = [&](const char* name, u64 value) {
        put(name);
        put_hex(value);
    };
    DWORD code = 0;
    const CONTEXT* ctx = pExp != nullptr ? pExp->ContextRecord : nullptr;
    if (pExp != nullptr && pExp->ExceptionRecord != nullptr) {
        code = pExp->ExceptionRecord->ExceptionCode;
    }
    put("shadPS4 crash: code=");
    put_hex(code);
    if (ctx != nullptr) {
        put_reg(" rip=", ctx->Rip);
        put_reg(" rsp=", ctx->Rsp);
        put_reg(" rax=", ctx->Rax);
        put_reg(" rbx=", ctx->Rbx);
        put_reg(" rcx=", ctx->Rcx);
        put_reg(" rdx=", ctx->Rdx);
        put_reg(" rsi=", ctx->Rsi);
        put_reg(" rdi=", ctx->Rdi);
        put_reg(" r8=", ctx->R8);
        put_reg(" r9=", ctx->R9);
        put_reg(" r10=", ctx->R10);
        put_reg(" r11=", ctx->R11);
        put_reg(" r12=", ctx->R12);
        put_reg(" r13=", ctx->R13);
        put_reg(" r14=", ctx->R14);
        put_reg(" r15=", ctx->R15);
        put_reg(" rbp=", ctx->Rbp);
    }
    if (pExp != nullptr && pExp->ExceptionRecord != nullptr &&
        code == EXCEPTION_ACCESS_VIOLATION && pExp->ExceptionRecord->NumberParameters >= 2) {
        put(" av_type=");
        put(pExp->ExceptionRecord->ExceptionInformation[0] == 0 ? "read" : "write");
        put_reg(" av_addr=", pExp->ExceptionRecord->ExceptionInformation[1]);
    }
    if (ctx != nullptr && MemoryPatcher::g_eboot_address != 0 &&
        ctx->Rip >= MemoryPatcher::g_eboot_address &&
        ctx->Rip < MemoryPatcher::g_eboot_address + MemoryPatcher::g_eboot_image_size) {
        put_reg(" eboot+", ctx->Rip - MemoryPatcher::g_eboot_address);
    }
    // Host module containing rip (for symbolizing emulator-side crashes)
    const u64 exe_base = reinterpret_cast<u64>(GetModuleHandleA(nullptr));
    if (ctx != nullptr) {
        HMODULE module = nullptr;
        if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                   GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                               reinterpret_cast<LPCSTR>(ctx->Rip), &module) &&
            module != nullptr) {
            static char name[MAX_PATH];
            const DWORD n = GetModuleFileNameA(module, name, MAX_PATH);
            const char* short_name = name;
            for (DWORD i = 0; i < n; ++i) {
                if (name[i] == '\\' || name[i] == '/') {
                    short_name = name + i + 1;
                }
            }
            put(" mod=");
            put(short_name);
            put_reg("+", ctx->Rip - reinterpret_cast<u64>(module));
        }
    }
    // Return addresses into eboot.bin / the emulator image found on the faulting stack, innermost first
    if (ctx != nullptr) {
        put(" stack:");
        u32 hits = 0;
        const u64 base = MemoryPatcher::g_eboot_address;
        const u64 end = base + MemoryPatcher::g_eboot_image_size;
        const u64 exe_end = exe_base + (256ull << 20);
        for (u64 addr = ctx->Rsp & ~7ull; hits < 12 && addr < (ctx->Rsp & ~7ull) + 16384;
             addr += 8) {
            u64 value = 0;
            if (!CopyCrashBytes(&value, addr, sizeof(value))) {
                break;
            }
            if (base != 0 && value >= base && value < end) {
                put_reg(" eboot+", value - base);
                ++hits;
            } else if (value >= exe_base && value < exe_end) {
                put_reg(" exe+", value - exe_base);
                ++hits;
            }
        }
    }
    buf[len++] = '\n';
    static volatile LONG opened = 0;
    HANDLE file = OpenCrashFile("shadps4_crash_raw.txt", opened);
    if (file != INVALID_HANDLE_VALUE) {
        DWORD written = 0;
        WriteFile(file, buf, len, &written, nullptr);
        CloseHandle(file);
    }
    if (ctx != nullptr) {
        WriteCrashStackWindow(ctx);
    }
    writing_crash_report = false;
    InterlockedExchange(&crash_file_busy, 0);
}

// "eboot.bin+0x1234 (game code)", "shadps4.exe+0x1234 (emulator code)", ...
static std::string DescribeCodeAddress(u64 address) {
    const u64 eboot = MemoryPatcher::g_eboot_address;
    if (eboot != 0 && address >= eboot && address < eboot + MemoryPatcher::g_eboot_image_size) {
        return fmt::format("eboot.bin+{:#x} (game code)", address - eboot);
    }
    HMODULE module = nullptr;
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                               GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           reinterpret_cast<LPCSTR>(address), &module) &&
        module != nullptr) {
        char name[MAX_PATH]{};
        GetModuleFileNameA(module, name, MAX_PATH);
        const std::string_view path{name};
        const auto file = path.substr(path.find_last_of("\\/") + 1);
        return fmt::format("{}+{:#x}{}", file, address - reinterpret_cast<u64>(module),
                           module == GetModuleHandleA(nullptr) ? " (emulator code)" : "");
    }
    return fmt::format("{:#x} (game module or generated code)", address);
}

// Console Crash event for an exception nothing handled
static void ReportCrashEvent(EXCEPTION_POINTERS* pExp) {
    if (!Common::Log::EventsEnabled()) {
        return;
    }
    const auto* record = pExp != nullptr ? pExp->ExceptionRecord : nullptr;
    const auto* ctx = pExp != nullptr ? pExp->ContextRecord : nullptr;
    const DWORD code = record != nullptr ? record->ExceptionCode : 0;
    std::string what;
    switch (code) {
    case EXCEPTION_ACCESS_VIOLATION:
        if (record->NumberParameters >= 2) {
            const auto type = record->ExceptionInformation[0];
            what = fmt::format("Access violation {} {:#x}",
                               type == 0   ? "reading"
                               : type == 8 ? "executing"
                                           : "writing",
                               record->ExceptionInformation[1]);
        } else {
            what = "Access violation";
        }
        break;
    case EXCEPTION_ILLEGAL_INSTRUCTION:
        what = "Illegal instruction";
        break;
    case EXCEPTION_PRIV_INSTRUCTION:
        what = "Privileged instruction";
        break;
    case EXCEPTION_STACK_OVERFLOW:
        what = "Stack overflow";
        break;
    case EXCEPTION_INT_DIVIDE_BY_ZERO:
        what = "Integer division by zero";
        break;
    case EXCEPTION_IN_PAGE_ERROR:
        what = "In-page I/O error";
        break;
    case EXCEPTION_BREAKPOINT:
        what = "Breakpoint";
        break;
    default:
        what = fmt::format("Exception {:#x}", code);
        break;
    }
    const u64 rip = ctx != nullptr      ? ctx->Rip
                    : record != nullptr ? reinterpret_cast<u64>(record->ExceptionAddress)
                                        : 0;
    if (auto* scope = Common::Log::CurrentErrorScope()) {
        Common::Log::Detail::ErrorScopeCallGuard guard;
        scope->OnFatal(what);
    }
    char cwd[MAX_PATH]{};
    GetCurrentDirectoryA(MAX_PATH, cwd);
    Common::Log::Event(Common::Log::EventKind::Crash,
                       fmt::format("{} at {}", what, DescribeCodeAddress(rip)),
                       {fmt::format("thread: {}", Common::GetCurrentThreadName()),
                        fmt::format("registers and stack: {}\\shadps4_crash_raw.txt", cwd),
                        fmt::format("log: {}", Common::Log::Detail::LogFilePath())});
}

static LPTOP_LEVEL_EXCEPTION_FILTER previous_top_level_filter = nullptr;

// Last-resort filter for exceptions that leave the vectored handler unhandled, e.g. a host C++ exception
// nothing catches. Chains to the filter it replaced (the C++ runtime's terminates)
static LONG WINAPI TopLevelExceptionFilter(EXCEPTION_POINTERS* pExp) noexcept {
    WriteCrashReportRaw(pExp);
    if (pExp != nullptr && pExp->ExceptionRecord != nullptr &&
        pExp->ExceptionRecord->ExceptionCode == MSVC_CPP_EXCEPTION) {
        // Checked here so the "all" console mode does not even format the event
        if (Common::Log::EventsEnabled()) {
            Common::Log::Event(Common::Log::EventKind::Crash,
                               fmt::format("Unhandled C++ exception thrown at {}",
                                           DescribeCodeAddress(reinterpret_cast<u64>(
                                               pExp->ExceptionRecord->ExceptionAddress))),
                               {fmt::format("thread: {}", Common::GetCurrentThreadName())});
        }
        Common::Log::Detail::FileOnlyCriticals file_only;
        LOG_CRITICAL(Debug, "Unhandled C++ exception at {}",
                     pExp->ExceptionRecord->ExceptionAddress);
    }
    return previous_top_level_filter != nullptr ? previous_top_level_filter(pExp)
                                                : EXCEPTION_CONTINUE_SEARCH;
}

// abort() on a host thread. A C++ exception escaping a std::thread calls std::terminate during the search
// phase (the thread entry is noexcept), so the filter above never sees it; the default terminate handler then
// calls abort(), which raises SIGABRT. The UCRT keeps the SIGABRT action process-wide, unlike the per-thread
// terminate handler. Returning lets abort() end the process
static void AbortSignalHandler(int) {
    static constexpr DWORD STATUS_FATAL_APP_EXIT_CODE = 0x40000015;
    CONTEXT context{};
    RtlCaptureContext(&context);
    EXCEPTION_RECORD record{};
    record.ExceptionCode = STATUS_FATAL_APP_EXIT_CODE;
    record.ExceptionAddress = reinterpret_cast<PVOID>(context.Rip);
    EXCEPTION_POINTERS pointers{&record, &context};
    WriteCrashReportRaw(&pointers);
    if (Common::Log::EventsEnabled()) {
        Common::Log::Event(
            Common::Log::EventKind::Crash,
            "abort() on a host thread (std::terminate, e.g. an uncaught C++ exception)",
            {fmt::format("thread: {}", Common::GetCurrentThreadName())});
    }
    Common::Log::Detail::FileOnlyCriticals file_only;
    LOG_CRITICAL(Debug, "abort() on a host thread (std::terminate, e.g. an uncaught C++ exception)");
    Common::Singleton<Core::Emulator>::Instance()->Shutdown();
}

// Keep the guest context and crash reporting in a separate function after memory tracking has had
// its chance to handle the fault
// The context alone takes about 1.3 KB, and a game fiber may have only 16 KiB of stack space
// Leaving that work on every page fault frame could overflow the fiber stack while Protect was
// still running
// The earlier 2.1 KB handler frame was enough to reach the neighboring fiber during that path
static SHAD_NO_INLINE LONG HandleUnclaimedException(EXCEPTION_POINTERS* pExp, DWORD code, PVOID address,
                                                    s32 signo, s32 si_code,
                                                    bool static_protection_exception) {
    using namespace Libraries::Kernel;
    const bool use_static_windows_guest_red_zone_protection =
        WindowsGuestRedZoneProtection::IsStaticPatchingEnabled();

    // Record faults that memory tracking did not handle before the game can consume them through
    // its own signal handler
    // Limit the reports because some games use these signals during normal execution and should not
    // flood the log
    static std::atomic<u32> pre_dispatch_reports{0};
    if ((code == EXCEPTION_ACCESS_VIOLATION || code == EXCEPTION_ILLEGAL_INSTRUCTION) &&
        pre_dispatch_reports.fetch_add(1, std::memory_order_relaxed) < 32) {
        WriteCrashReportRaw(pExp);
    }

    if (signo != 0) {
        Ucontext guest_context{pExp->ContextRecord};
        Siginfo guest_info{
            ._si_signo = signo,
            ._si_errno = 0,
            ._si_code = si_code,
            ._si_addr = (void*)guest_context.uc_mcontext.mc_rip,
        };
        if (g_curthread && g_curthread->DispatchSignal(signo, &guest_info, &guest_context)) {
            return EXCEPTION_CONTINUE_EXECUTION;
        }
    }

    const bool report_unhandled =
        use_static_windows_guest_red_zone_protection ? static_protection_exception : true;
    if (report_unhandled) {
        WriteCrashReportRaw(pExp);
        if (pExp->ExceptionRecord != nullptr) {
            const auto* record = pExp->ExceptionRecord;
            Common::GuestWriteJournal::DumpOnCrash(
                reinterpret_cast<u64>(record->ExceptionAddress),
                record->NumberParameters > 1 ? record->ExceptionInformation[1] : 0);
        }
        ReportCrashEvent(pExp);
        {
            Common::Log::Detail::FileOnlyCriticals file_only;
            LOG_CRITICAL(Debug, "Unhandled Exception code {:#x} at {}", code, address);
        }
        Common::Singleton<Core::Emulator>::Instance()->Shutdown();
    }

    return EXCEPTION_CONTINUE_SEARCH;
}

static LONG WINAPI SignalHandler(EXCEPTION_POINTERS* pExp) noexcept {
    using namespace Libraries::Kernel;
    if (writing_crash_report) {
        // A probing read of the crash report itself: its __except block handles it
        return EXCEPTION_CONTINUE_SEARCH;
    }
    const auto* signals = Signals::Instance();

    // One nested fault is legitimate (a guest signal handler, which runs inside this handler, touching a
    // tracked page); an unhandled one is reported below. A fault while handling a nested fault means the
    // fault handling itself is failing: persist its context before anything else. Bounded, because a guest
    // handler that longjmps out never unwinds the guard
    static thread_local int handler_depth = 0;
    struct DepthGuard {
        int& depth;
        ~DepthGuard() {
            --depth;
        }
    } depth_guard{handler_depth};
    static std::atomic<u32> nested_reports{0};
    if (++handler_depth > 2 && nested_reports.fetch_add(1, std::memory_order_relaxed) < 8) {
        WriteCrashReportRaw(pExp);
    }

    const bool use_static_windows_guest_red_zone_protection =
        WindowsGuestRedZoneProtection::IsStaticPatchingEnabled();
    DWORD code = 0;
    PVOID address = nullptr;

    if (pExp != nullptr && pExp->ExceptionRecord != nullptr) {
        code = pExp->ExceptionRecord->ExceptionCode;
        address = pExp->ExceptionRecord->ExceptionAddress;
    }

    s32 signo = 0;
    s32 si_code = POSIX_SI_NOINFO;

    bool handled = false;
    bool static_protection_exception = false; // Windows static guest red-zone protection
    switch (code) {
    case EXCEPTION_ACCESS_VIOLATION:
        signo = POSIX_SIGSEGV;
        si_code = POSIX_SEGV_MAPERR;
        static_protection_exception = true; // Windows static guest red-zone protection
        handled = signals->DispatchAccessViolation(
            pExp, reinterpret_cast<void*>(pExp->ExceptionRecord->ExceptionInformation[1]));
        break;
    case EXCEPTION_ILLEGAL_INSTRUCTION:
        signo = POSIX_SIGILL;
        si_code = POSIX_ILL_ILLOPC;
        static_protection_exception = true; // Windows static guest red-zone protection
        handled = signals->DispatchIllegalInstruction(pExp);
        break;
    case EXCEPTION_PRIV_INSTRUCTION: // Windows static guest red-zone protection
        if (use_static_windows_guest_red_zone_protection) {
            static_protection_exception = true;
            handled = signals->DispatchIllegalInstruction(pExp);
        }
        break;
    case EXCEPTION_IN_PAGE_ERROR:
        signo = POSIX_SIGBUS;
        si_code = POSIX_BUS_ADRALN;
        break;
    case EXCEPTION_INT_DIVIDE_BY_ZERO:
        signo = POSIX_SIGFPE;
        si_code = POSIX_FPE_INTDIV;
        break;
    case EXCEPTION_INT_OVERFLOW:
        signo = POSIX_SIGFPE;
        si_code = POSIX_FPE_INTOVF;
        break;
    case EXCEPTION_FLT_DIVIDE_BY_ZERO:
        signo = POSIX_SIGFPE;
        si_code = POSIX_FPE_FLTDIV;
        break;
    case EXCEPTION_FLT_INVALID_OPERATION:
        signo = POSIX_SIGFPE;
        si_code = POSIX_FPE_FLTINV;
        break;
    case EXCEPTION_FLT_OVERFLOW:
        signo = POSIX_SIGFPE;
        si_code = POSIX_FPE_FLTOVF;
        break;
    case EXCEPTION_FLT_UNDERFLOW:
        signo = POSIX_SIGFPE;
        si_code = POSIX_FPE_FLTUND;
        break;
    case EXCEPTION_FLT_DENORMAL_OPERAND:
        signo = POSIX_SIGFPE;
        si_code = POSIX_FPE_FLTSUB; // This mapping for denormal floating point operands is still uncertain and needs confirmation
// against the guest signal behavior
        break;
    case EXCEPTION_FLT_INEXACT_RESULT:
        signo = POSIX_SIGFPE;
        si_code = POSIX_FPE_FLTRES;
        break;
    case EXCEPTION_FLT_STACK_CHECK:
        signo = POSIX_SIGILL;
        si_code = POSIX_ILL_BADSTK; // This mapping for a floating point stack fault is still uncertain and needs confirmation against
// the guest signal behavior
        break;
    case EXCEPTION_BREAKPOINT:
    case EXCEPTION_SINGLE_STEP:
        signo = POSIX_SIGTRAP;
        si_code = POSIX_TRAP_BRKPT;
        break;
    case DBG_PRINTEXCEPTION_C:
    case DBG_PRINTEXCEPTION_WIDE_C:
        // Used by OutputDebugString functions.
        return EXCEPTION_CONTINUE_EXECUTION;
    case MS_VC_EXCEPTION:
        LOG_DEBUG(Debug, "Pass MS_VC_EXCEPTION at {} to handler", address);
        return EXCEPTION_EXECUTE_HANDLER;
    case MSVC_CPP_EXCEPTION:
        // First chance of a host C++ throw (e.g. std::filesystem errors in the save data backup thread). The
        // C++ runtime delivers it to its catch block; treating it as an unhandled crash would log a false
        // Critical and run the emulator shutdown path
        return EXCEPTION_CONTINUE_SEARCH;
    default:
        break;
    }

    if (handled) {
        return EXCEPTION_CONTINUE_EXECUTION;
    }

    return HandleUnclaimedException(pExp, code, address, signo, si_code, static_protection_exception);
}

#else

static std::string DisassembleInstruction(void* code_address) {
    char buffer[256] = "<unable to decode>";

#ifdef ARCH_X86_64
    ZydisDecodedInstruction instruction;
    ZydisDecodedOperand operands[ZYDIS_MAX_OPERAND_COUNT];
    const auto status =
        Common::Decoder::Instance()->decodeInstruction(instruction, operands, code_address);
    if (ZYAN_SUCCESS(status)) {
        ZydisFormatter formatter;
        ZydisFormatterInit(&formatter, ZYDIS_FORMATTER_STYLE_INTEL);
        ZydisFormatterFormatInstruction(&formatter, &instruction, operands,
                                        instruction.operand_count_visible, buffer, sizeof(buffer),
                                        reinterpret_cast<u64>(code_address), ZYAN_NULL);
    }
#endif

    return buffer;
}

static s32 NativeSiCodeToGuest(s32 sig, s32 code) {
    using namespace Libraries::Kernel;
    switch (sig) {
    case SIGUSR1:
        return POSIX_SI_LWP;
    case SIGSEGV:
        switch (code) {
        case SEGV_MAPERR:
            return POSIX_SEGV_MAPERR;
        case SEGV_ACCERR:
            return POSIX_SEGV_ACCERR;
        }
    case SIGBUS:
        switch (code) {
        case BUS_ADRALN:
            return POSIX_BUS_ADRALN;
        case BUS_ADRERR:
            return POSIX_BUS_ADRERR;
        case BUS_OBJERR:
            return POSIX_BUS_OBJERR;
        }
    case SIGILL:
        switch (code) {
        case ILL_ILLOPC:
            return POSIX_ILL_ILLOPC;
        case ILL_ILLOPN:
            return POSIX_ILL_ILLOPN;
        case ILL_ILLADR:
            return POSIX_ILL_ILLADR;
        case ILL_ILLTRP:
            return POSIX_ILL_ILLTRP;
        case ILL_PRVOPC:
            return POSIX_ILL_PRVOPC;
        case ILL_PRVREG:
            return POSIX_ILL_PRVREG;
        case ILL_COPROC:
            return POSIX_ILL_COPROC;
        case ILL_BADSTK:
            return POSIX_ILL_BADSTK;
        }
    case SIGFPE:
        switch (code) {
        case FPE_INTOVF:
            return POSIX_FPE_INTOVF;
        case FPE_INTDIV:
            return POSIX_FPE_INTDIV;
        case FPE_FLTDIV:
            return POSIX_FPE_FLTDIV;
        case FPE_FLTOVF:
            return POSIX_FPE_FLTOVF;
        case FPE_FLTUND:
            return POSIX_FPE_FLTUND;
        case FPE_FLTRES:
            return POSIX_FPE_FLTRES;
        case FPE_FLTINV:
            return POSIX_FPE_FLTINV;
        case FPE_FLTSUB:
            return POSIX_FPE_FLTSUB;
        }
    case SIGTRAP:
        switch (code) {
        case TRAP_BRKPT:
            return POSIX_TRAP_BRKPT;
        case TRAP_TRACE:
            return POSIX_TRAP_TRACE;
#ifdef __FreeBSD__
        case TRAP_DTRACE:
            return POSIX_TRAP_DTRACE;
#endif
        }

    default:
        return POSIX_SI_NOINFO;
    }
}

void SignalHandler(int sig, siginfo_t* info, void* raw_context) {
    using namespace Libraries::Kernel;
    auto* thread = g_curthread;
    const auto* signals = Signals::Instance();

    auto* code_address = Common::GetRip(raw_context);

    Ucontext context{info, reinterpret_cast<ucontext_t*>(raw_context)};
    Siginfo guest_info{};
    if (info) {
        guest_info = *reinterpret_cast<Siginfo*>(info);
        guest_info._si_signo = sig == SIGUSR1 ? 0 : NativeToOrbisSignal(info->si_signo);
        guest_info._si_errno = NativeToPosixErrno(info->si_errno);
        guest_info._si_code = NativeSiCodeToGuest(sig, info->si_code);
        guest_info._si_addr = (void*)context.uc_mcontext.mc_rip;
    }
    Siginfo* info_p = info ? &guest_info : nullptr;
    Ucontext* context_p = raw_context ? &context : nullptr;

    switch (sig) {
    case SIGSEGV:
    case SIGBUS: {
        const bool is_write = Common::IsWriteError(raw_context);
        if (!signals->DispatchAccessViolation(raw_context, info->si_addr)) {
            if (thread && thread->DispatchSignal(NativeToOrbisSignal(sig), info_p, context_p)) {
                return;
            }
            UNREACHABLE_MSG("Unhandled access violation at code address {}: {} address {}",
                            fmt::ptr(code_address), is_write ? "Write to" : "Read from",
                            fmt::ptr(info->si_addr));
        }
        break;
    }
    case SIGILL:
        if (signals->DispatchIllegalInstruction(raw_context)) {
            return;
        }
    case SIGFPE:
    case SIGTRAP:
    case SIGSYS: {
        if (thread && thread->DispatchSignal(NativeToOrbisSignal(sig), info_p, context_p)) {
            return;
        }

        UNREACHABLE_MSG("Unhandled signal {} at code address {}", sig, fmt::ptr(code_address));
    }
    case SIGSLEEP: {
        // Sleep thread until signal is received again
        sigset_t sigset;
        sigemptyset(&sigset);
        sigaddset(&sigset, SIGSLEEP);
        sigwait(&sigset, &sig);
        break;
    }
    case SIGUSR1:
        if (thread) {
            thread->DispatchPendingSignals(info_p, context_p);
        }
        break;
    default:
        UNREACHABLE_MSG("Unhandled signal {} at code address {}", sig, fmt::ptr(code_address));
    }
}

#endif

SignalDispatch::SignalDispatch() {
#if defined(_WIN32)
    ASSERT_MSG(handle = AddVectoredExceptionHandler(0, SignalHandler),
               "Failed to register exception handler.");
    const auto previous = SetUnhandledExceptionFilter(TopLevelExceptionFilter);
    previous_top_level_filter = previous != TopLevelExceptionFilter ? previous : nullptr;
    std::signal(SIGABRT, AbortSignalHandler);
#else
    struct sigaction action{};
    action.sa_sigaction = SignalHandler;
    action.sa_flags = SA_SIGINFO | SA_ONSTACK;
    sigemptyset(&action.sa_mask);

    ASSERT_MSG(
        sigaction(SIGSEGV, &action, nullptr) == 0 && sigaction(SIGBUS, &action, nullptr) == 0 &&
            sigaction(SIGILL, &action, nullptr) == 0 && sigaction(SIGFPE, &action, nullptr) == 0 &&
            sigaction(SIGTRAP, &action, nullptr) == 0 && sigaction(SIGSYS, &action, nullptr) == 0 &&
            sigaction(SIGUSR1, &action, nullptr) == 0 && sigaction(SIGSLEEP, &action, nullptr) == 0,
        "Failed to register signal handlers.");
#endif
}

void SignalDispatch::RemoveHandlers() {
    // asserting here would get into an infinite loop until too
    // many nested exceptions makes the OS kill the process
#if defined(_WIN32)
    if (!(RemoveVectoredExceptionHandler(handle))) {
        LOG_CRITICAL(Core, "Failed to remove exception handler.");
        std::quick_exit(1);
    }
#else
    struct sigaction action{};
    action.sa_handler = SIG_DFL;
    action.sa_flags = 0;
    sigemptyset(&action.sa_mask);

    if (!(sigaction(SIGSEGV, &action, nullptr) == 0 && sigaction(SIGBUS, &action, nullptr) == 0 &&
          sigaction(SIGILL, &action, nullptr) == 0 && sigaction(SIGFPE, &action, nullptr) == 0 &&
          sigaction(SIGTRAP, &action, nullptr) == 0 && sigaction(SIGSYS, &action, nullptr) == 0 &&
          sigaction(SIGUSR1, &action, nullptr) == 0 &&
          sigaction(SIGSLEEP, &action, nullptr) == 0)) {
        LOG_CRITICAL(Core, "Failed to remove signal handlers.");
        std::quick_exit(1);
    }
#endif
}

SignalDispatch::~SignalDispatch() {
    RemoveHandlers();
}

bool SignalDispatch::DispatchAccessViolation(void* context, void* fault_address) const {
    for (const auto& [handler, _] : access_violation_handlers) {
        if (handler(context, fault_address)) {
            return true;
        }
    }
    return false;
}

bool SignalDispatch::DispatchIllegalInstruction(void* context) const {
    for (const auto& [handler, _] : illegal_instruction_handlers) {
        if (handler(context)) {
            return true;
        }
    }
    return false;
}

} // namespace Core
