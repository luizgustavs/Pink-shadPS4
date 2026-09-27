// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/cp_profiler.h"

#ifdef _WIN32

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <windows.h>
// After windows.h: the K32 module queries live in kernel32, no psapi.lib needed
#include <psapi.h>

#include <fmt/format.h>

#include "common/logging/log.h"
#include "common/path_util.h"
#include "common/thread.h"
#include "common/types.h"

namespace Common::CpProfiler {

namespace {

constexpr size_t MaxFrames = 96;

struct Module {
    u64 base;
    u64 size;
    const RUNTIME_FUNCTION* functions;
    u32 num_functions;
    std::string name;
};

struct Target {
    HANDLE thread;
    u64 stack_low;
    u64 stack_high;
};

struct StackHash {
    size_t operator()(const std::vector<u64>& frames) const {
        size_t hash = 1469598103934665603ULL;
        for (const u64 frame : frames) {
            hash = (hash ^ frame) * 1099511628211ULL;
        }
        return hash;
    }
};

// Loaded modules and their .pdata tables, read once when sampling starts. The unwinder looks function entries
// up here instead of calling RtlLookupFunctionEntry, which takes a loader lock the suspended thread may hold
// while it dispatches a fault
std::vector<Module> LoadModules() {
    std::vector<Module> modules;
    std::array<HMODULE, 1024> handles{};
    DWORD needed = 0;
    const HANDLE process = GetCurrentProcess();
    if (!K32EnumProcessModules(process, handles.data(), sizeof(handles), &needed)) {
        return modules;
    }
    const size_t count = std::min<size_t>(needed / sizeof(HMODULE), handles.size());
    for (size_t i = 0; i < count; ++i) {
        MODULEINFO info{};
        if (!K32GetModuleInformation(process, handles[i], &info, sizeof(info))) {
            continue;
        }
        char name[MAX_PATH] = {};
        K32GetModuleBaseNameA(process, handles[i], name, MAX_PATH);
        const u64 base = reinterpret_cast<u64>(info.lpBaseOfDll);
        const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
        const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(base + dos->e_lfanew);
        const auto& dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXCEPTION];
        modules.push_back(Module{
            .base = base,
            .size = info.SizeOfImage,
            .functions = dir.Size ? reinterpret_cast<const RUNTIME_FUNCTION*>(
                                        base + dir.VirtualAddress)
                                  : nullptr,
            .num_functions = static_cast<u32>(dir.Size / sizeof(RUNTIME_FUNCTION)),
            .name = name,
        });
    }
    std::sort(modules.begin(), modules.end(),
              [](const Module& a, const Module& b) { return a.base < b.base; });
    return modules;
}

const Module* FindModule(const std::vector<Module>& modules, u64 pc) {
    auto it = std::upper_bound(modules.begin(), modules.end(), pc,
                               [](u64 value, const Module& m) { return value < m.base; });
    if (it == modules.begin()) {
        return nullptr;
    }
    --it;
    return pc - it->base < it->size ? &*it : nullptr;
}

PRUNTIME_FUNCTION FindFunction(const Module& module, u64 pc) {
    const u32 rva = static_cast<u32>(pc - module.base);
    const RUNTIME_FUNCTION* first = module.functions;
    const RUNTIME_FUNCTION* last = module.functions + module.num_functions;
    const auto* it = std::upper_bound(first, last, rva, [](u32 value, const RUNTIME_FUNCTION& f) {
        return value < f.BeginAddress;
    });
    if (it == first) {
        return nullptr;
    }
    --it;
    return rva < it->EndAddress ? const_cast<PRUNTIME_FUNCTION>(it) : nullptr;
}

// Walks the suspended thread's stack from its context. Kept free of C++ objects that need unwinding so it can
// sit inside __try: a torn stack must end the walk, not the emulator
u32 Unwind(const std::vector<Module>& modules, const Target& target, CONTEXT* ctx, u64* frames) {
    u32 count = 0;
    __try {
        while (count < MaxFrames) {
            const u64 pc = ctx->Rip;
            if (pc == 0 || ctx->Rsp < target.stack_low || ctx->Rsp >= target.stack_high) {
                break;
            }
            frames[count++] = pc;
            const Module* module = FindModule(modules, pc);
            PRUNTIME_FUNCTION function =
                module && module->functions ? FindFunction(*module, pc) : nullptr;
            if (function) {
                PVOID handler_data = nullptr;
                DWORD64 establisher = 0;
                RtlVirtualUnwind(UNW_FLAG_NHANDLER, module->base, pc, function, ctx,
                                 &handler_data, &establisher, nullptr);
            } else if (count == 1) {
                // Leaf function without unwind data: the return address is on top of the stack
                ctx->Rip = *reinterpret_cast<const u64*>(ctx->Rsp);
                ctx->Rsp += 8;
            } else {
                break;
            }
        }
    } __except (EXCEPTION_EXECUTE_HANDLER) {
    }
    return count;
}

void WriteReport(const std::vector<Module>& modules,
                 const std::unordered_map<std::vector<u64>, u32, StackHash>& stacks,
                 const std::string& header, const char* filename) {
    const auto path = GetUserPath(FS::PathType::LogDir) / filename;
    std::FILE* file = _wfopen(path.c_str(), L"w");
    if (!file) {
        LOG_ERROR(Render_Vulkan, "Harness cp_profile: cannot write {}", path.string());
        return;
    }
    fmt::print(file, "# {}\n", header);
    for (const auto& module : modules) {
        fmt::print(file, "module {} {:#x} {:#x}\n", module.name, module.base, module.size);
    }
    for (const auto& [frames, count] : stacks) {
        fmt::print(file, "stack {}", count);
        for (const u64 frame : frames) {
            fmt::print(file, " {:x}", frame);
        }
        fmt::print(file, "\n");
    }
    std::fclose(file);
    LOG_WARNING(Render_Vulkan, "Harness cp_profile: {} unique stacks, {} -> {}", stacks.size(),
                header, path.string());
}

void Sample(Target target, double start_s, double duration_s, u32 hz, const char* filename,
            const char* name) {
    SetCurrentThreadName("shadPS4:CpProfiler");
    std::this_thread::sleep_for(std::chrono::duration<double>(start_s));

    const auto modules = LoadModules();
    HANDLE timer = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
                                          TIMER_ALL_ACCESS);
    if (!timer) {
        timer = CreateWaitableTimerExW(nullptr, nullptr, 0, TIMER_ALL_ACCESS);
    }

    std::unordered_map<std::vector<u64>, u32, StackHash> stacks;
    std::array<u64, MaxFrames> frames{};
    u64 samples = 0;
    u64 failed = 0;
    u64 suspended_ns = 0;
    const auto begin = std::chrono::steady_clock::now();
    const auto end = begin + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                                 std::chrono::duration<double>(duration_s));
    LARGE_INTEGER due{};
    due.QuadPart = -static_cast<LONGLONG>(10'000'000 / hz);
    while (std::chrono::steady_clock::now() < end) {
        SetWaitableTimer(timer, &due, 0, nullptr, nullptr, FALSE);
        WaitForSingleObject(timer, INFINITE);

        const auto suspend_start = std::chrono::steady_clock::now();
        if (SuspendThread(target.thread) == static_cast<DWORD>(-1)) {
            ++failed;
            continue;
        }
        CONTEXT ctx{};
        ctx.ContextFlags = CONTEXT_FULL;
        u32 count = 0;
        // GetThreadContext also waits until the suspension has taken effect
        if (GetThreadContext(target.thread, &ctx)) {
            Target sample_target = target;
            if (ctx.Rsp < target.stack_low || ctx.Rsp >= target.stack_high) {
                // Guest workers run on game/fiber stacks. GetCurrentThreadStackLimits at registration
                // describes only the stack active then; use the sampled stack's committed region when that
                // fiber has since switched
                MEMORY_BASIC_INFORMATION region{};
                if (VirtualQuery(reinterpret_cast<void*>(ctx.Rsp), &region, sizeof(region)) &&
                    region.State == MEM_COMMIT) {
                    sample_target.stack_low = reinterpret_cast<u64>(region.BaseAddress);
                    sample_target.stack_high = sample_target.stack_low + region.RegionSize;
                }
            }
            count = Unwind(modules, sample_target, &ctx, frames.data());
        }
        ResumeThread(target.thread);
        suspended_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - suspend_start)
                            .count();
        if (count == 0) {
            ++failed;
            continue;
        }
        ++samples;
        ++stacks[std::vector<u64>(frames.begin(), frames.begin() + count)];
    }
    const double elapsed_s =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
    CloseHandle(timer);
    CloseHandle(target.thread);

    const std::string header =
        fmt::format("{} hz={} start_s={} duration_s={:.3f} samples={} failed={} "
                    "suspend_us_avg={:.1f}",
                    name, hz, start_s, elapsed_s, samples, failed,
                    samples + failed ? suspended_ns / 1000.0 / (samples + failed) : 0.0);
    WriteReport(modules, stacks, header, filename);
}

void RegisterTarget(const char* variable, const char* filename, const char* name) {
    const char* value = std::getenv(variable);
    if (!value || !*value) {
        return;
    }
    double start_s = 0.0;
    double duration_s = 0.0;
    u32 hz = 1000;
    if (std::sscanf(value, "%lf:%lf:%u", &start_s, &duration_s, &hz) < 2 || duration_s <= 0.0 ||
        hz == 0 || hz > 10000) {
        LOG_ERROR(Render_Vulkan, "Harness cp_profile: bad {} '{}'", variable, value);
        return;
    }
    Target target{};
    if (!DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(),
                         &target.thread,
                         THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION,
                         FALSE, 0)) {
        return;
    }
    ULONG_PTR low = 0;
    ULONG_PTR high = 0;
    GetCurrentThreadStackLimits(&low, &high);
    target.stack_low = low;
    target.stack_high = high;
    LOG_WARNING(Render_Vulkan, "Harness {}: sampling at {} Hz from {} s for {} s", name, hz,
                start_s, duration_s);
    std::thread{Sample, target, start_s, duration_s, hz, filename, name}.detach();
}

// M3b (Frente J): every guest thread, sampled by one shared sampler so that the rate does not scale with the
// number of threads. The window starts when the first guest thread registers
struct GuestThread {
    Target target;
    std::string name;
    std::unordered_map<std::vector<u64>, u32, StackHash> stacks;
    u64 samples = 0;
    u64 failed = 0;
};

struct GuestSampler {
    std::mutex mutex;
    std::vector<std::unique_ptr<GuestThread>> threads;
    double start_s = 0.0;
    double duration_s = 0.0;
    u32 hz = 250;
};

void SampleGuests(GuestSampler* sampler) {
    SetCurrentThreadName("shadPS4:GuestProfiler");
    std::this_thread::sleep_for(std::chrono::duration<double>(sampler->start_s));
    const auto modules = LoadModules();
    HANDLE timer = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
                                          TIMER_ALL_ACCESS);
    if (!timer) {
        timer = CreateWaitableTimerExW(nullptr, nullptr, 0, TIMER_ALL_ACCESS);
    }
    std::array<u64, MaxFrames> frames{};
    u64 ticks = 0;
    const auto begin = std::chrono::steady_clock::now();
    const auto end = begin + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                                 std::chrono::duration<double>(sampler->duration_s));
    LARGE_INTEGER due{};
    due.QuadPart = -static_cast<LONGLONG>(10'000'000 / sampler->hz);
    while (std::chrono::steady_clock::now() < end) {
        SetWaitableTimer(timer, &due, 0, nullptr, nullptr, FALSE);
        WaitForSingleObject(timer, INFINITE);
        ++ticks;
        std::scoped_lock lk{sampler->mutex};
        for (auto& thread : sampler->threads) {
            if (SuspendThread(thread->target.thread) == static_cast<DWORD>(-1)) {
                ++thread->failed;
                continue;
            }
            CONTEXT ctx{};
            ctx.ContextFlags = CONTEXT_FULL;
            u32 count = 0;
            if (GetThreadContext(thread->target.thread, &ctx)) {
                Target sample_target = thread->target;
                if (ctx.Rsp < sample_target.stack_low || ctx.Rsp >= sample_target.stack_high) {
                    // Guest code runs on the guest stack, not the host one seen at registration
                    MEMORY_BASIC_INFORMATION region{};
                    if (VirtualQuery(reinterpret_cast<void*>(ctx.Rsp), &region, sizeof(region)) &&
                        region.State == MEM_COMMIT) {
                        sample_target.stack_low = reinterpret_cast<u64>(region.BaseAddress);
                        sample_target.stack_high = sample_target.stack_low + region.RegionSize;
                    }
                }
                count = Unwind(modules, sample_target, &ctx, frames.data());
            }
            ResumeThread(thread->target.thread);
            if (count == 0) {
                // An exited thread fails SuspendThread or GetThreadContext; count it as no sample
                ++thread->failed;
                continue;
            }
            ++thread->samples;
            ++thread->stacks[std::vector<u64>(frames.begin(), frames.begin() + count)];
        }
    }
    const double elapsed_s =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
    CloseHandle(timer);
    std::scoped_lock lk{sampler->mutex};
    std::string index;
    for (size_t i = 0; i < sampler->threads.size(); ++i) {
        const auto& thread = *sampler->threads[i];
        if (thread.samples == 0) {
            continue;
        }
        std::string safe_name = thread.name;
        for (char& c : safe_name) {
            if (!std::isalnum(static_cast<unsigned char>(c))) {
                c = '_';
            }
        }
        const std::string filename = fmt::format("guest_profile_{:03}_{}.txt", i, safe_name);
        const std::string header = fmt::format(
            "guest_profile thread={} hz={} start_s={} duration_s={:.3f} ticks={} samples={} "
            "failed={}",
            thread.name, sampler->hz, sampler->start_s, elapsed_s, ticks, thread.samples,
            thread.failed);
        WriteReport(modules, thread.stacks, header, filename.c_str());
        index += fmt::format("{} {} {} {}\n", filename, thread.samples, ticks, thread.name);
    }
    const auto path = GetUserPath(FS::PathType::LogDir) / "guest_profile_index.txt";
    if (std::FILE* file = _wfopen(path.c_str(), L"w")) {
        fmt::print(file, "{}", index);
        std::fclose(file);
    }
}

} // Anonymous namespace

void RegisterGuestThread(const char* name) {
    static const char* value = std::getenv("SHADPS4_GUEST_PROFILE");
    if (!value || !*value) {
        return;
    }
    static GuestSampler* sampler = [] {
        auto* new_sampler = new GuestSampler{};
        if (std::sscanf(value, "%lf:%lf:%u", &new_sampler->start_s, &new_sampler->duration_s,
                        &new_sampler->hz) < 2 ||
            new_sampler->duration_s <= 0.0 || new_sampler->hz == 0 || new_sampler->hz > 2000) {
            LOG_ERROR(Render_Vulkan, "Harness guest_profile: bad SHADPS4_GUEST_PROFILE '{}'",
                      value);
            return static_cast<GuestSampler*>(nullptr);
        }
        LOG_WARNING(Render_Vulkan, "Harness guest_profile: all guest threads at {} Hz from {} s for {} s",
                    new_sampler->hz, new_sampler->start_s, new_sampler->duration_s);
        std::thread{SampleGuests, new_sampler}.detach();
        return new_sampler;
    }();
    if (!sampler) {
        return;
    }
    auto thread = std::make_unique<GuestThread>();
    if (!DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(),
                         &thread->target.thread,
                         THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION,
                         FALSE, 0)) {
        return;
    }
    ULONG_PTR low = 0;
    ULONG_PTR high = 0;
    GetCurrentThreadStackLimits(&low, &high);
    thread->target.stack_low = low;
    thread->target.stack_high = high;
    thread->name = name;
    std::scoped_lock lk{sampler->mutex};
    sampler->threads.push_back(std::move(thread));
}

void RegisterCurrentThread() {
    RegisterTarget("SHADPS4_CP_PROFILE", "cp_profile.txt", "cp_profile");
}

void RegisterGameRenderThread() {
    // Called on every sceGnmSubmitDone
    static const bool enabled = std::getenv("SHADPS4_GAME_PROFILE") != nullptr;
    if (!enabled) {
        return;
    }
    static std::atomic_flag registered = ATOMIC_FLAG_INIT;
    if (!registered.test_and_set(std::memory_order_relaxed)) {
        RegisterTarget("SHADPS4_GAME_PROFILE", "game_profile.txt", "game_profile");
    }
}

void RegisterGameMainThread() {
    RegisterTarget("SHADPS4_GAME_PROFILE", "game_main_profile.txt", "game_main_profile");
}

} // namespace Common::CpProfiler

#else

namespace Common::CpProfiler {

void RegisterCurrentThread() {}
void RegisterGuestThread(const char*) {}
void RegisterGameRenderThread() {}
void RegisterGameMainThread() {}

} // namespace Common::CpProfiler

#endif
