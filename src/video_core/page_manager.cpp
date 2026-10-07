// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>
#include "common/adaptive_mutex.h"
#include "common/assert.h"
#include "common/debug.h"
#include "common/div_ceil.h"
#include "common/error.h"
#include "common/multi_level_page_table.h"
#include "common/perf_stats.h"
#include "common/signal_context.h"
#include "common/thread.h"
#include "core/emulator_settings.h"
#include "core/memory.h"
#include "core/signals.h"
#include "shader_recompiler/ir/passes/srt.h"
#include "video_core/page_manager.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"

#ifndef _WIN64
#include <sys/mman.h>
#include "common/adaptive_mutex.h"
#else
#include <windows.h>
#endif

#ifdef __linux__
#include <thread>
#include <fcntl.h>
#include <linux/userfaultfd.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#else
#include "common/spin_lock.h"
#endif

#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#endif

namespace VideoCore {

namespace {
/// Use one atomic byte per page instead of the larger MSVC mutex
/// Yield after a short spin because the holder may be in VirtualProtect
struct PageSpinLock {
    std::atomic<u8> held{0};

    void lock() noexcept {
        u32 spins = 0;
        while (held.exchange(1, std::memory_order_acquire) != 0) {
            while (held.load(std::memory_order_relaxed) != 0) {
                if (++spins < 64) {
#if defined(_M_X64) || defined(__x86_64__)
                    _mm_pause();
#endif
                } else {
                    std::this_thread::yield();
                }
            }
        }
    }

    void unlock() noexcept {
        held.store(0, std::memory_order_release);
    }
};
} // Anonymous namespace

struct PageManager::Impl {
    struct PageState {
        u8 num_write_watchers;
        u8 num_read_watchers;

        Core::MemoryPermission WritePerm() const noexcept {
            return num_write_watchers == 0 ? Core::MemoryPermission::Write
                                           : Core::MemoryPermission::None;
        }

        Core::MemoryPermission ReadPerm() const noexcept {
            return num_read_watchers == 0 ? Core::MemoryPermission::Read
                                          : Core::MemoryPermission::None;
        }

        Core::MemoryPermission Perms() const noexcept {
            return ReadPerm() | WritePerm();
        }

        template <bool is_read>
        u8 GetPage() const {
            if constexpr (is_read) {
                return num_read_watchers;
            } else {
                return num_write_watchers;
            }
        }

        constexpr Core::MemoryPermission Update(PageOp write_op, bool update_write = true,
                                                PageOp read_op = PageOp::None,
                                                bool update_read = false) {
            if (update_read) {
                if (read_op == PageOp::Track) {
                    ASSERT_MSG(num_read_watchers == 0, "Too many watchers");
                } else if (read_op == PageOp::Untrack) {
                    ASSERT_MSG(num_read_watchers > 0, "Not enough watchers");
                }
                num_read_watchers += std::to_underlying(read_op);
            }
            if (update_write) {
                if (write_op == PageOp::Track) {
                    ASSERT_MSG(num_write_watchers < 255, "Too many watchers");
                } else if (write_op == PageOp::Untrack) {
                    ASSERT_MSG(num_write_watchers > 0, "Not enough watchers");
                }
                num_write_watchers += std::to_underlying(write_op);
            }
            return Perms();
        }
    };

    static constexpr size_t ADDRESS_BITS = 40;
    static constexpr size_t NUM_ADDRESS_PAGES = 1ULL << (40 - PM_PAGE_BITS);
    static constexpr size_t NUM_ADDRESS_LOCKS = NUM_ADDRESS_PAGES / NUM_REGION_PAGES;
    inline static Vulkan::Rasterizer* rasterizer;

    Impl() : page_spin_locks{EmulatorSettings.IsPageSpinLocks()} {
        if (page_spin_locks) {
            LOG_WARNING(Render, "Workaround page_spin_locks enabled");
        }
    }
    virtual ~Impl() = default;

    virtual void OnMap(VAddr address, size_t size) {
        // No-op
        EnsurePages(address, address + size);
    }

    virtual void OnUnmap(VAddr address, size_t size) {
        // No-op
    }

    virtual void Protect(VAddr address, size_t size, Core::MemoryPermission perms) = 0;

    void EnsurePages(VAddr begin, VAddr end) {
        end = std::min(end, VAddr{1} << ADDRESS_BITS) - 1;
        const size_t start_page = begin >> PM_PAGE_BITS;
        const size_t end_page = end >> PM_PAGE_BITS;
        cached_pages.reserve(start_page, end_page);
        if (page_spin_locks) {
            spin_locks.reserve(start_page, end_page);
        } else {
            locks.reserve(start_page, end_page);
        }
    }

    /// Locks a page that has a state; its lock was reserved with it
    void LockPage(u64 page) {
        if (page_spin_locks) {
            spin_locks[page].lock();
        } else {
            locks[page].lock();
        }
    }

    void UnlockPage(u64 page) {
        if (page_spin_locks) {
            if (auto* lock = spin_locks.find(page)) {
                lock->unlock();
            }
        } else if (auto* lock = locks.find(page)) {
            lock->unlock();
        }
    }

    void UpdatePageWatchers(VAddr addr, u64 size, PageOp write_op) {
        const u64 page_start = addr >> PM_PAGE_BITS;
        const u64 page_end = Common::DivCeil(addr + size, PM_PAGE_SIZE);

        Core::MemoryPermission perms{};
        u64 range_begin = page_start;
        u64 range_pages = 0;
        u64 potential_pages = 0;

        const auto release_pending = [&] {
            if (range_pages > 0) {
                Protect(range_begin << PM_PAGE_BITS, range_pages << PM_PAGE_BITS, perms);
                range_pages = 0;
                potential_pages = 0;
            }
        };

        // Iterate requested pages
        const u64 aligned_addr = page_start << PM_PAGE_BITS;
        const u64 aligned_end = page_end << PM_PAGE_BITS;
        if (!rasterizer->IsMapped(aligned_addr, aligned_end - aligned_addr)) {
            LOG_WARNING(Render,
                        "Tracking memory region {:#x} - {:#x} which is not fully GPU mapped.",
                        aligned_addr, aligned_end);
            EnsurePages(aligned_addr, aligned_end);
        }

        for (u64 page = page_start; page != page_end; ++page) {
            PageState* state = cached_pages.find(page);
            if (!state) {
                continue;
            }

            LockPage(page);

            const auto old_perms = state->Perms();
            if (page == page_start) {
                perms = old_perms;
            }

            // Apply the change to the page state
            const auto new_perms = state->Update(write_op);
            if (new_perms != perms) [[unlikely]] {
                // If the protection changed add pending (un)protect action
                release_pending();
                perms = new_perms;
            } else if (range_pages != 0) {
                ++potential_pages;
            }

            // If the page must be (un)protected
            if (new_perms != old_perms) {
                if (range_pages == 0) {
                    // Start a new potential range
                    range_begin = page;
                    potential_pages = 1;
                }
                // Extend current range up to potential range
                range_pages = potential_pages;
            }
        }

        // Add pending (un)protect action
        release_pending();

        for (u64 page = page_start; page != page_end; ++page) {
            UnlockPage(page);
        }
    }

    void UpdatePageWatchersForRegion(VAddr base_addr, const Bounds& bounds,
                                     const RegionBits& write_mask, const RegionBits& read_mask,
                                     PageOp write_op, PageOp read_op) {
        const u64 base_page = base_addr >> PM_PAGE_BITS;
        const u64 page_start = bounds.start_word * PAGES_PER_WORD + bounds.start_page;
        const u64 page_end = bounds.end_word * PAGES_PER_WORD + bounds.end_page + 1;

        Core::MemoryPermission perms{};
        u64 range_begin = base_page + page_start;
        u64 range_pages = 0;
        u64 potential_pages = 0;

        const auto release_pending = [&] {
            if (range_pages > 0) {
                Protect(range_begin << PM_PAGE_BITS, range_pages << PM_PAGE_BITS, perms);
                range_pages = 0;
                potential_pages = 0;
            }
        };

        for (u64 page = page_start; page != page_end; ++page) {
            PageState* state = cached_pages.find(base_page + page);
            if (!state) {
                continue;
            }

            LockPage(base_page + page);

            const auto old_perms = state->Perms();
            if (page == page_start) {
                perms = old_perms;
            }

            // Apply the change to the page state
            const bool update_write = write_op != PageOp::None && write_mask.GetPage(page);
            const bool update_read = read_op != PageOp::None && read_mask.GetPage(page);
            const auto new_perms = state->Update(write_op, update_write, read_op, update_read);

            if (new_perms != perms) [[unlikely]] {
                // If the protection changed add pending (un)protect action
                release_pending();
                perms = new_perms;
            } else if (range_pages != 0) {
                // If the protection did not change, extend the potential range
                ++potential_pages;
            }

            // If the page must be (un)protected
            if (new_perms != old_perms) {
                if (range_pages == 0) {
                    // Start a new potential range
                    range_begin = base_page + page;
                    potential_pages = 1;
                }
                // Extend current rango up to potential range
                range_pages = potential_pages;
            }
        }

        // Add pending (un)protect action
        release_pending();

        for (u64 page = page_start; page != page_end; ++page) {
            UnlockPage(base_page + page);
        }
    }

    struct PageTraits {
        using Entry = PageState;
        static constexpr size_t ADDRESS_SPACE_BITS = ADDRESS_BITS;
        static constexpr size_t L1_BITS = 16;
        static constexpr size_t PAGE_BITS = PM_PAGE_BITS;
        static constexpr bool NULL_CHECK = false;
    };
    Common::MultiLevelPageTable<PageTraits> cached_pages;
    struct MutexTraits {
#ifdef PTHREAD_ADAPTIVE_MUTEX_INITIALIZER_NP
        using Entry = Common::AdaptiveMutex;
#else
        using Entry = std::mutex;
#endif
        static constexpr size_t ADDRESS_SPACE_BITS = ADDRESS_BITS;
        static constexpr size_t L1_BITS = 16;
        static constexpr size_t PAGE_BITS = PM_PAGE_BITS;
        static constexpr bool NULL_CHECK = false;
    };
    Common::MultiLevelPageTable<MutexTraits> locks;
    struct SpinLockTraits {
        using Entry = PageSpinLock;
        static constexpr size_t ADDRESS_SPACE_BITS = ADDRESS_BITS;
        static constexpr size_t L1_BITS = 16;
        static constexpr size_t PAGE_BITS = PM_PAGE_BITS;
        static constexpr bool NULL_CHECK = false;
    };
    Common::MultiLevelPageTable<SpinLockTraits> spin_locks;
    // page_spin_locks: fixed at boot, since switching tables while pages are held would
    // break exclusion; the pages are locked through one table or the other
    const bool page_spin_locks;
};

#ifdef __linux__
struct UffdImpl : public PageManager::Impl {
private:
    std::jthread ufd_thread;
    int uffd;

public:
    UffdImpl(Vulkan::Rasterizer* rasterizer_) : Impl() {
        rasterizer = rasterizer_;
        uffd = syscall(SYS_userfaultfd, O_CLOEXEC | O_NONBLOCK | UFFD_USER_MODE_ONLY);
        if (uffd == -1) {
            LOG_ERROR(Common_Memory,
                      "userfaultfd syscall failed: {}, falling back to signal implementation",
                      Common::GetLastErrorMsg());
            throw std::runtime_error("userfaultfd");
        }

        // Request uffdio features from kernel.
        uffdio_api api;
        api.api = UFFD_API;
        api.features = UFFD_FEATURE_THREAD_ID;
        const int ret = ioctl(uffd, UFFDIO_API, &api);
        if (ret != 0) {
            LOG_ERROR(Common_Memory,
                      "uffdio_api call failed: {}, falling back to signal implementation",
                      Common::GetLastErrorMsg());
            throw std::runtime_error("uffdio_api");
        }

        // Create uffd handler thread
        ufd_thread = std::jthread([&](std::stop_token token) { UffdHandler(token); });
    }

    ~UffdImpl() = default;

    void OnMap(VAddr address, size_t size) override {
        PageManager::Impl::OnMap(address, size);
        uffdio_register reg;
        reg.range.start = address;
        reg.range.len = size;
        reg.mode = UFFDIO_REGISTER_MODE_WP;
        const int ret = ioctl(uffd, UFFDIO_REGISTER, &reg);
        ASSERT_MSG(ret != -1, "Uffdio register failed with error: {}", Common::GetLastErrorMsg());
    }

    void OnUnmap(VAddr address, size_t size) override {
        uffdio_range range;
        range.start = address;
        range.len = size;
        const int ret = ioctl(uffd, UFFDIO_UNREGISTER, &range);
        ASSERT_MSG(ret != -1, "Uffdio unregister failed with error: {}", Common::GetLastErrorMsg());
    }

    void Protect(VAddr address, size_t size, Core::MemoryPermission perms) override {
        bool allow_write = True(perms & Core::MemoryPermission::Write);
        uffdio_writeprotect wp;
        wp.range.start = address;
        wp.range.len = size;
        wp.mode = allow_write ? UFFDIO_WRITEPROTECT_MODE_DONTWAKE : UFFDIO_WRITEPROTECT_MODE_WP;
        const int ret = ioctl(uffd, UFFDIO_WRITEPROTECT, &wp);
        ASSERT_MSG(ret != -1, "Uffdio writeprotect failed with error: {}",
                   Common::GetLastErrorMsg());
    }

    void UffdHandler(std::stop_token token) {
        Common::SetCurrentThreadName("shadPS4:Uffd");

        auto regions = Core::Memory::Instance()->GetAddressSpace().GetUsableRegions();
        for (auto& region : regions) {
            OnMap(region.lower(), region.upper());
        }
        LOG_INFO(Common_Memory, "registered reserved memory with userfaultfd");

        while (!token.stop_requested()) {
            pollfd pollfd;
            pollfd.fd = uffd;
            pollfd.events = POLLIN;

            // Block until the descriptor is ready for data reads.
            const int pollres = poll(&pollfd, 1, -1);
            switch (pollres) {
            case -1:
                perror("Poll userfaultfd");
                continue;
                break;
            case 0:
                continue;
            case 1:
                break;
            default:
                UNREACHABLE_MSG("Unexpected number of descriptors {} out of poll", pollres);
            }

            // We don't want an error condition to have occured.
            ASSERT_MSG(!(pollfd.revents & POLLERR), "POLLERR on userfaultfd");

            // We waited until there is data to read, we don't care about anything else.
            if (!(pollfd.revents & POLLIN)) {
                continue;
            }

            // Read message from kernel.
            uffd_msg msg;
            const int readret = read(uffd, &msg, sizeof(msg));
            if (readret == -1) {
                if (errno == EAGAIN || errno == EWOULDBLOCK) {
                    continue;
                }
                LOG_ERROR(Common_Memory, "Unexpected result of uffd read: {}",
                          Common::GetLastErrorMsg());
                break;
            }
            ASSERT_MSG(readret == sizeof(msg), "Unexpected short read, exiting");
            ASSERT(msg.arg.pagefault.flags & UFFD_PAGEFAULT_FLAG_WP);

            // Notify rasterizer about the fault.
            const VAddr addr = msg.arg.pagefault.address;
            const auto ptid = msg.arg.pagefault.feat.ptid;
            rasterizer->InvalidateMemory(addr, 1,
                                         ptid == rasterizer->GetGpuCommandProcessorThreadId());

            // Some calls to InvalidateMemory never reach the UFFDIO_WRITEPROTECT ioctl in
            // ::Protect, therefore we use MODE_DONTWAKE and wake the thread with UFFDIO_WAKE here
            uffdio_range wake;
            wake.start = msg.arg.pagefault.address;
            wake.len = PageManager::PM_PAGE_SIZE;
            const int ret = ioctl(uffd, UFFDIO_WAKE, &wake);
            ASSERT_MSG(ret != -1, "Waking thread {} failed with: {}", ptid,
                       Common::GetLastErrorMsg());
        }
    }
};
#endif // __linux__

struct SignalImpl : public PageManager::Impl {
    SignalImpl(Vulkan::Rasterizer* rasterizer_) : Impl() {
        rasterizer = rasterizer_;
        instance = this;
        Shader::SetSrtCleanLoad(&SrtWalkerCleanLoad);

        // Should be called first.
        constexpr auto priority = std::numeric_limits<u32>::min();
        Core::Signals::Instance()->RegisterAccessViolationHandler(GuestFaultSignalHandler,
                                                                  priority);
    }

    /// SHADPS4_PERF_STATS: time spent changing protections, all threads and command processor thread only
    /// (re-protection after uploads runs there)
    struct ProtectTimer {
        bool active = Common::PerfStats::Enabled();
        std::chrono::steady_clock::time_point start =
            active ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        ~ProtectTimer() {
            if (!active) {
                return;
            }
            using namespace Common::PerfStats;
            const u64 ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                               std::chrono::steady_clock::now() - start)
                               .count();
            Add(Id::ProtectNs, ns);
            if (IsGpuThread()) {
                Add(Id::ProtectGpuThread);
                Add(Id::ProtectGpuThreadNs, ns);
            }
        }
    };

    /// SHADPS4_PROTECT_CHECK=<start>-<end> tracks expected page protection and reports OS changes made behind
    /// the tracker's back, including those caused by Windows placeholder splits
    void CheckExternalChanges(VAddr address, size_t size, Core::MemoryPermission perms) {
        static const std::pair<VAddr, VAddr> range = [] {
            std::pair<VAddr, VAddr> value{};
            if (const char* env = std::getenv("SHADPS4_PROTECT_CHECK")) {
                char* end = nullptr;
                value.first = std::strtoull(env, &end, 16);
                value.second = end && *end == '-' ? std::strtoull(end + 1, nullptr, 16) : 0;
            }
            return value;
        }();
        if (range.second == 0 || address >= range.second || address + size <= range.first) {
            return;
        }
#ifdef _WIN32
        static std::mutex check_mutex;
        static std::unordered_map<VAddr, DWORD> expected;
        static u32 reports = 0;
        std::scoped_lock lk{check_mutex};
        const VAddr begin = std::max(address, range.first);
        const VAddr end = std::min<VAddr>(address + size, range.second);
        const DWORD applied = True(perms & Core::MemoryPermission::Write)  ? PAGE_READWRITE
                              : True(perms & Core::MemoryPermission::Read) ? PAGE_READONLY
                                                                           : PAGE_NOACCESS;
        for (VAddr page = PageManager::GetPageAddr(begin); page < end;
             page += PageManager::PM_PAGE_SIZE) {
            MEMORY_BASIC_INFORMATION mbi{};
            if (!VirtualQuery(reinterpret_cast<const void*>(page), &mbi, sizeof(mbi))) {
                continue;
            }
            const auto it = expected.find(page);
            if (it != expected.end() && it->second != mbi.Protect && reports < 200) {
                ++reports;
                const PageState* state = cached_pages.find(page >> PageManager::PM_PAGE_BITS);
                LOG_WARNING(Render,
                            "Protect check {:#x}: OS protection {:#x}, tracker applied {:#x} "
                            "(watchers r={} w={}), now applying {:#x}",
                            page, mbi.Protect, it->second, state ? state->num_read_watchers : 0,
                            state ? state->num_write_watchers : 0, applied);
            }
            expected[page] = applied;
        }
#endif
    }

    void Protect(VAddr address, size_t size, Core::MemoryPermission perms) override {
        RENDERER_TRACE;
        auto* memory = Core::Memory::Instance();
        auto& impl = memory->GetAddressSpace();
        CheckExternalChanges(address, size, perms);
        Common::PerfStats::Add(Common::PerfStats::Id::Protects);
        const ProtectTimer perf_timer;
        ASSERT_MSG(perms != Core::MemoryPermission::Write,
                   "Attempted to protect region as write-only which is not a valid permission");
        // Leave guest stack pages writable; the OS cannot dispatch a fault raised while pushing to a protected
        // stack page, so skip tracking there
        if (True(perms & Core::MemoryPermission::Write) ||
            !memory->OverlapsStackRange(address, size)) {
            impl.Protect(address, size, perms);
            return;
        }
        for (const auto& [sub_address, sub_size] : memory->SubtractStackRanges(address, size)) {
            impl.Protect(sub_address, sub_size, perms);
        }
    }

    static bool GuestFaultSignalHandler(void* context, void* fault_address) {
        const auto addr = reinterpret_cast<VAddr>(fault_address);
        const auto is_gpu_thread =
            std::this_thread::get_id() == rasterizer->GetGpuCommandProcessorThread();
        const bool is_write = Common::IsWriteError(context);
        const bool perf = Common::PerfStats::Enabled();
        const auto start =
            perf ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        if (!is_write && is_gpu_thread && EmulatorSettings.IsSrtWalkerCleanReads() &&
            TryCleanWalkerRead(context, addr)) {
            if (perf) {
                using namespace Common::PerfStats;
                Add(Id::SrtCleanReads);
                Add(Id::SrtCleanReadNs, std::chrono::duration_cast<std::chrono::nanoseconds>(
                                            std::chrono::steady_clock::now() - start)
                                            .count());
            }
            return true;
        }
        // The probe stays within the faulting page: the next one may not be mapped for the GPU (#5150)
        const auto size = std::min<u64>(8, PageManager::GetNextPageAddr(addr) - addr);
        bool handled = is_write ? rasterizer->InvalidateMemory(addr, size, is_gpu_thread)
                                : rasterizer->ReadMemory(addr, size, is_gpu_thread);
        if (handled && EmulatorSettings.IsPreserveSplitProtection()) {
            const Common::PerfStats::ScopedTimer repair_timer{
                Common::PerfStats::Id::FaultRepairCalls, Common::PerfStats::Id::FaultRepairNs};
            handled = instance->RepairStaleProtection(addr, is_write);
        }
        if (perf && handled) {
            using namespace Common::PerfStats;
            const u64 ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                               std::chrono::steady_clock::now() - start)
                               .count();
            Add(is_write ? Id::FaultWrite : Id::FaultRead);
            Add(Id::FaultNs, ns);
            RecordFaultRegion(addr, is_write);
            if (IsGpuThread()) {
                Add(Id::FaultGpuThread);
                Add(Id::FaultGpuThreadNs, ns);
            } else if (is_write) {
                Add(Id::FaultWriteGuestNs, ns);
            }
        }
        return handled;
    }

    /// Completes clean SRT loads from guest backing memory without a GPU readback
    static bool TryCleanWalkerRead(void* context, VAddr addr) {
        const auto load = Shader::DecodeSrtLoad(context);
        if (!load || addr < load->address || addr >= load->address + load->size) {
            return false;
        }
        u64 value = 0;
        if (!rasterizer->ReadCleanMemory(load->address, &value, load->size)) {
            return false;
        }
        Shader::CompleteSrtLoad(context, *load, value);
        // Let later walker loads try the clean path first
        Shader::MarkSrtCleanPages(load->address, load->size);
        ReportCleanRead(load->address);
        return true;
    }

    /// Clean-load callback used by generated walkers before the normal fault path
    static bool PS4_SYSV_ABI SrtWalkerCleanLoad(u64 address, u32 size, u64* out) {
        if (!EmulatorSettings.IsSrtWalkerCleanReads() || !Common::PerfStats::IsGpuThread()) {
            return false;
        }
        u64 value = 0;
        if (!rasterizer->ReadCleanMemory(address, &value, size)) {
            return false;
        }
        *out = value;
        Common::PerfStats::Add(Common::PerfStats::Id::SrtCleanLoads);
        return true;
    }

    static void ReportCleanRead(VAddr addr) {
        // Command processor thread only
        static u64 total = 0;
        static u64 last_minute = 0;
        static auto last_report = std::chrono::steady_clock::now();
        ++total;
        ++last_minute;
        const auto now = std::chrono::steady_clock::now();
        if (total == 1) {
            LOG_WARNING(Render,
                        "Workaround srt_walker_clean_reads: SRT walker read at {:#x} served from "
                        "guest memory without a readback",
                        addr);
            last_report = now;
            last_minute = 0;
        } else if (now - last_report >= std::chrono::minutes{1}) {
            LOG_WARNING(Render,
                        "Workaround srt_walker_clean_reads: {} walker read faults served without a "
                        "readback in the last minute (total {}, last at {:#x})",
                        last_minute, total, addr);
            last_report = now;
            last_minute = 0;
        }
    }

    /// Per-game preserve_split_protection safety net: a fault the tracker handled must leave the page
    /// accessible, or the faulting instruction retries forever (a silent hang). When the tracker's own page
    /// state allows the access but the OS still denies it, the tracker's protection is applied again. A
    /// denial the tracker agrees with is left alone: another thread is still releasing the page (texture
    /// cache, a split remapping it) and the retry succeeds once it is done
    bool RepairStaleProtection(VAddr addr, bool is_write) {
#ifdef _WIN32
        MEMORY_BASIC_INFORMATION mbi{};
        if (!VirtualQuery(reinterpret_cast<const void*>(addr), &mbi, sizeof(mbi)) ||
            mbi.State != MEM_COMMIT) {
            return true;
        }
        constexpr DWORD writable = PAGE_READWRITE | PAGE_EXECUTE_READWRITE;
        constexpr DWORD readable = writable | PAGE_READONLY | PAGE_EXECUTE_READ;
        if ((mbi.Protect & (is_write ? writable : readable)) != 0) {
            return true;
        }
        const VAddr page = addr & ~(PageManager::PM_PAGE_SIZE - 1);
        const u64 page_index = page >> PageManager::PM_PAGE_BITS;
        PageState* state = cached_pages.find(page_index);
        if (!state) {
            return true;
        }
        Core::MemoryPermission perms;
        {
            LockPage(page_index);
            perms = state->Perms();
            if (!True(perms & (is_write ? Core::MemoryPermission::Write
                                        : Core::MemoryPermission::Read))) {
                UnlockPage(page_index);
                return true;
            }
            Protect(page, PageManager::PM_PAGE_SIZE, perms);
            UnlockPage(page_index);
        }
        static std::atomic<u64> repairs{0};
        if (const u64 count = repairs.fetch_add(1) + 1; count == 1 || count % 1000 == 0) {
            LOG_WARNING(Render,
                        "Workaround preserve_split_protection: re-applied tracker protection {} "
                        "on {:#x} after a handled {} fault left it at OS {:#x} (#{})",
                        static_cast<u32>(perms), page, is_write ? "write" : "read", mbi.Protect,
                        count);
        }
#endif
        return true;
    }

    inline static SignalImpl* instance{};
};

PageManager::PageManager(Vulkan::Rasterizer* rasterizer_) {
#ifdef __linux__
    if (EmulatorSettings.IsUserfaultfdTracking()) {
        try {
            impl = std::make_unique<UffdImpl>(rasterizer_);
            LOG_INFO(Config, "Memory tracking method: userfaultfd");
            return;
        } catch (const std::runtime_error& e) {
            // if uffd is unsupported, falls back to SignalImpl
        }
    }
    LOG_INFO(Config, "Memory tracking method: signals");
#endif
    impl = std::make_unique<SignalImpl>(rasterizer_);
}

PageManager::~PageManager() = default;

std::pair<u32, u32> PageManager::GetWatchers(VAddr addr) const {
    const auto* state = impl->cached_pages.find(addr >> PM_PAGE_BITS);
    return state ? std::pair<u32, u32>{state->num_read_watchers, state->num_write_watchers}
                 : std::pair<u32, u32>{};
}

void PageManager::OnGpuMap(VAddr address, size_t size) {
    impl->OnMap(address, size);
}

void PageManager::OnGpuUnmap(VAddr address, size_t size) {
    impl->OnUnmap(address, size);
}

void PageManager::UpdatePageWatchers(VAddr addr, u64 size, PageOp write_op) const {
    impl->UpdatePageWatchers(addr, size, write_op);
}

void PageManager::UpdatePageWatchersForRegion(VAddr base_addr, const Bounds& bounds,
                                              const RegionBits& write_mask,
                                              const RegionBits& read_mask, PageOp write_op,
                                              PageOp read_op) const {
    impl->UpdatePageWatchersForRegion(base_addr, bounds, write_mask, read_mask, write_op, read_op);
}

} // namespace VideoCore
