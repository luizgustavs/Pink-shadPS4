// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <boost/preprocessor/stringize.hpp>

#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_map>

#include "common/assert.h"
#include "common/cp_profiler.h"
#include "common/debug.h"
#include "common/guest_write_journal.h"
#include "common/logging/events.h"
#include "common/perf_stats.h"
#include "common/polyfill_thread.h"
#include "common/thread.h"
#include "common/unique_function.h"
#include "core/debug_state.h"
#include "core/emulator_settings.h"
#include "core/libraries/kernel/process.h"
#include "core/libraries/videoout/driver.h"
#include "core/memory.h"
#include "core/platform.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/amdgpu/pm4_cmds.h"
#include "video_core/renderdoc.h"
#include "video_core/renderer_vulkan/vk_draw_trace.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"

namespace AmdGpu {

/// IT_COND_EXEC: true when the packets that follow must be skipped. The condition address comes from the
/// command buffer; a garbage one must not crash the host, so the packets then execute, like a condition that
/// always passes
static bool CondExecSkips(const PM4CmdCondExec* cond_exec, const char* queue) {
    if (cond_exec->command.Value() != 0) {
        LOG_WARNING(Render, "IT_COND_EXEC used a reserved command ({})", queue);
    }
    const bool* cond_addr = cond_exec->Address();
    if (!Core::Memory::Instance()->IsMappedLocked(reinterpret_cast<VAddr>(cond_addr),
                                                  sizeof(bool))) {
        LOG_WARNING(Render, "IT_COND_EXEC with unmapped address {:#x}, executing ({})",
                    reinterpret_cast<VAddr>(cond_addr), queue);
        return false;
    }
    return *cond_addr == false;
}

/// SHADPS4_DRAW_TRACE: the synchronization packets of the traced frames, with their queue, so a trace shows
/// what each dispatch waited for (or did not wait for) before it ran. The next step on the SotC TDR root
/// cause (§17 of the port guide) is a dispatch that runs before the game wrote its constant buffer
static bool TraceSync() {
    return Vulkan::DrawTrace::Instance().Active();
}

template <typename... Args>
static void LogSync(std::string_view queue, fmt::format_string<Args...> format, Args&&... args) {
    LOG_WARNING(Render, "DrawTrace f={} {} {}", Vulkan::DrawTrace::Instance().Frame(), queue,
                fmt::format(format, std::forward<Args>(args)...));
}

/// Delay gfx EOP labels with a timer as a temporary workaround for the SotC rock flicker
/// The game starts CPU visibility jobs after reading the previous BOTTOM_OF_PIPE label for their
/// memory
/// Writing that label as soon as the packet is read can release the jobs too early and put their
/// output in another frame
/// The timer adds a fixed delay without waiting for the GPU or submitting work, and 2000 us helped
/// in earlier tests
/// It can still publish the label too early when the GPU falls behind, and readback_ahead can bring
/// the flicker back
/// A complete fix still needs to publish labels after the GPU finishes the commands before them
/// Treat the delay as a timing adjustment rather than proof that the preceding work has completed,
/// because the timer cannot see how far the GPU has progressed
namespace DelayedLabels {
namespace {
struct Pending {
    u64 value;
    u32 bytes;
    u64 seq;
};
std::mutex pending_mutex;
std::unordered_map<VAddr, Pending> pending;
u64 next_seq{};
std::atomic<u32> pending_count{};

struct Item {
    u64 due_ns;
    Common::UniqueFunction<void> func;
};
/// Keep this timer state alive until process exit because the detached worker may still be waiting
/// on it during static destruction
struct TimerState {
    std::mutex mutex;
    std::condition_variable cv;
    std::deque<Item> items;
};
TimerState& Timer() {
    static auto* const state = new TimerState;
    return *state;
}

u64 NowNs() {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count());
}

void Worker() {
    Common::SetCurrentThreadName("shadPS4:LabelTimer");
    auto& [timer_mutex, timer_cv, items] = Timer();
    std::unique_lock lk{timer_mutex};
    while (true) {
        timer_cv.wait(lk, [&] { return !items.empty(); });
        const u64 due = items.front().due_ns;
        const u64 now = NowNs();
        if (now < due) {
            // The OS timer has roughly 1 ms resolution, so sleep for most of the remaining delay
            // and yield near the deadline
            if (due - now > 1'500'000) {
                timer_cv.wait_for(lk, std::chrono::nanoseconds{due - now - 1'000'000});
            } else {
                lk.unlock();
                std::this_thread::yield();
                lk.lock();
            }
            continue;
        }
        // Every label uses the same delay and enters in command order, so the front of the queue is
        // also the next label due
        auto func = std::move(items.front().func);
        items.pop_front();
        lk.unlock();
        func();
        lk.lock();
    }
}
} // Anonymous namespace

/// Return the configured EOP label delay in microseconds, or 0 when the temporary timing workaround
/// is disabled
u32 DelayUs() {
    static const u32 delay_us = [] {
        const u32 value = EmulatorSettings.GetEopLabelDelayUs();
        if (value != 0) {
            LOG_WARNING(Render, "Workaround eop_label_delay_us enabled: gfx EOP labels {} us after "
                                "the packet (crude timing patch)",
                        value);
            if (EmulatorSettings.IsReadbackAhead()) {
                LOG_WARNING(Render, "eop_label_delay_us with readback_ahead: the flicker it hides "
                                    "comes back with readback_ahead");
            }
        }
        return value;
    }();
    return delay_us;
}

/// Keep the label waiting for its timer together with the value that command processor waits should
/// already see
u64 AddPending(VAddr addr, u64 value, u32 bytes) {
    std::scoped_lock lock{pending_mutex};
    pending[addr] = Pending{value, bytes, ++next_seq};
    pending_count.store(static_cast<u32>(pending.size()), std::memory_order_release);
    return next_seq;
}

void RemovePending(VAddr addr, u64 seq) {
    std::scoped_lock lock{pending_mutex};
    const auto it = pending.find(addr);
    if (it != pending.end() && it->second.seq == seq) {
        pending.erase(it);
        pending_count.store(static_cast<u32>(pending.size()), std::memory_order_release);
    }
}

std::optional<u32> PendingDword(VAddr addr) {
    if (pending_count.load(std::memory_order_acquire) == 0) {
        return std::nullopt;
    }
    std::scoped_lock lock{pending_mutex};
    if (const auto it = pending.find(addr); it != pending.end()) {
        return static_cast<u32>(it->second.value);
    }
    if (const auto it = pending.find(addr - 4); it != pending.end() && it->second.bytes == 8) {
        return static_cast<u32>(it->second.value >> 32);
    }
    return std::nullopt;
}

void Push(u64 delay_ns, Common::UniqueFunction<void>&& func) {
    static std::once_flag started;
    std::call_once(started, [] { std::thread{Worker}.detach(); });
    auto& timer = Timer();
    {
        std::scoped_lock lk{timer.mutex};
        timer.items.push_back(Item{NowNs() + delay_ns, std::move(func)});
    }
    timer.cv.notify_one();
}
} // namespace DelayedLabels

/// Check WAIT_REG_MEM using the value visible to the command processor in command order
/// With eop_label_delay_us, a pending label already has its queued value for these waits even
/// before the timer writes guest memory
static bool CpWaitPasses(const PM4CmdWaitRegMem* wait, std::span<const u32> regs) {
    if (DelayedLabels::DelayUs() != 0 &&
        wait->mem_space.Value() == PM4CmdWaitRegMem::MemSpace::Memory) {
        if (const auto value = DelayedLabels::PendingDword(wait->Address<VAddr>())) {
            return wait->TestValue(*value);
        }
    }
    return wait->Test(regs);
}

namespace CpHistory {

namespace {
struct Entry {
    u64 ns;
    const void* header; // packet header, or the first dword of a submit
    std::array<u32, 4> words;
    u32 size_dw; // submits only
    s16 queue;
    u16 opcode; // SubmitOpcode for a submit
};
constexpr u16 SubmitOpcode = 0xFFFF;
constexpr u32 NumEntries = 1024;
std::array<Entry, NumEntries> entries{};
std::atomic<u64> next_entry{0};

u64 NowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

std::string QueueName(s16 queue) {
    return queue == Queue::Ce ? "CE" : queue == Queue::De ? "DE" : fmt::format("ASC{}", queue);
}
} // Anonymous namespace

const bool enabled = [] {
    const char* value = std::getenv("SHADPS4_CBUF_PROBE");
    return value && *value && std::string_view{value} != "0";
}();

void RecordPacket(s16 queue, std::span<const u32> packet) {
    // Packets come from the command processor thread only; submits race with it, which is fine for
    // a diagnostic (a torn entry shows up as garbage in the dump)
    auto& entry = entries[next_entry.fetch_add(1, std::memory_order_relaxed) % NumEntries];
    const auto* pm4 = reinterpret_cast<const PM4Header*>(packet.data());
    const u32* body = packet.data() + 1;
    // Keep truncated packets inside the submission
    const u32 count = std::min<u32>({pm4->type3.NumWords(), static_cast<u32>(packet.size() - 1), 4});
    entry.ns = NowNs();
    entry.header = packet.data();
    entry.words = {};
    std::memcpy(entry.words.data(), body, count * sizeof(u32));
    entry.size_dw = 0;
    entry.queue = queue;
    entry.opcode = static_cast<u16>(pm4->type3.opcode.Value());
}

void RecordSubmit(s16 queue, std::span<const u32> commands) {
    auto& entry = entries[next_entry.fetch_add(1, std::memory_order_relaxed) % NumEntries];
    entry.ns = NowNs();
    entry.header = commands.data();
    entry.words = {};
    entry.size_dw = static_cast<u32>(commands.size());
    entry.queue = queue;
    entry.opcode = SubmitOpcode;
}

void Dump(u32 count) {
    const u64 end = next_entry.load(std::memory_order_relaxed);
    const u64 begin = end - std::min<u64>({end, count, NumEntries});
    const u64 now = NowNs();
    for (u64 i = begin; i < end; ++i) {
        const Entry entry = entries[i % NumEntries];
        const double ago_ms = static_cast<double>(now - entry.ns) / 1e6;
        if (entry.opcode == SubmitOpcode) {
            LOG_WARNING(Render, "CbufProbe hist -{:.3f}ms {} SUBMIT {} dwords at {}", ago_ms,
                        QueueName(entry.queue), entry.size_dw, entry.header);
            continue;
        }
        const auto opcode = static_cast<PM4ItOpcode>(entry.opcode);
        // enum_name is empty above 127 (the CE opcodes), so the raw opcode goes along
        LOG_WARNING(Render,
                    "CbufProbe hist -{:.3f}ms {} {}({:#04x}) at {} [{:08x} {:08x} {:08x} {:08x}]",
                    ago_ms, QueueName(entry.queue), magic_enum::enum_name(opcode), entry.opcode,
                    entry.header, entry.words[0], entry.words[1], entry.words[2], entry.words[3]);
    }
}

} // namespace CpHistory

static const char* dcb_task_name{"DCB_TASK"};
static const char* ccb_task_name{"CCB_TASK"};

#define MAX_NAMES 56
static_assert(Liverpool::NumComputeRings <= MAX_NAMES);

#define NAME_NUM(z, n, name) BOOST_PP_STRINGIZE(name) BOOST_PP_STRINGIZE(n),
#define NAME_ARRAY(name, num) {BOOST_PP_REPEAT(num, NAME_NUM, name)}

static const char* acb_task_name[] = NAME_ARRAY(ACB_TASK, MAX_NAMES);

#define YIELD(name)                                                                                \
    FIBER_EXIT;                                                                                    \
    co_yield {};                                                                                   \
    FIBER_ENTER(name);

#define YIELD_CE() YIELD(ccb_task_name)
#define YIELD_GFX() YIELD(dcb_task_name)
#define YIELD_ASC(id) YIELD(acb_task_name[id])

#define RESUME(task, name)                                                                         \
    FIBER_EXIT;                                                                                    \
    task.handle.resume();                                                                          \
    FIBER_ENTER(name);

#define RESUME_CE(task) RESUME(task, ccb_task_name)
#define RESUME_GFX(task) RESUME(task, dcb_task_name)
#define RESUME_ASC(task, id) RESUME(task, acb_task_name[id])

std::array<u8, 48_KB> Liverpool::ConstantEngine::constants_heap;

static std::span<const u32> NextPacket(std::span<const u32> span, size_t offset) {
    if (offset > span.size()) {
        LOG_ERROR(
            Lib_GnmDriver,
            ": packet length exceeds remaining submission size. Packet dword count={}, remaining "
            "submission dwords={}",
            offset, span.size());
        // Return empty subspan so check for next packet bails out
        return {};
    }

    return span.subspan(offset);
}

Liverpool::Liverpool() : guest_markers_enabled{EmulatorSettings.IsVkGuestMarkersEnabled()} {
    num_counter_pairs = Libraries::Kernel::sceKernelIsNeoMode() ? 16 : 8;
    process_thread = std::jthread{std::bind_front(&Liverpool::Process, this)};
}

Liverpool::~Liverpool() {
    process_thread.request_stop();
    process_thread.join();
}

void Liverpool::ProcessCommands() {
    // Process incoming commands with high priority
    while (num_commands) {
        Common::UniqueFunction<void> callback{};
        {
            std::scoped_lock lk{submit_mutex};
            callback = std::move(command_queue.front());
            command_queue.pop();
            --num_commands;
        }
        callback();
    }
}

void Liverpool::Process(std::stop_token stoken) {
    Common::SetCurrentThreadName("shadPS4:GpuCommandProcessor");
    gpu_id = std::this_thread::get_id();
#ifdef __linux__
    gpu_tid = gettid();
#endif
    Common::PerfStats::MarkGpuThread();
    Common::CpProfiler::RegisterCurrentThread();

    while (!stoken.stop_requested()) {
        {
            std::unique_lock lk{submit_mutex};
            Common::CondvarWait(submit_cv, lk, stoken,
                                [this] { return num_commands || num_submits || submit_done; });
        }
        if (stoken.stop_requested()) {
            break;
        }
        const bool perf = Common::PerfStats::Enabled();
        const auto busy_start =
            perf ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};

        VideoCore::StartCapture();

        curr_qid = -1;

        while (num_submits || num_commands) {
            ProcessCommands();

            curr_qid = (curr_qid + 1) % num_mapped_queues;

            auto& queue = mapped_queues[curr_qid];

            Task::Handle task{};
            {
                std::scoped_lock lock{queue.m_access};
                if (queue.submits.empty()) {
                    continue;
                }
                task = queue.submits.front();
            }
            task.resume();

            if (task.done()) {
                task.destroy();

                std::scoped_lock lock{queue.m_access};
                queue.submits.pop();
                if (!queue.submit_times.empty()) {
                    queue.submit_times.pop();
                }

                --num_submits;
                std::scoped_lock lock2{submit_mutex};
                submit_cv.notify_all();
            }
        }

        if (submit_done) {
            VideoCore::EndCapture();
            if (rasterizer) {
                rasterizer->OnSubmit();
                rasterizer->Flush();
            }
            submit_done = false;
        }
        if (perf) {
            Common::PerfStats::Add(Common::PerfStats::Id::CpBusyNs,
                                   std::chrono::duration_cast<std::chrono::nanoseconds>(
                                       std::chrono::steady_clock::now() - busy_start)
                                       .count());
        }

        Platform::IrqC::Instance()->Signal(Platform::InterruptId::GpuIdle);
    }
}

Liverpool::Task Liverpool::ProcessCeUpdate(std::span<const u32> ccb) {
    FIBER_ENTER(ccb_task_name);

    while (!ccb.empty()) {
        ProcessCommands();

        const auto* header = reinterpret_cast<const PM4Header*>(ccb.data());
        const u32 type = header->type;
        if (type != 3) {
            // No other types of packets were spotted so far
            UNREACHABLE_MSG("Invalid PM4 type {}", type);
        }

        const PM4ItOpcode opcode = header->type3.opcode;
        const auto* it_body = reinterpret_cast<const u32*>(header) + 1;
        if (CpHistory::Enabled()) {
            CpHistory::RecordPacket(CpHistory::Queue::Ce, ccb);
        }
        switch (opcode) {
        case PM4ItOpcode::Nop: {
            // const auto* nop = reinterpret_cast<const PM4CmdNop*>(header);
            break;
        }
        case PM4ItOpcode::WriteConstRam: {
            const auto* write_const = reinterpret_cast<const PM4WriteConstRam*>(header);
            memcpy(cblock.constants_heap.data() + write_const->Offset(), &write_const->data,
                   write_const->Size());
            break;
        }
        case PM4ItOpcode::DumpConstRam: {
            const auto* dump_const = reinterpret_cast<const PM4DumpConstRam*>(header);
            const u32 size = dump_const->Size();
            if (rasterizer) {
                auto& buffer_cache = rasterizer->GetBufferCache();
                if (buffer_cache.IsRegionInSyncBatch(dump_const->Address<VAddr>(), size)) {
                    buffer_cache.FlushSyncBatch();
                }
            }
            Common::GuestWriteJournal::Record(Common::GuestWriteJournal::Source::DumpConstRam,
                                              dump_const->Address<VAddr>(), size,
                                              cblock.constants_heap.data() + dump_const->Offset());
            memcpy(dump_const->Address<void*>(),
                   cblock.constants_heap.data() + dump_const->Offset(), size);
            break;
        }
        case PM4ItOpcode::IncrementCeCounter: {
            ++cblock.ce_count;
            break;
        }
        case PM4ItOpcode::WaitOnDeCounterDiff: {
            const auto diff = it_body[0];
            while ((cblock.de_count - cblock.ce_count) >= diff) {
                YIELD_CE();
            }
            break;
        }
        case PM4ItOpcode::IndirectBufferConst: {
            const auto* indirect_buffer = reinterpret_cast<const PM4CmdIndirectBuffer*>(header);
            auto task =
                ProcessCeUpdate({indirect_buffer->Address<const u32>(), indirect_buffer->ib_size});
            RESUME_CE(task);

            while (!task.handle.done()) {
                YIELD_CE();
                RESUME_CE(task);
            }
            break;
        }
        case PM4ItOpcode::PredExec: {
            // Executed unconditionally, like a predicate that always passes (as in the DE)
            LOG_DEBUG(Render, "IT_PRED_EXEC ignored (CE)");
            break;
        }
        case PM4ItOpcode::CondExec: {
            const auto* cond_exec = reinterpret_cast<const PM4CmdCondExec*>(header);
            if (CondExecSkips(cond_exec, "CE")) {
                ccb = NextPacket(ccb, header->type3.NumWords() + 1 + cond_exec->exec_count.Value());
                continue;
            }
            break;
        }
        default:
            const u32 count = header->type3.NumWords();
            UNREACHABLE_MSG("Unknown PM4 type 3 opcode {:#x} with count {}",
                            static_cast<u32>(opcode), count);
        }
        ccb = NextPacket(ccb, header->type3.NumWords() + 1);
    }

    FIBER_EXIT;
}

Liverpool::Task Liverpool::ProcessGraphics(std::span<const u32> dcb, std::span<const u32> ccb) {
    FIBER_ENTER(dcb_task_name);

    cblock.Reset();

    // TODO: potentially, ASCs also can depend on CE and in this case the
    // CE task should be moved into more global scope
    Task ce_task{};

    if (!ccb.empty()) {
        // In case of CCB provided kick off CE asap to have the constant heap ready to use
        ce_task = ProcessCeUpdate(ccb);
        RESUME_GFX(ce_task);
    }

    const auto base_addr = reinterpret_cast<uintptr_t>(dcb.data());
    while (!dcb.empty()) {
        ProcessCommands();

        const auto* header = reinterpret_cast<const PM4Header*>(dcb.data());
        const u32 type = header->type;

        switch (type) {
        default:
            UNREACHABLE_MSG("Wrong PM4 type {}", type);
            break;
        case 0:
            UNREACHABLE_MSG("Unimplemented PM4 type 0, base reg: {}, size: {}",
                            header->type0.base.Value(), header->type0.NumWords());
            break;
        case 2:
            // Type-2 packet are used for padding purposes
            dcb = NextPacket(dcb, 1);
            continue;
        case 3:
            const u32 count = header->type3.NumWords();
            const PM4ItOpcode opcode = header->type3.opcode;
            if (CpHistory::Enabled()) {
                CpHistory::RecordPacket(CpHistory::Queue::De, dcb);
            }
            switch (opcode) {
            case PM4ItOpcode::Nop: {
                const auto* nop = reinterpret_cast<const PM4CmdNop*>(header);
                if (nop->header.count.Value() == 0) {
                    break;
                }

                switch (nop->data_block[0]) {
                case PM4CmdNop::PayloadType::PatchedFlip: {
                    // There is no evidence that GPU CP drives flip events by parsing
                    // special NOP packets. For convenience lets assume that it does.
                    Platform::IrqC::Instance()->Signal(Platform::InterruptId::GfxFlip);
                    break;
                }
                case PM4CmdNop::PayloadType::DebugMarkerPush: {
                    if (guest_markers_enabled) {
                        const auto marker_sz = nop->header.count.Value() * 2;
                        const std::string_view label{
                            reinterpret_cast<const char*>(&nop->data_block[1]), marker_sz};
                        rasterizer->ScopeMarkerBegin(label, true);
                    }
                    break;
                }
                case PM4CmdNop::PayloadType::DebugColorMarkerPush: {
                    if (guest_markers_enabled) {
                        const auto marker_sz = nop->header.count.Value() * 2;
                        const std::string_view label{
                            reinterpret_cast<const char*>(&nop->data_block[1]), marker_sz};
                        const u32 color = *reinterpret_cast<const u32*>(
                            reinterpret_cast<const u8*>(&nop->data_block[1]) + marker_sz);
                        rasterizer->ScopedMarkerInsertColor(label, color, true);
                    }
                    break;
                }
                case PM4CmdNop::PayloadType::DebugMarkerPop: {
                    if (guest_markers_enabled) {
                        rasterizer->ScopeMarkerEnd(true);
                    }
                    break;
                }
                default:
                    break;
                }
                break;
            }
            case PM4ItOpcode::ContextControl: {
                break;
            }
            case PM4ItOpcode::ClearState: {
                regs.SetDefaults();
                break;
            }
            case PM4ItOpcode::SetConfigReg: {
                const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
                const auto reg_addr = Regs::ConfigRegWordOffset + set_data->reg_offset;
                const auto* payload = reinterpret_cast<const u32*>(header + 2);
                std::memcpy(&regs.reg_array[reg_addr], payload, (count - 1) * sizeof(u32));
                break;
            }
            case PM4ItOpcode::SetContextReg: {
                const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
                const auto reg_addr = Regs::ContextRegWordOffset + set_data->reg_offset;
                const auto* payload = reinterpret_cast<const u32*>(header + 2);

                std::memcpy(&regs.reg_array[reg_addr], payload, (count - 1) * sizeof(u32));

                // In the case of HW, render target memory has alignment as color block operates on
                // tiles. There is no information of actual resource extents stored in CB context
                // regs, so any deduction of it from slices/pitch will lead to a larger surface
                // created. The same applies to the depth targets. Fortunately, the guest always
                // sends a trailing NOP packet right after the context regs setup, so we can use the
                // heuristic below and extract the hint to determine actual resource dims.

                switch (reg_addr) {
                case ContextRegs::CbColor0Base:
                case ContextRegs::CbColor1Base:
                case ContextRegs::CbColor2Base:
                case ContextRegs::CbColor3Base:
                case ContextRegs::CbColor4Base:
                case ContextRegs::CbColor5Base:
                case ContextRegs::CbColor6Base:
                case ContextRegs::CbColor7Base: {
                    const auto col_buf_id = (reg_addr - ContextRegs::CbColor0Base) /
                                            (ContextRegs::CbColor1Base - ContextRegs::CbColor0Base);
                    ASSERT(col_buf_id < NUM_COLOR_BUFFERS);

                    const auto nop_offset = header->type3.count;
                    if (nop_offset == 0x0e || nop_offset == 0x0d || nop_offset == 0x0b) {
                        ASSERT_MSG(payload[nop_offset] == 0xc0001000,
                                   "NOP hint is missing in CB setup sequence");
                        last_cb_extent[col_buf_id].raw = payload[nop_offset + 1];
                    } else {
                        last_cb_extent[col_buf_id].raw = 0;
                    }
                    break;
                }
                case ContextRegs::CbColor0Cmask:
                case ContextRegs::CbColor1Cmask:
                case ContextRegs::CbColor2Cmask:
                case ContextRegs::CbColor3Cmask:
                case ContextRegs::CbColor4Cmask:
                case ContextRegs::CbColor5Cmask:
                case ContextRegs::CbColor6Cmask:
                case ContextRegs::CbColor7Cmask: {
                    const auto col_buf_id =
                        (reg_addr - ContextRegs::CbColor0Cmask) /
                        (ContextRegs::CbColor1Cmask - ContextRegs::CbColor0Cmask);
                    ASSERT(col_buf_id < NUM_COLOR_BUFFERS);

                    const auto nop_offset = header->type3.count;
                    if (nop_offset == 0x04) {
                        ASSERT_MSG(payload[nop_offset] == 0xc0001000,
                                   "NOP hint is missing in CB setup sequence");
                        last_cb_extent[col_buf_id].raw = payload[nop_offset + 1];
                    }
                    break;
                }
                case ContextRegs::DbZInfo: {
                    if (header->type3.count == 8) {
                        ASSERT_MSG(payload[20] == 0xc0001000,
                                   "NOP hint is missing in DB setup sequence");
                        last_db_extent.raw = payload[21];
                    } else {
                        last_db_extent.raw = 0;
                    }
                    break;
                }
                default:
                    break;
                }
                break;
            }
            case PM4ItOpcode::SetShReg: {
                const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
                const auto set_size = (count - 1) * sizeof(u32);

                if (set_data->reg_offset >= 0x200 &&
                    set_data->reg_offset <= (0x200 + sizeof(ComputeProgram) / 4)) {
                    ASSERT(set_size <= sizeof(ComputeProgram));
                    auto* addr = reinterpret_cast<u32*>(&mapped_queues[GfxQueueId].cs_state) +
                                 (set_data->reg_offset - 0x200);
                    std::memcpy(addr, header + 2, set_size);
                } else {
                    std::memcpy(&regs.reg_array[Regs::ShRegWordOffset + set_data->reg_offset],
                                header + 2, set_size);
                }
                break;
            }
            case PM4ItOpcode::SetUconfigReg: {
                const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
                std::memcpy(&regs.reg_array[Regs::UconfigRegWordOffset + set_data->reg_offset],
                            header + 2, (count - 1) * sizeof(u32));
                break;
            }
            case PM4ItOpcode::SetPredication: {
                // The packet stores START_ADDR_LO followed by START_ADDR_HI[7:0],
                // PREDICATION_BOOL[8] and HINT[12]
                // PRED_OP[18:16] selects clear, ZPass or PrimCount, and CONTINUE[31] carries the
                // continuation flag
                // SotC only sends the clear operation here, about 15 times per frame, which needs
                // no work while predication is never enabled
                const auto* body = reinterpret_cast<const u32*>(header) + 1;
                const u32 lo = count >= 1 ? body[0] : 0;
                const u32 ctl = count >= 2 ? body[1] : 0;
                if (((ctl >> 16) & 7) == 0) {
                    break;
                }
                LOG_RENDER_PROBLEM(Render, Warning,
                                   "Unimplemented IT_SET_PREDICATION op={} bool={} hint={} "
                                   "continue={} addr={:#x}",
                                   (ctl >> 16) & 7, (ctl >> 8) & 1, (ctl >> 12) & 1, ctl >> 31,
                                   (u64{ctl & 0xff} << 32) | (lo & ~0xfu));
                break;
            }
            case PM4ItOpcode::IndexType: {
                const auto* index_type = reinterpret_cast<const PM4CmdDrawIndexType*>(header);
                regs.index_buffer_type.raw = index_type->raw;
                break;
            }
            case PM4ItOpcode::DrawIndex2: {
                const auto* draw_index = reinterpret_cast<const PM4CmdDrawIndex2*>(header);
                regs.max_index_size = draw_index->max_size;
                regs.index_base_address.base_addr_lo = draw_index->index_base_lo;
                regs.index_base_address.base_addr_hi = draw_index->index_base_hi;
                regs.num_indices = draw_index->index_count;
                regs.draw_initiator = draw_index->draw_initiator;
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (!rasterizer) {
                    break;
                }
                const auto cmd_address = reinterpret_cast<const void*>(header);
                rasterizer->ScopeMarker("gfx:{}:DrawIndex2", fmt::make_format_args(cmd_address),
                                        [&] { rasterizer->Draw(true); });
                break;
            }
            case PM4ItOpcode::DrawIndexOffset2: {
                const auto* draw_index_off =
                    reinterpret_cast<const PM4CmdDrawIndexOffset2*>(header);
                regs.max_index_size = draw_index_off->max_size;
                regs.num_indices = draw_index_off->index_count;
                regs.draw_initiator = draw_index_off->draw_initiator;
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (!rasterizer) {
                    break;
                }
                const auto cmd_address = reinterpret_cast<const void*>(header);
                rasterizer->ScopeMarker(
                    "gfx:{}:DrawIndexOffset2", fmt::make_format_args(cmd_address),
                    [&] { rasterizer->Draw(true, draw_index_off->index_offset); });
                break;
            }
            case PM4ItOpcode::DrawIndexAuto: {
                const auto* draw_index = reinterpret_cast<const PM4CmdDrawIndexAuto*>(header);
                regs.num_indices = draw_index->index_count;
                regs.draw_initiator = draw_index->draw_initiator;
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (!rasterizer) {
                    break;
                }
                const auto cmd_address = reinterpret_cast<const void*>(header);
                rasterizer->ScopeMarker("gfx:{}:DrawIndexAuto", fmt::make_format_args(cmd_address),
                                        [&] { rasterizer->Draw(false); });
                break;
            }
            case PM4ItOpcode::DrawIndirect: {
                const auto* draw_indirect = reinterpret_cast<const PM4CmdDrawIndirect*>(header);
                const auto offset = draw_indirect->data_offset;
                const auto stride = sizeof(DrawIndirectArgs);
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (!rasterizer) {
                    break;
                }
                const auto cmd_address = reinterpret_cast<const void*>(header);
                rasterizer->ScopeMarker(
                    "gfx:{}:DrawIndirect", fmt::make_format_args(cmd_address), [&] {
                        rasterizer->DrawIndirect(false, indirect_args_addr, offset, stride, 1, 0,
                                                 draw_indirect->base_vtx_loc,
                                                 draw_indirect->start_inst_loc);
                    });
                break;
            }
            case PM4ItOpcode::DrawIndirectMulti: {
                const auto* draw_indirect =
                    reinterpret_cast<const PM4CmdDrawIndirectMulti*>(header);
                const auto offset = draw_indirect->data_offset;
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (!rasterizer) {
                    break;
                }
                const auto cmd_address = reinterpret_cast<const void*>(header);
                rasterizer->ScopeMarker(
                    "gfx:{}:DrawIndirectMulti", fmt::make_format_args(cmd_address), [&] {
                        rasterizer->DrawIndirect(false, indirect_args_addr, offset,
                                                 draw_indirect->stride, draw_indirect->count, 0,
                                                 draw_indirect->base_vtx_loc,
                                                 draw_indirect->start_inst_loc);
                    });
                break;
            }
            case PM4ItOpcode::DrawIndexIndirect: {
                const auto* draw_index_indirect =
                    reinterpret_cast<const PM4CmdDrawIndexIndirect*>(header);
                const auto offset = draw_index_indirect->data_offset;
                const auto stride = sizeof(DrawIndexedIndirectArgs);
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (!rasterizer) {
                    break;
                }
                const auto cmd_address = reinterpret_cast<const void*>(header);
                rasterizer->ScopeMarker(
                    "gfx:{}:DrawIndexIndirect", fmt::make_format_args(cmd_address), [&] {
                        rasterizer->DrawIndirect(true, indirect_args_addr, offset, stride, 1, 0,
                                                 draw_index_indirect->base_vtx_loc,
                                                 draw_index_indirect->start_inst_loc);
                    });
                break;
            }
            case PM4ItOpcode::DrawIndexIndirectMulti: {
                const auto* draw_index_indirect =
                    reinterpret_cast<const PM4CmdDrawIndexIndirectMulti*>(header);
                const auto offset = draw_index_indirect->data_offset;
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (!rasterizer) {
                    break;
                }
                const auto cmd_address = reinterpret_cast<const void*>(header);
                rasterizer->ScopeMarker(
                    "gfx:{}:DrawIndexIndirectMulti", fmt::make_format_args(cmd_address), [&] {
                        rasterizer->DrawIndirect(
                            true, indirect_args_addr, offset, draw_index_indirect->stride,
                            draw_index_indirect->count, 0, draw_index_indirect->base_vtx_loc,
                            draw_index_indirect->start_inst_loc);
                    });
                break;
            }
            case PM4ItOpcode::DrawIndexIndirectCountMulti: {
                const auto* draw_index_indirect =
                    reinterpret_cast<const PM4CmdDrawIndexIndirectCountMulti*>(header);
                const auto offset = draw_index_indirect->data_offset;
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (!rasterizer) {
                    break;
                }
                const auto cmd_address = reinterpret_cast<const void*>(header);
                rasterizer->ScopeMarker(
                    "gfx:{}:DrawIndexIndirectCountMulti", fmt::make_format_args(cmd_address), [&] {
                        rasterizer->DrawIndirect(
                            true, indirect_args_addr, offset, draw_index_indirect->stride,
                            draw_index_indirect->count,
                            draw_index_indirect->count_indirect_enable.Value()
                                ? draw_index_indirect->count_addr
                                : 0,
                            draw_index_indirect->base_vtx_loc, draw_index_indirect->start_inst_loc);
                    });
                break;
            }
            case PM4ItOpcode::DispatchDirect: {
                const auto* dispatch_direct = reinterpret_cast<const PM4CmdDispatchDirect*>(header);
                auto& cs_program = GetCsRegs();
                cs_program.dim_x = dispatch_direct->dim_x;
                cs_program.dim_y = dispatch_direct->dim_y;
                cs_program.dim_z = dispatch_direct->dim_z;
                cs_program.dispatch_initiator = dispatch_direct->dispatch_initiator;
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDumpCompute(base_addr, reinterpret_cast<uintptr_t>(header),
                                                   cs_program);
                }
                if (!rasterizer || (cs_program.dispatch_initiator & 1) == 0) {
                    break;
                }
                const auto cmd_address = reinterpret_cast<const void*>(header);
                rasterizer->ScopeMarker("gfx:{}:DispatchDirect", fmt::make_format_args(cmd_address),
                                        [&] { rasterizer->DispatchDirect(); });
                break;
            }
            case PM4ItOpcode::DispatchIndirect: {
                const auto* dispatch_indirect =
                    reinterpret_cast<const PM4CmdDispatchIndirect*>(header);
                auto& cs_program = GetCsRegs();
                const auto offset = dispatch_indirect->data_offset;
                const auto size = sizeof(PM4CmdDispatchIndirect::GroupDimensions);
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDumpCompute(base_addr, reinterpret_cast<uintptr_t>(header),
                                                   cs_program);
                }
                if (!rasterizer || (cs_program.dispatch_initiator & 1) == 0) {
                    break;
                }
                const auto cmd_address = reinterpret_cast<const void*>(header);
                rasterizer->ScopeMarker(
                    "gfx:{}:DispatchIndirect", fmt::make_format_args(cmd_address),
                    [&] { rasterizer->DispatchIndirect(indirect_args_addr, offset, size); });
                break;
            }
            case PM4ItOpcode::NumInstances: {
                const auto* num_instances = reinterpret_cast<const PM4CmdDrawNumInstances*>(header);
                regs.num_instances.num_instances = num_instances->num_instances;
                break;
            }
            case PM4ItOpcode::IndexBase: {
                const auto* index_base = reinterpret_cast<const PM4CmdDrawIndexBase*>(header);
                regs.index_base_address.base_addr_lo = index_base->addr_lo;
                regs.index_base_address.base_addr_hi = index_base->addr_hi;
                break;
            }
            case PM4ItOpcode::IndexBufferSize: {
                const auto* index_size = reinterpret_cast<const PM4CmdDrawIndexBufferSize*>(header);
                regs.num_indices = index_size->num_indices;
                break;
            }
            case PM4ItOpcode::SetBase: {
                const auto* set_base = reinterpret_cast<const PM4CmdSetBase*>(header);
                ASSERT(set_base->base_index == PM4CmdSetBase::BaseIndex::DrawIndexIndirPatchTable);
                indirect_args_addr = set_base->Address<u64>();
                break;
            }
            case PM4ItOpcode::EventWrite: {
                const auto* event = reinterpret_cast<const PM4CmdEventWrite*>(header);
                LOG_DEBUG(Render, "Encountered EventWrite: event_type = {}, event_index = {}",
                          magic_enum::enum_name(event->event_type.Value()),
                          magic_enum::enum_name(event->event_index.Value()));
                if (TraceSync()) {
                    LogSync("DE", "EVENT_WRITE type={} index={}",
                            magic_enum::enum_name(event->event_type.Value()),
                            magic_enum::enum_name(event->event_index.Value()));
                }
                if (event->event_type.Value() == EventType::SoVgtStreamoutFlush) {
                    // TODO: handle proper synchronization, for now signal that update is done
                    // immediately
                    regs.cp_strmout_cntl.offset_update_done = 1;
                } else if (event->event_index.Value() == EventIndex::ZpassDone) {
                    if (event->event_type.Value() == EventType::PixelPipeStatDump) {
                        static constexpr u64 OcclusionCounterValidMask = 0x8000000000000000ULL;
                        static constexpr u64 OcclusionCounterStep = 0x2FFFFFFULL;
                        u64* results = event->Address<u64*>();
                        for (s32 i = 0; i < num_counter_pairs; ++i, results += 2) {
                            *results = pixel_counter | OcclusionCounterValidMask;
                        }
                        pixel_counter += OcclusionCounterStep;
                    }
                }
                break;
            }
            case PM4ItOpcode::EventWriteEos: {
                const auto* event_eos = reinterpret_cast<const PM4CmdEventWriteEos*>(header);
                if (TraceSync()) {
                    LogSync("DE", "EVENT_WRITE_EOS type={} command={} addr={:#x} data={:#x}",
                            event_eos->event_type.Value(), u32(event_eos->command.Value()),
                            event_eos->Address<VAddr>(), event_eos->data);
                }
                if (rasterizer) {
                    rasterizer->OnFence();
                }
                RecordLabelLag(GfxQueueId);
                event_eos->SignalFence([](void* address, u64 data, u32 num_bytes) {
                    auto* memory = Core::Memory::Instance();
                    Common::GuestWriteJournal::Record(Common::GuestWriteJournal::Source::Label,
                                                      reinterpret_cast<u64>(address), num_bytes,
                                                      &data, 0);
                    ASSERT(memory->TryWriteBacking(address, &data, num_bytes));
                });
                if (event_eos->command == PM4CmdEventWriteEos::Command::GdsStore) {
                    ASSERT(event_eos->size == 1);
                    if (rasterizer) {
                        rasterizer->Finish();
                        const u32 value = rasterizer->ReadDataFromGds(event_eos->gds_index);
                        *event_eos->Address() = value;
                    }
                }
                break;
            }
            case PM4ItOpcode::EventWriteEop: {
                const auto* event_eop = reinterpret_cast<const PM4CmdEventWriteEop*>(header);
                if (TraceSync()) {
                    LogSync("DE", "EVENT_WRITE_EOP type={} data_sel={} int_sel={} addr={} data={:#x}",
                            event_eop->event_type.Value(), u32(event_eop->data_sel.Value()),
                            u32(event_eop->int_sel.Value()), fmt::ptr(event_eop->Address<u32>()),
                            event_eop->data_lo);
                }
                if (rasterizer) {
                    rasterizer->OnFence();
                }
                RecordLabelLag(GfxQueueId);
                const auto write_label = [](void* address, u64 data, u32 num_bytes) {
                    auto* memory = Core::Memory::Instance();
                    Common::GuestWriteJournal::Record(Common::GuestWriteJournal::Source::Label,
                                                      reinterpret_cast<u64>(address), num_bytes,
                                                      &data, 1);
                    ASSERT(memory->TryWriteBacking(address, &data, num_bytes));
                };
                const auto signal_irq = [] {
                    Platform::IrqC::Instance()->Signal(Platform::InterruptId::GfxEop);
                };
                // With eop_label_delay_us, write this gfx label and raise its interrupt after the
                // fixed timer delay
                // GPU timestamps and video output labels still use their normal immediate write
                // path
                const auto data_sel = event_eop->data_sel.Value();
                if (const u32 delay_us = DelayedLabels::DelayUs();
                    delay_us != 0 && data_sel != DataSelect::GpuClock64 &&
                    data_sel != DataSelect::PerfCounter &&
                    !vo_port->IsVoLabel(event_eop->Address<u64>())) {
                    const PM4CmdEventWriteEop packet = *event_eop;
                    const auto label_addr = reinterpret_cast<VAddr>(event_eop->Address<u8>());
                    const u64 seq =
                        data_sel == DataSelect::Data32Low
                            ? DelayedLabels::AddPending(label_addr, packet.DataDWord(), sizeof(u32))
                        : data_sel == DataSelect::Data64
                            ? DelayedLabels::AddPending(label_addr, packet.DataQWord(), sizeof(u64))
                            : 0;
                    // The guest may have unmapped this label while the timer was waiting, so drop
                    // the delayed write instead of aborting on an invalid address
                    const auto write_delayed = [](void* address, u64 data, u32 num_bytes) {
                        Common::GuestWriteJournal::Record(Common::GuestWriteJournal::Source::Label,
                                                          reinterpret_cast<u64>(address), num_bytes,
                                                          &data, 1);
                        if (!Core::Memory::Instance()->TryWriteBacking(address, &data, num_bytes)) {
                            LOG_WARNING(Render, "eop_label_delay_us: label {} unmapped before its "
                                                "delayed write, dropped",
                                        fmt::ptr(address));
                        }
                    };
                    DelayedLabels::Push(u64{delay_us} * 1000,
                                        [packet, write_delayed, signal_irq, label_addr, seq] {
                                            packet.SignalFence(write_delayed, signal_irq);
                                            if (seq != 0) {
                                                DelayedLabels::RemovePending(label_addr, seq);
                                            }
                                        });
                    break;
                }
                event_eop->SignalFence(write_label, signal_irq);
                break;
            }
            case PM4ItOpcode::DmaData: {
                const auto* dma_data = reinterpret_cast<const PM4DmaData*>(header);
                if (dma_data->dst_addr_lo == 0x3022C || !rasterizer) {
                    break;
                }
                ASSERT(dma_data->command.das == 0);
                if (Vulkan::DrawTrace::Instance().Active()) {
                    LOG_WARNING(Render, "DrawTrace f={} DMA_DATA src_sel={} dst_sel={} src={:#x} "
                                        "dst={:#x} bytes={:#x}",
                                Vulkan::DrawTrace::Instance().Frame(), u32(dma_data->src_sel),
                                u32(dma_data->dst_sel), dma_data->SrcAddress<VAddr>(),
                                dma_data->DstAddress<VAddr>(), dma_data->NumBytes());
                }
                if (dma_data->src_sel == DmaDataSrc::Data && dma_data->dst_sel == DmaDataDst::Gds) {
                    rasterizer->FillBuffer(dma_data->dst_addr_lo, dma_data->NumBytes(),
                                           dma_data->data, true);
                } else if ((dma_data->src_sel == DmaDataSrc::Memory ||
                            dma_data->src_sel == DmaDataSrc::MemoryUsingL2) &&
                           dma_data->dst_sel == DmaDataDst::Gds) {
                    rasterizer->CopyBuffer(dma_data->dst_addr_lo, dma_data->SrcAddress<VAddr>(),
                                           dma_data->NumBytes(), true, false);
                } else if (dma_data->src_sel == DmaDataSrc::Data &&
                           (dma_data->dst_sel == DmaDataDst::Memory ||
                            dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                    rasterizer->FillBuffer(dma_data->DstAddress<VAddr>(), dma_data->NumBytes(),
                                           dma_data->data, false);
                } else if (dma_data->src_sel == DmaDataSrc::Gds &&
                           (dma_data->dst_sel == DmaDataDst::Memory ||
                            dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                    rasterizer->CopyBuffer(dma_data->DstAddress<VAddr>(), dma_data->src_addr_lo,
                                           dma_data->NumBytes(), false, true);
                } else if ((dma_data->src_sel == DmaDataSrc::Memory ||
                            dma_data->src_sel == DmaDataSrc::MemoryUsingL2) &&
                           (dma_data->dst_sel == DmaDataDst::Memory ||
                            dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                    rasterizer->CopyBuffer(dma_data->DstAddress<VAddr>(),
                                           dma_data->SrcAddress<VAddr>(), dma_data->NumBytes(),
                                           false, false);
                } else {
                    UNREACHABLE_MSG("WriteData src_sel = {}, dst_sel = {}", u32(dma_data->src_sel),
                                    u32(dma_data->dst_sel));
                }
                break;
            }
            case PM4ItOpcode::WriteData: {
                const auto* write_data = reinterpret_cast<const PM4CmdWriteData*>(header);
                ASSERT(write_data->dst_sel.Value() == 2 || write_data->dst_sel.Value() == 5);
                const u32 data_size = (header->type3.count.Value() - 2) * 4;
                u64* address = write_data->Address<u64*>();
                if (Vulkan::DrawTrace::Instance().Active()) {
                    LOG_WARNING(Render, "DrawTrace f={} WRITE_DATA dst={} bytes={:#x}",
                                Vulkan::DrawTrace::Instance().Frame(), fmt::ptr(address),
                                data_size);
                }
                if (!write_data->wr_one_addr.Value()) {
                    if (rasterizer) {
                        rasterizer->OnFence();
                    }
                    Common::GuestWriteJournal::Record(Common::GuestWriteJournal::Source::WriteData,
                                                      reinterpret_cast<u64>(address), data_size,
                                                      write_data->data, 0);
                    std::memcpy(address, write_data->data, data_size);
                } else {
                    UNREACHABLE();
                }
                break;
            }
            case PM4ItOpcode::CopyData: {
                const auto* copy_data = reinterpret_cast<const PM4CmdCopyData*>(header);
                LOG_RENDER_PROBLEM(Render, Warning,
                                   "unhandled IT_COPY_DATA src_sel = {}, dst_sel = {}, "
                                   "count_sel = {}, wr_confirm = {}, engine_sel = {}",
                                   u32(copy_data->src_sel.Value()), u32(copy_data->dst_sel.Value()),
                                   copy_data->count_sel.Value(), copy_data->wr_confirm.Value(),
                                   u32(copy_data->engine_sel.Value()));
                break;
            }
            case PM4ItOpcode::MemSemaphore: {
                const auto* mem_semaphore = reinterpret_cast<const PM4CmdMemSemaphore*>(header);
                if (TraceSync()) {
                    LogSync("DE", "MEM_SEMAPHORE {} addr={:#x}",
                            mem_semaphore->IsSignaling() ? "signal" : "wait",
                            mem_semaphore->Address<VAddr>());
                }
                if (mem_semaphore->IsSignaling()) {
                    mem_semaphore->Signal();
                } else {
                    while (!mem_semaphore->Signaled()) {
                        YIELD_GFX();
                    }
                    mem_semaphore->Decrement();
                }
                break;
            }
            case PM4ItOpcode::AcquireMem: {
                const auto* acquire_mem = reinterpret_cast<const PM4CmdAcquireMem*>(header);
                if (TraceSync()) {
                    LogSync("DE", "ACQUIRE_MEM cntl={:#x} base={:#x}", acquire_mem->cp_coher_cntl,
                            acquire_mem->cp_coher_base_lo);
                }
                break;
            }
            case PM4ItOpcode::Rewind: {
                if (!rasterizer) {
                    break;
                }
                const PM4CmdRewind* rewind = reinterpret_cast<const PM4CmdRewind*>(header);
                while (!rewind->Valid()) {
                    YIELD_GFX();
                }
                break;
            }
            case PM4ItOpcode::WaitRegMem: {
                const auto* wait_reg_mem = reinterpret_cast<const PM4CmdWaitRegMem*>(header);
                // ASSERT(wait_reg_mem->engine.Value() == PM4CmdWaitRegMem::Engine::Me);
                // Optimization: VO label waits are special because the emulator
                // will write to the label when presentation is finished. So if
                // there are no other submits to yield to we can sleep the thread
                // instead and allow other tasks to run.
                const u64* wait_addr = wait_reg_mem->Address<u64*>();
                if (TraceSync()) {
                    LogSync("DE", "WAIT_REG_MEM {} {:#x} func={} ref={:#x} mask={:#x} passes={}",
                            wait_reg_mem->mem_space.Value() == PM4CmdWaitRegMem::MemSpace::Memory
                                ? "mem"
                                : "reg",
                            wait_reg_mem->mem_space.Value() == PM4CmdWaitRegMem::MemSpace::Memory
                                ? reinterpret_cast<VAddr>(wait_addr)
                                : VAddr{wait_reg_mem->Reg()},
                            u32(wait_reg_mem->function.Value()), wait_reg_mem->ref,
                            wait_reg_mem->mask, wait_reg_mem->Test(regs.reg_array));
                }
                if (vo_port->IsVoLabel(wait_addr) &&
                    num_submits == mapped_queues[GfxQueueId].submits.size()) {
                    const Common::PerfStats::ScopedNs perf_timer{Common::PerfStats::Id::CpVoWaitNs};
                    vo_port->WaitVoLabel([&] { return wait_reg_mem->Test(regs.reg_array); });
                    break;
                }
                while (!CpWaitPasses(wait_reg_mem, regs.reg_array)) {
                    Common::PerfStats::Add(Common::PerfStats::Id::CpWaitYields);
                    YIELD_GFX();
                }
                break;
            }
            case PM4ItOpcode::IndirectBuffer: {
                const auto* indirect_buffer = reinterpret_cast<const PM4CmdIndirectBuffer*>(header);
                auto task = ProcessGraphics(
                    {indirect_buffer->Address<const u32>(), indirect_buffer->ib_size}, {});
                RESUME_GFX(task);

                while (!task.handle.done()) {
                    YIELD_GFX();
                    RESUME_GFX(task);
                }
                break;
            }
            case PM4ItOpcode::IncrementDeCounter: {
                ++cblock.de_count;
                break;
            }
            case PM4ItOpcode::WaitOnCeCounter: {
                while (cblock.ce_count <= cblock.de_count && !ce_task.handle.done()) {
                    RESUME_GFX(ce_task);
                }
                break;
            }
            case PM4ItOpcode::PfpSyncMe: {
                break;
            }
            case PM4ItOpcode::StrmoutBufferUpdate: {
                const auto* strmout = reinterpret_cast<const PM4CmdStrmoutBufferUpdate*>(header);
                LOG_RENDER_PROBLEM(Render_Vulkan, Warning,
                                   "Unimplemented IT_STRMOUT_BUFFER_UPDATE, update_memory = {}, "
                                   "source_select = {}, buffer_select = {}",
                                   strmout->update_memory.Value(),
                                   magic_enum::enum_name(strmout->source_select.Value()),
                                   strmout->buffer_select.Value());
                break;
            }
            case PM4ItOpcode::GetLodStats: {
                const auto* lod_stats = reinterpret_cast<const PM4CmdGetLodStats*>(header);
                if (rasterizer &&
                    rasterizer->WriteLodStats(lod_stats->Address(), lod_stats->NumBytes())) {
                    break;
                }
                // Safely ignored unless lod_stats_from_bindings is on. Kept at DEBUG: some titles emit it
                // every frame
                LOG_DEBUG(Render_Vulkan, "IT_GET_LOD_STATS ignored");
                break;
            }
            case PM4ItOpcode::PredExec: {
                // Predicated execution of the next packets on a GPU-evaluated condition; the packets are
                // executed unconditionally, like a predicate that always passes
                LOG_DEBUG(Render, "IT_PRED_EXEC ignored");
                break;
            }
            case PM4ItOpcode::CondExec: {
                const auto* cond_exec = reinterpret_cast<const PM4CmdCondExec*>(header);
                if (CondExecSkips(cond_exec, "DE")) {
                    dcb = NextPacket(dcb,
                                     header->type3.NumWords() + 1 + cond_exec->exec_count.Value());
                    continue;
                }
                break;
            }
            default:
                UNREACHABLE_MSG("Unknown PM4 type 3 opcode {:#x} with count {}",
                                static_cast<u32>(opcode), count);
            }
            dcb = NextPacket(dcb, header->type3.NumWords() + 1);
            break;
        }
    }

    if (ce_task.handle) {
        while (!ce_task.handle.done()) {
            RESUME_GFX(ce_task);
        }
        ce_task.handle.destroy();
    }

    FIBER_EXIT;
}

template <bool is_indirect>
Liverpool::Task Liverpool::ProcessCompute(std::span<const u32> acb, u32 vqid) {
    FIBER_ENTER(acb_task_name[vqid]);
    auto& queue = asc_queues[{vqid}];

    struct IndirectPatch {
        const PM4Header* header;
        VAddr indirect_addr;
    };
    boost::container::small_vector<IndirectPatch, 4> indirect_patches;

    auto base_addr = reinterpret_cast<VAddr>(acb.data());
    size_t acb_size = acb.size_bytes();
    while (!acb.empty()) {
        ProcessCommands();

        auto* header = reinterpret_cast<const PM4Header*>(acb.data());
        u32 next_dw_off = header->type3.NumWords() + 1;

        // If we have a buffered packet, use it.
        if (queue.tmp_dwords > 0) [[unlikely]] {
            header = reinterpret_cast<const PM4Header*>(queue.tmp_packet.data());
            next_dw_off = header->type3.NumWords() + 1 - queue.tmp_dwords;
            std::memcpy(queue.tmp_packet.data() + queue.tmp_dwords, acb.data(),
                        next_dw_off * sizeof(u32));
            queue.tmp_dwords = 0;
        }

        // If the packet is split across ring boundary, buffer until next submission
        if (next_dw_off > acb.size()) [[unlikely]] {
            std::memcpy(queue.tmp_packet.data(), acb.data(), acb.size_bytes());
            queue.tmp_dwords = acb.size();
            if constexpr (!is_indirect) {
                *queue.read_addr += acb.size();
                *queue.read_addr %= queue.ring_size_dw;
            }
            break;
        }

        if (header->type == 2) {
            // Type-2 packet are used for padding purposes
            next_dw_off = 1;
            acb = NextPacket(acb, next_dw_off);
            if constexpr (!is_indirect) {
                *queue.read_addr += next_dw_off;
                *queue.read_addr %= queue.ring_size_dw;
            }
            continue;
        }

        if (header->type != 3) {
            // No other types of packets were spotted so far
            UNREACHABLE_MSG("Invalid PM4 type {}", header->type.Value());
        }

        const PM4ItOpcode opcode = header->type3.opcode;
        if (CpHistory::Enabled()) {
            CpHistory::RecordPacket(static_cast<s16>(vqid), acb);
        }

        const auto* it_body = reinterpret_cast<const u32*>(header) + 1;
        switch (opcode) {
        case PM4ItOpcode::Nop: {
            const auto* nop = reinterpret_cast<const PM4CmdNop*>(header);
            break;
        }
        case PM4ItOpcode::IndirectBuffer: {
            const auto* indirect_buffer = reinterpret_cast<const PM4CmdIndirectBuffer*>(header);
            auto task = ProcessCompute<true>(
                {indirect_buffer->Address<const u32>(), indirect_buffer->ib_size}, vqid);
            RESUME_ASC(task, vqid);

            while (!task.handle.done()) {
                YIELD_ASC(vqid);
                RESUME_ASC(task, vqid);
            }
            break;
        }
        case PM4ItOpcode::DmaData: {
            const auto* dma_data = reinterpret_cast<const PM4DmaData*>(header);
            if (dma_data->dst_addr_lo == 0x3022C || !rasterizer) {
                break;
            }
            ASSERT(dma_data->command.das == 0);
            if (dma_data->src_sel == DmaDataSrc::Data && dma_data->dst_sel == DmaDataDst::Gds) {
                rasterizer->FillBuffer(dma_data->dst_addr_lo, dma_data->NumBytes(), dma_data->data,
                                       true);
            } else if ((dma_data->src_sel == DmaDataSrc::Memory ||
                        dma_data->src_sel == DmaDataSrc::MemoryUsingL2) &&
                       dma_data->dst_sel == DmaDataDst::Gds) {
                rasterizer->CopyBuffer(dma_data->dst_addr_lo, dma_data->SrcAddress<VAddr>(),
                                       dma_data->NumBytes(), true, false);
            } else if (dma_data->src_sel == DmaDataSrc::Data &&
                       (dma_data->dst_sel == DmaDataDst::Memory ||
                        dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                rasterizer->FillBuffer(dma_data->DstAddress<VAddr>(), dma_data->NumBytes(),
                                       dma_data->data, false);
            } else if (dma_data->src_sel == DmaDataSrc::Gds &&
                       (dma_data->dst_sel == DmaDataDst::Memory ||
                        dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                rasterizer->CopyBuffer(dma_data->DstAddress<VAddr>(), dma_data->src_addr_lo,
                                       dma_data->NumBytes(), false, true);
            } else if ((dma_data->src_sel == DmaDataSrc::Memory ||
                        dma_data->src_sel == DmaDataSrc::MemoryUsingL2) &&
                       (dma_data->dst_sel == DmaDataDst::Memory ||
                        dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                const u32 num_bytes = dma_data->NumBytes();
                const VAddr src_addr = dma_data->SrcAddress<VAddr>();
                const VAddr dst_addr = dma_data->DstAddress<VAddr>();
                const PM4Header* header =
                    reinterpret_cast<const PM4Header*>(dst_addr - sizeof(PM4Header));
                if (dst_addr >= base_addr && dst_addr < base_addr + acb_size &&
                    num_bytes == sizeof(PM4CmdDispatchIndirect::GroupDimensions) &&
                    header->type == 3 && header->type3.opcode == PM4ItOpcode::DispatchDirect) {
                    indirect_patches.emplace_back(header, src_addr);
                } else {
                    rasterizer->CopyBuffer(dst_addr, src_addr, num_bytes, false, false);
                }
            } else {
                UNREACHABLE_MSG("WriteData src_sel = {}, dst_sel = {}", u32(dma_data->src_sel),
                                u32(dma_data->dst_sel));
            }
            break;
        }
        case PM4ItOpcode::AcquireMem: {
            if (TraceSync()) {
                const auto* acquire_mem = reinterpret_cast<const PM4CmdAcquireMem*>(header);
                LogSync(fmt::format("ASC{}", vqid), "ACQUIRE_MEM cntl={:#x} base={:#x}",
                        acquire_mem->cp_coher_cntl, acquire_mem->cp_coher_base_lo);
            }
            break;
        }
        case PM4ItOpcode::Rewind: {
            if (!rasterizer) {
                break;
            }
            const PM4CmdRewind* rewind = reinterpret_cast<const PM4CmdRewind*>(header);
            while (!rewind->Valid()) {
                YIELD_ASC(vqid);
            }
            break;
        }
        case PM4ItOpcode::SetShReg: {
            const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
            const auto set_size = (header->type3.NumWords() - 1) * sizeof(u32);

            if (set_data->reg_offset >= 0x200 &&
                set_data->reg_offset <= (0x200 + sizeof(ComputeProgram) / 4)) {
                ASSERT(set_size <= sizeof(ComputeProgram));
                auto* addr = reinterpret_cast<u32*>(&mapped_queues[vqid + 1].cs_state) +
                             (set_data->reg_offset - 0x200);
                std::memcpy(addr, header + 2, set_size);
            } else {
                std::memcpy(&regs.reg_array[Regs::ShRegWordOffset + set_data->reg_offset],
                            header + 2, set_size);
            }
            break;
        }
        case PM4ItOpcode::SetQueueReg: {
            const auto* set_data = reinterpret_cast<const PM4CmdSetQueueReg*>(header);
            LOG_WARNING(Render, "Encountered compute SetQueueReg: vqid = {}, reg_offset = {:#x}",
                        set_data->vqid.Value(), set_data->reg_offset.Value());
            break;
        }
        case PM4ItOpcode::DispatchDirect: {
            const auto* dispatch_direct = reinterpret_cast<const PM4CmdDispatchDirect*>(header);
            if (TraceSync()) {
                LogSync(fmt::format("ASC{}", vqid), "DISPATCH_DIRECT {}x{}x{}",
                        dispatch_direct->dim_x, dispatch_direct->dim_y, dispatch_direct->dim_z);
            }
            if (auto it = std::ranges::find(indirect_patches, header, &IndirectPatch::header);
                it != indirect_patches.end()) {
                const auto size = sizeof(PM4CmdDispatchIndirect::GroupDimensions);
                rasterizer->DispatchIndirect(it->indirect_addr, 0, size);
                break;
            }
            auto& cs_program = GetCsRegs();
            cs_program.dim_x = dispatch_direct->dim_x;
            cs_program.dim_y = dispatch_direct->dim_y;
            cs_program.dim_z = dispatch_direct->dim_z;
            cs_program.dispatch_initiator = dispatch_direct->dispatch_initiator;
            if (DebugState.DumpingCurrentReg()) {
                DebugState.PushRegsDumpCompute(base_addr, reinterpret_cast<uintptr_t>(header),
                                               cs_program);
            }
            if (!rasterizer || (cs_program.dispatch_initiator & 1) == 0) {
                break;
            }
            const auto cmd_address = reinterpret_cast<const void*>(header);
            rasterizer->ScopeMarker("asc[{}]:{}:DispatchDirect",
                                    fmt::make_format_args(vqid, cmd_address),
                                    [&] { rasterizer->DispatchDirect(); });
            break;
        }
        case PM4ItOpcode::DispatchIndirect: {
            const auto* dispatch_indirect =
                reinterpret_cast<const PM4CmdDispatchIndirectMec*>(header);
            auto& cs_program = GetCsRegs();
            const auto ib_address = dispatch_indirect->Address<VAddr>();
            if (TraceSync()) {
                LogSync(fmt::format("ASC{}", vqid), "DISPATCH_INDIRECT args={:#x}", ib_address);
            }
            const auto size = sizeof(PM4CmdDispatchIndirect::GroupDimensions);
            if (DebugState.DumpingCurrentReg()) {
                DebugState.PushRegsDumpCompute(base_addr, reinterpret_cast<uintptr_t>(header),
                                               cs_program);
            }
            if (!rasterizer || (cs_program.dispatch_initiator & 1) == 0) {
                break;
            }
            const auto cmd_address = reinterpret_cast<const void*>(header);
            rasterizer->ScopeMarker("asc[{}]:{}:DispatchIndirect",
                                    fmt::make_format_args(vqid, cmd_address),
                                    [&] { rasterizer->DispatchIndirect(ib_address, 0, size); });
            break;
        }
        case PM4ItOpcode::WriteData: {
            const auto* write_data = reinterpret_cast<const PM4CmdWriteData*>(header);
            ASSERT(write_data->dst_sel.Value() == 2 || write_data->dst_sel.Value() == 5);
            const u32 data_size = (header->type3.count.Value() - 2) * 4;
            if (TraceSync()) {
                LogSync(fmt::format("ASC{}", vqid), "WRITE_DATA dst={:#x} bytes={:#x}",
                        write_data->Address<VAddr>(), data_size);
            }
            if (!write_data->wr_one_addr.Value()) {
                if (rasterizer) {
                    rasterizer->OnFence();
                }
                Common::GuestWriteJournal::Record(Common::GuestWriteJournal::Source::WriteData,
                                                  write_data->Address<VAddr>(), data_size,
                                                  write_data->data, 1);
                std::memcpy(write_data->Address<void*>(), write_data->data, data_size);
            } else {
                UNREACHABLE();
            }
            break;
        }
        case PM4ItOpcode::MemSemaphore: {
            const auto* mem_semaphore = reinterpret_cast<const PM4CmdMemSemaphore*>(header);
            if (TraceSync()) {
                LogSync(fmt::format("ASC{}", vqid), "MEM_SEMAPHORE {} addr={:#x}",
                        mem_semaphore->IsSignaling() ? "signal" : "wait",
                        mem_semaphore->Address<VAddr>());
            }
            if (mem_semaphore->IsSignaling()) {
                mem_semaphore->Signal();
            } else {
                while (!mem_semaphore->Signaled()) {
                    YIELD_ASC(vqid);
                }
                mem_semaphore->Decrement();
            }
            break;
        }
        case PM4ItOpcode::WaitRegMem: {
            const auto* wait_reg_mem = reinterpret_cast<const PM4CmdWaitRegMem*>(header);
            ASSERT(wait_reg_mem->engine.Value() == PM4CmdWaitRegMem::Engine::Me);
            if (TraceSync()) {
                const bool is_mem =
                    wait_reg_mem->mem_space.Value() == PM4CmdWaitRegMem::MemSpace::Memory;
                LogSync(fmt::format("ASC{}", vqid),
                        "WAIT_REG_MEM {} {:#x} func={} ref={:#x} mask={:#x} passes={}",
                        is_mem ? "mem" : "reg",
                        is_mem ? wait_reg_mem->Address<VAddr>() : VAddr{wait_reg_mem->Reg()},
                        u32(wait_reg_mem->function.Value()), wait_reg_mem->ref,
                        wait_reg_mem->mask, wait_reg_mem->Test(regs.reg_array));
            }
            while (!CpWaitPasses(wait_reg_mem, regs.reg_array)) {
                Common::PerfStats::Add(Common::PerfStats::Id::CpWaitYields);
                YIELD_ASC(vqid);
            }
            break;
        }
        case PM4ItOpcode::ReleaseMem: {
            const auto* release_mem = reinterpret_cast<const PM4CmdReleaseMem*>(header);
            if (TraceSync()) {
                LogSync(fmt::format("ASC{}", vqid),
                        "RELEASE_MEM type={} data_sel={} int_sel={} addr={:#x} data={:#x}",
                        release_mem->event_type.Value(), u32(release_mem->data_sel.Value()),
                        u32(release_mem->int_sel.Value()),
                        u64(release_mem->address_hi) << 32 | release_mem->address_lo,
                        release_mem->data_lo);
            }
            if (rasterizer) {
                rasterizer->OnFence();
            }
            RecordLabelLag(vqid + 1);
            release_mem->SignalFence(
                [pipe_id = queue.pipe_id] {
                    Platform::IrqC::Instance()->Signal(static_cast<Platform::InterruptId>(pipe_id));
                },
                [this](VAddr dst, u16 gds_index, u16 num_dwords) {
                    rasterizer->CopyBuffer(dst, gds_index, num_dwords * sizeof(u32), false, true);
                });
            break;
        }
        case PM4ItOpcode::EventWrite: {
            // const auto* event = reinterpret_cast<const PM4CmdEventWrite*>(header);
            break;
        }
        case PM4ItOpcode::GetLodStats: {
            const auto* lod_stats = reinterpret_cast<const PM4CmdGetLodStats*>(header);
            if (!rasterizer ||
                !rasterizer->WriteLodStats(lod_stats->Address(), lod_stats->NumBytes())) {
                LOG_DEBUG(Render_Vulkan, "IT_GET_LOD_STATS ignored (ASC)");
            }
            break;
        }
        case PM4ItOpcode::PredExec: {
            // Executed unconditionally, like a predicate that always passes (as on the DE)
            LOG_DEBUG(Render, "IT_PRED_EXEC ignored (ASC)");
            break;
        }
        case PM4ItOpcode::CondExec: {
            const auto* cond_exec = reinterpret_cast<const PM4CmdCondExec*>(header);
            if (CondExecSkips(cond_exec, "ASC")) {
                // The skipped packets must be in this submission: past its end they wrap around the ring and
                // would need the next one. Those are executed instead
                const u32 exec_count = cond_exec->exec_count.Value();
                if (next_dw_off + exec_count <= acb.size()) {
                    next_dw_off += exec_count;
                } else {
                    LOG_WARNING(Render, "IT_COND_EXEC skip of {} dwords crosses the end of the "
                                        "ASC submission, executing",
                                exec_count);
                }
            }
            break;
        }
        default:
            UNREACHABLE_MSG("Unknown PM4 type 3 opcode {:#x} with count {}",
                            static_cast<u32>(opcode), header->type3.NumWords());
        }

        acb = NextPacket(acb, next_dw_off);

        if constexpr (!is_indirect) {
            *queue.read_addr += next_dw_off;
            *queue.read_addr %= queue.ring_size_dw;
        }
    }

    FIBER_EXIT;
}

Liverpool::CmdBuffer Liverpool::CopyCmdBuffers(std::span<const u32> dcb, std::span<const u32> ccb) {
    auto& queue = mapped_queues[GfxQueueId];
    ASSERT_MSG(queue.dcb_buffer.capacity() >= queue.dcb_buffer_offset + dcb.size(),
               "dcb copy buffer out of reserved space");
    ASSERT_MSG(queue.ccb_buffer.capacity() >= queue.ccb_buffer_offset + ccb.size(),
               "ccb copy buffer out of reserved space");

    queue.dcb_buffer.resize(
        std::max(queue.dcb_buffer.size(), queue.dcb_buffer_offset + dcb.size()));
    queue.ccb_buffer.resize(
        std::max(queue.ccb_buffer.size(), queue.ccb_buffer_offset + ccb.size()));

    const u32 prev_dcb_buffer_offset = queue.dcb_buffer_offset;
    const u32 prev_ccb_buffer_offset = queue.ccb_buffer_offset;
    if (!dcb.empty()) {
        std::memcpy(queue.dcb_buffer.data() + queue.dcb_buffer_offset, dcb.data(),
                    dcb.size_bytes());
        queue.dcb_buffer_offset += dcb.size();
        dcb = std::span<const u32>{queue.dcb_buffer.begin() + prev_dcb_buffer_offset,
                                   queue.dcb_buffer.begin() + queue.dcb_buffer_offset};
    }

    if (!ccb.empty()) {
        std::memcpy(queue.ccb_buffer.data() + queue.ccb_buffer_offset, ccb.data(),
                    ccb.size_bytes());
        queue.ccb_buffer_offset += ccb.size();
        ccb = std::span<const u32>{queue.ccb_buffer.begin() + prev_ccb_buffer_offset,
                                   queue.ccb_buffer.begin() + queue.ccb_buffer_offset};
    }

    return std::make_pair(dcb, ccb);
}

void Liverpool::RecordLabelLag(u32 qid) {
    if (!Common::PerfStats::Enabled()) {
        return;
    }
    auto& queue = mapped_queues[qid];
    std::chrono::steady_clock::time_point submitted;
    {
        std::scoped_lock lock{queue.m_access};
        if (queue.submit_times.empty()) {
            return;
        }
        submitted = queue.submit_times.front();
    }
    const u64 lag_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                           std::chrono::steady_clock::now() - submitted)
                           .count();
    Common::PerfStats::Add(Common::PerfStats::Id::LabelWrites);
    Common::PerfStats::Add(Common::PerfStats::Id::LabelLagNs, lag_ns);
    Common::PerfStats::Max(Common::PerfStats::Id::LabelLagMaxNs, lag_ns);
}

void Liverpool::SubmitGfx(std::span<const u32> dcb, std::span<const u32> ccb) {
    auto& queue = mapped_queues[GfxQueueId];

    if (EmulatorSettings.IsCopyGpuBuffers()) {
        std::tie(dcb, ccb) = CopyCmdBuffers(dcb, ccb);
    }
    if (CpHistory::Enabled()) {
        CpHistory::RecordSubmit(CpHistory::Queue::De, dcb);
    }

    auto task = ProcessGraphics(dcb, ccb);
    {
        std::scoped_lock lock{queue.m_access};
        queue.submits.emplace(task.handle);
        if (Common::PerfStats::Enabled()) {
            queue.submit_times.emplace(std::chrono::steady_clock::now());
            Common::PerfStats::Max(Common::PerfStats::Id::GfxQueueMax, queue.submits.size());
        }
    }

    std::scoped_lock lk{submit_mutex};
    ++num_submits;
    submit_cv.notify_one();
}

void Liverpool::SubmitAsc(u32 gnm_vqid, std::span<const u32> acb) {
    ASSERT_MSG(gnm_vqid > 0 && gnm_vqid < NumTotalQueues, "Invalid virtual ASC queue index");
    auto& queue = mapped_queues[gnm_vqid];

    const auto vqid = gnm_vqid - 1;
    if (CpHistory::Enabled()) {
        CpHistory::RecordSubmit(static_cast<s16>(vqid), acb);
    }
    const auto& task = ProcessCompute(acb, vqid);
    {
        std::scoped_lock lock{queue.m_access};
        queue.submits.emplace(task.handle);
        if (Common::PerfStats::Enabled()) {
            queue.submit_times.emplace(std::chrono::steady_clock::now());
        }
    }

    std::scoped_lock lk{submit_mutex};
    num_mapped_queues = std::max(num_mapped_queues, gnm_vqid + 1);
    ++num_submits;
    submit_cv.notify_one();
}

} // namespace AmdGpu
