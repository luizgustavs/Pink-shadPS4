// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <thread>
#include <tuple>
#include <fmt/ranges.h>

#include "common/debug.h"
#include "common/guest_write_journal.h"
#include "common/logging/events.h"
#include "common/perf_stats.h"
#include "core/debug_state.h"
#include "core/emulator_settings.h"
#include "core/libraries/kernel/process.h"
#include "core/memory.h"
#include "shader_recompiler/runtime_info.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/amdgpu/tiling.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/vk_draw_trace.h"
#include "video_core/renderer_vulkan/vk_gpu_checkpoints.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_hle.h"
#include "video_core/texture_cache/image_view.h"
#include "video_core/texture_cache/texture_cache.h"
#include "video_core/texture_cache/tile.h"

namespace Vulkan {

namespace {
// Frametime audit probe (perf stats only): how many pipeline binds repeat the last bind of the same command
// buffer and bind point
void CountPipelineBind(vk::CommandBuffer cmdbuf, vk::PipelineBindPoint point, vk::Pipeline pipeline) {
    if (!Common::PerfStats::Enabled()) {
        return;
    }
    thread_local std::array<std::pair<VkCommandBuffer, VkPipeline>, 2> last{};
    auto& slot = last[point == vk::PipelineBindPoint::eCompute ? 1 : 0];
    Common::PerfStats::Add(Common::PerfStats::Id::PipelineBinds);
    if (slot.first == static_cast<VkCommandBuffer>(cmdbuf) &&
        slot.second == static_cast<VkPipeline>(pipeline)) {
        Common::PerfStats::Add(Common::PerfStats::Id::PipelineBindsSame);
    }
    slot = {static_cast<VkCommandBuffer>(cmdbuf), static_cast<VkPipeline>(pipeline)};
}
} // Anonymous namespace

namespace {

/// Diagnostic: SHADPS4_WRITE_WATCH=<addr> (hex) logs the first 64 distinct stage/range pairs that mark it
/// GPU-modified, i.e. the writers of a corrupted address
VAddr WriteWatchAddress() {
    static const VAddr watch = [] {
        const char* env = std::getenv("SHADPS4_WRITE_WATCH");
        return env ? static_cast<VAddr>(std::strtoull(env, nullptr, 16)) : VAddr{0};
    }();
    return watch;
}

void WatchGpuWrite(std::string_view source, u64 pgm_hash, VAddr addr, u64 size) {
    const VAddr watch = WriteWatchAddress();
    if (watch == 0 || watch < addr || watch >= addr + size) {
        return;
    }
    static std::mutex watch_mutex;
    static std::set<std::tuple<u64, VAddr, u64>> seen;
    std::scoped_lock lk{watch_mutex};
    if (seen.size() < 64 && seen.emplace(pgm_hash, addr, size).second) {
        LOG_WARNING(Render, "Write watch {:#x}: {} (stage {:#x}) marks {:#x}+{:#x} GPU-modified",
                    watch, source, pgm_hash, addr, size);
    }
}

// SHADPS4_DRAW_TRACE: one line per draw or dispatch in the selected frames, then its targets, buffers and
// textures. Null T#s also dump the raw fetch and the SRT pointer chain

void TraceMemory(s32 frame, u32 index, std::string_view label, VAddr addr,
                 Core::MemoryManager* memory, VideoCore::BufferCache& buffer_cache,
                 const VideoCore::PageManager& page_manager) {
    constexpr u32 NumDwords = 16;
    if (!memory->IsMappedLocked(addr, NumDwords * sizeof(u32))) {
        LOG_WARNING(Render_Vulkan, "DrawTrace f={} #{}     {} {:#x} unmapped", frame, index, label,
                    addr);
        return;
    }
    const bool gpu_modified = buffer_cache.IsRegionGpuModified(addr, NumDwords * sizeof(u32));
    const bool cpu_modified = buffer_cache.IsRegionCpuModified(addr, NumDwords * sizeof(u32));
    u32 protection = 0;
#ifdef _WIN32
    MEMORY_BASIC_INFORMATION mbi{};
    if (VirtualQuery(reinterpret_cast<const void*>(addr), &mbi, sizeof(mbi))) {
        protection = mbi.Protect;
    }
#endif
    const auto [read_watchers, write_watchers] = page_manager.GetWatchers(addr);
    // Read through the backing: a protected page would fault (and read back) otherwise
    std::array<u32, NumDwords> words{};
    const bool read = memory->TryReadBacking(addr, words.data(), sizeof(words));
    std::string text;
    for (const u32 word : words) {
        text += fmt::format(" {:08x}", word);
    }
    LOG_WARNING(Render_Vulkan,
                "DrawTrace f={} #{}     {} {:#x} cpu_modified={} gpu_modified={} protect={:#x} "
                "watchers r={} w={}{}",
                frame, index, label, addr, cpu_modified, gpu_modified, protection, read_watchers,
                write_watchers, read ? text : std::string{" (no backing)"});
}

/// Dumps the descriptor table behind user_data[0:1] and every pointer it holds, so a null T# can be traced
/// back to the memory the SRT walker read. SHADPS4_DRAW_TRACE_CHAIN=<hash> limits this dump to one shader; it
/// is slow
void TracePointerChain(s32 frame, u32 index, const Shader::Info& stage,
                       Core::MemoryManager* memory, VideoCore::BufferCache& buffer_cache,
                       const VideoCore::PageManager& page_manager) {
    static const u64 chain_hash = [] {
        const char* value = std::getenv("SHADPS4_DRAW_TRACE_CHAIN");
        return value ? std::strtoull(value, nullptr, 16) : 0ULL;
    }();
    if (stage.pgm_hash != chain_hash || stage.user_data.size() < 2) {
        return;
    }
    constexpr VAddr AddressMask = 0xFFFFFFFFFFFFULL;
    const VAddr root = (VAddr(stage.user_data[0]) | VAddr(stage.user_data[1]) << 32) & AddressMask;
    TraceMemory(frame, index, "ud[0:1]", root, memory, buffer_cache, page_manager);
    std::array<u64, 8> pointers{};
    if (!memory->TryReadBacking(root, pointers.data(), sizeof(pointers))) {
        return;
    }
    for (u32 i = 0; i < pointers.size(); ++i) {
        const VAddr pointer = pointers[i] & AddressMask;
        if (pointer < 0x100000000ULL || !memory->IsMappedLocked(pointer, 0x40)) {
            continue;
        }
        const u32 offset = i * 8;
        TraceMemory(frame, index, fmt::format("[ud+{:#x}]", offset), pointer, memory,
                    buffer_cache, page_manager);
        TraceMemory(frame, index, fmt::format("[ud+{:#x}]+0x40", offset), pointer + 0x40, memory,
                    buffer_cache, page_manager);
    }
}

void TraceStageBuffers(s32 frame, u32 index, const Shader::Info& stage) {
    for (u32 i = 0; i < stage.buffers.size(); ++i) {
        const auto& desc = stage.buffers[i];
        if (desc.IsSpecial()) {
            continue;
        }
        const auto vsharp = desc.GetSharp(stage);
        LOG_WARNING(Render_Vulkan,
                    "DrawTrace f={} #{}   {:#x} buf{} {} addr={:#x} size={:#x} stride={} "
                    "records={:#x} fmt={}/{}",
                    frame, index, stage.pgm_hash, i, desc.is_written ? "rw" : "ro",
                    vsharp.base_address, vsharp.GetSize(), u32(vsharp.stride),
                    u32(vsharp.num_records), static_cast<u32>(vsharp.GetDataFmt()),
                    static_cast<u32>(vsharp.GetNumberFmt()));
    }
}

void TraceStageImages(s32 frame, u32 index, const Shader::Info& stage, Core::MemoryManager* memory,
                      VideoCore::BufferCache& buffer_cache,
                      const VideoCore::PageManager& page_manager) {
    for (u32 i = 0; i < stage.images.size(); ++i) {
        const auto& desc = stage.images[i];
        const bool unknown = std::ranges::find(desc.sharp_fetch.offsets,
                                               Shader::UNKNOWN_LOCATION) !=
                             desc.sharp_fetch.offsets.end();
        const auto tsharp = desc.GetSharp(stage);
        LOG_WARNING(Render_Vulkan,
                    "DrawTrace f={} #{}   {:#x} img{} {} addr={:#x} {}x{}x{} pitch={} fmt={}/{} "
                    "tile={} type={} mips={}-{} array={}{}",
                    frame, index, stage.pgm_hash, i, desc.is_written ? "rw" : "ro",
                    tsharp.Address(), u32(tsharp.width) + 1, u32(tsharp.height) + 1,
                    u32(tsharp.depth) + 1, u32(tsharp.pitch) + 1,
                    static_cast<u32>(tsharp.GetDataFmt()), static_cast<u32>(tsharp.GetNumberFmt()),
                    u32(tsharp.tiling_index), u32(tsharp.type), u32(tsharp.base_level),
                    u32(tsharp.last_level), desc.array_size,
                    unknown                 ? " NULL(sharp not flattened)"
                    : tsharp.Address() == 0 ? " NULL"
                                            : "");
        if (tsharp.Address() == 0 && !unknown) {
            // Raw fetch before validation, and where each dword comes from in the flat buffer
            std::array<u32, 8> raw{};
            const auto& fetch = desc.sharp_fetch;
            std::string where;
            for (u32 dw = 0; dw < raw.size(); ++dw) {
                const bool loaded = (fetch.load_mask >> dw) & 1u;
                raw[dw] = loaded && fetch.offsets[dw] < stage.flattened_ud_buf.size()
                              ? stage.flattened_ud_buf[fetch.offsets[dw]]
                              : fetch.immediates[dw];
                where += loaded ? fmt::format(" f{}", fetch.offsets[dw]) : " imm";
            }
            LOG_WARNING(Render_Vulkan,
                        "DrawTrace f={} #{}   {:#x} img{} raw={:08x} {:08x} {:08x} {:08x} {:08x} "
                        "{:08x} {:08x} {:08x} src={}",
                        frame, index, stage.pgm_hash, i, raw[0], raw[1], raw[2], raw[3], raw[4],
                        raw[5], raw[6], raw[7], where);
            std::string ud;
            for (const u32 value : stage.user_data) {
                ud += fmt::format(" {:08x}", value);
            }
            LOG_WARNING(Render_Vulkan, "DrawTrace f={} #{}   {:#x} user_data={}", frame, index,
                        stage.pgm_hash, ud);
            TracePointerChain(frame, index, stage, memory, buffer_cache, page_manager);
        }
    }
}

const Shader::Info* TryGetStage(const Pipeline* pipeline, Shader::SwStage stage) {
    return pipeline ? pipeline->TryGetStage(stage) : nullptr;
}

void TraceGraphics(const AmdGpu::Regs& regs, const GraphicsPipeline* pipeline,
                   std::string_view kind, std::string_view outcome, Core::MemoryManager* memory,
                   VideoCore::BufferCache& buffer_cache,
                   const VideoCore::PageManager& page_manager) {
    auto& tracer = DrawTrace::Instance();
    const s32 frame = tracer.Frame();
    const u32 index = tracer.NextIndex();
    const auto* vs = TryGetStage(pipeline, Shader::SwStage::Vertex);
    const auto* fs = TryGetStage(pipeline, Shader::SwStage::Fragment);
    LOG_WARNING(Render_Vulkan,
                "DrawTrace f={} #{} {} vs={:#x} ps={:#x} ps_addr={:#x} n={} inst={} {}", frame,
                index, kind, vs ? vs->pgm_hash : 0, fs ? fs->pgm_hash : 0,
                regs.ps_program.Address<uintptr_t>(), regs.num_indices,
                regs.num_instances.NumInstances(), outcome);
    for (u32 cb = 0; cb < AmdGpu::NUM_COLOR_BUFFERS; ++cb) {
        const auto& col_buf = regs.color_buffers[cb];
        const u32 target_mask = regs.color_target_mask.GetMask(cb);
        if (!col_buf || !target_mask) {
            continue;
        }
        LOG_WARNING(Render_Vulkan,
                    "DrawTrace f={} #{}   cb{} addr={:#x} fmt={}/{} pitch={} h={} mask={:#x}",
                    frame, index, cb, col_buf.Address(), static_cast<u32>(col_buf.GetDataFmt()),
                    static_cast<u32>(col_buf.GetNumberFmt()), col_buf.Pitch(), col_buf.Height(),
                    target_mask);
    }
    if (regs.depth_buffer.DepthValid() || regs.depth_buffer.StencilValid()) {
        LOG_WARNING(Render_Vulkan,
                    "DrawTrace f={} #{}   db addr={:#x} z_test={} z_write={} stencil={}", frame,
                    index, regs.depth_buffer.DepthAddress(), u32(regs.depth_control.depth_enable),
                    u32(regs.depth_control.depth_write_enable),
                    u32(regs.depth_control.stencil_enable));
    }
    for (const auto* stage : {vs, fs}) {
        if (stage) {
            TraceStageBuffers(frame, index, *stage);
            TraceStageImages(frame, index, *stage, memory, buffer_cache, page_manager);
        }
    }
}

void TraceCompute(const Shader::Info& cs, const AmdGpu::ComputeProgram& cs_program,
                  std::string_view kind, Core::MemoryManager* memory,
                  VideoCore::BufferCache& buffer_cache,
                  const VideoCore::PageManager& page_manager) {
    auto& tracer = DrawTrace::Instance();
    const s32 frame = tracer.Frame();
    const u32 index = tracer.NextIndex();
    LOG_WARNING(Render_Vulkan, "DrawTrace f={} #{} {} cs={:#x} dims={}x{}x{}", frame, index, kind,
                cs.pgm_hash, cs_program.dim_x, cs_program.dim_y, cs_program.dim_z);
    TraceStageBuffers(frame, index, cs);
    TraceStageImages(frame, index, cs, memory, buffer_cache, page_manager);
}

/// SHADPS4_DIAG_PROBE_TABLE=<cs hash> (hex): every 10 s, dumps the reflection-probe table read by SotC's
/// local IBL pass (the layout dynamic_tsharp_array_size was built from). P = *(u64*)(SGPR0:1 + 8); probe k
/// keeps its cubemap slot at P + k*224 + 36 and slot s has its 128-bit T# at P + s*32 + 14480 (S# at +16).
/// Reads go through the backing, so they never fault
void DiagProbeTable(const Shader::Info& cs, const AmdGpu::ComputeProgram& cs_program,
                    Core::MemoryManager* memory) {
    static const u64 target = [] {
        const char* value = std::getenv("SHADPS4_DIAG_PROBE_TABLE");
        return value ? std::strtoull(value, nullptr, 16) : 0ULL;
    }();
    static std::chrono::steady_clock::time_point last{};
    if (target == 0 || cs.pgm_hash != target) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    if (last != decltype(now){} && now - last < std::chrono::seconds{10}) {
        return;
    }
    last = now;
    const auto read = [memory](VAddr addr, void* out, u64 size) {
        return addr != 0 && memory->TryReadBacking(addr, out, size);
    };
    const VAddr ud = VAddr{cs_program.user_data[0]} | (VAddr{cs_program.user_data[1]} << 32);
    VAddr table = 0;
    u32 header = 0;
    if (!read(ud + 8, &table, sizeof(table)) || !read(table, &header, sizeof(header))) {
        LOG_WARNING(Render_Vulkan, "DiagProbeTable: ud={:#x} unreadable", ud);
        return;
    }
    std::map<u32, u32> slots;
    std::string records;
    for (u32 k = 0; k < 64; ++k) {
        u32 slot = 0;
        if (!read(table + k * 224 + 36, &slot, sizeof(slot))) {
            break;
        }
        ++slots[slot];
        records += fmt::format(" {}", slot);
    }
    LOG_WARNING(Render_Vulkan, "DiagProbeTable: ud={:#x} P={:#x} P[0]={} slots(k=0..63):{}", ud,
                table, header, records);
    u32 shown = 0;
    for (const auto& [slot, count] : slots) {
        if (shown++ == 16 || slot > 4096) {
            break;
        }
        std::array<u32, 8> words{};
        if (!read(table + u64{slot} * 32 + 14480, words.data(), sizeof(words))) {
            continue;
        }
        AmdGpu::Image image{};
        std::memcpy(&image, words.data(), 4 * sizeof(u32));
        LOG_WARNING(Render_Vulkan,
                    "DiagProbeTable: slot {} (x{}) T#={:08x} {:08x} {:08x} {:08x} addr={:#x} "
                    "{}x{} fmt={}/{} type={} mips={}-{} tile={} S#={:08x} {:08x} {:08x} {:08x}",
                    slot, count, words[0], words[1], words[2], words[3], image.Address(),
                    u32(image.width) + 1, u32(image.height) + 1, u32(image.data_format),
                    u32(image.num_format), u32(image.type), u32(image.base_level),
                    u32(image.last_level), u32(image.tiling_index), words[4], words[5], words[6],
                    words[7]);
    }
}

/// SHADPS4_CBUF_PROBE=<cs hash>:<buffer>:<dword>[,...] (hash in hex): the SotC TDR root cause (§7.5 of the
/// port guide). Some compute shaders loop up to a constant buffer dword; the device losses showed that dword
/// holding old ring contents (~10^9 iterations) when the dispatch was recorded. For each listed shader the
/// dword is read through the backing when the dispatch is recorded (the value the stream buffer copies);
/// above SHADPS4_CBUF_PROBE_MAX (default 0x100000) the probe logs the words, the tracker state, the last PM4
/// packets of every queue, and starts a watcher thread that logs when the game changes the dword afterwards
void ProbeStaleCbuf(const Shader::Info& cs, Core::MemoryManager* memory,
                    VideoCore::BufferCache& buffer_cache) {
    struct Target {
        u64 hash;
        u32 buffer;
        u32 dword;
        u64 seen;
        u64 stale;
    };
    static std::vector<Target> targets = [] {
        std::vector<Target> parsed;
        const char* value = std::getenv("SHADPS4_CBUF_PROBE");
        for (std::string_view rest = value ? value : ""; !rest.empty();) {
            const size_t comma = rest.find(',');
            const std::string item{rest.substr(0, comma)};
            rest = comma == std::string_view::npos ? std::string_view{} : rest.substr(comma + 1);
            Target target{};
            if (std::sscanf(item.c_str(), "%" SCNx64 ":%u:%u", &target.hash, &target.buffer,
                            &target.dword) == 3) {
                parsed.push_back(target);
                LOG_WARNING(Render_Vulkan, "CbufProbe: watching cs {:#x} buffer {} dword {}",
                            target.hash, target.buffer, target.dword);
            }
        }
        return parsed;
    }();
    static const u32 max_value = [] {
        const char* value = std::getenv("SHADPS4_CBUF_PROBE_MAX");
        return value ? static_cast<u32>(std::strtoul(value, nullptr, 0)) : 0x100000u;
    }();
    static std::atomic<bool> watcher_running{false};
    static u32 reports = 0;

    for (auto& target : targets) {
        if (target.hash != cs.pgm_hash || target.buffer >= cs.buffers.size() ||
            cs.buffers[target.buffer].IsSpecial()) {
            continue;
        }
        const auto vsharp = cs.buffers[target.buffer].GetSharp(cs);
        const VAddr addr = vsharp.base_address + u64{target.dword} * sizeof(u32);
        std::array<u32, 8> words{};
        const VAddr head = Common::AlignDown(addr, sizeof(words));
        if (vsharp.base_address <= 1 || !memory->TryReadBacking(head, words.data(), sizeof(words))) {
            continue;
        }
        const u32 value = words[(addr - head) / sizeof(u32)];
        ++target.seen;
        const bool stale = value > max_value;
        target.stale += stale;
        if (!stale) {
            if (target.seen <= 3 || target.seen % 10000 == 0) {
                LOG_WARNING(Render_Vulkan,
                            "CbufProbe ok cs {:#x}: [{:#x}]={:#x} (seen {}, stale {}) words {:08x}",
                            cs.pgm_hash, addr, value, target.seen, target.stale,
                            fmt::join(words, " "));
            }
            continue;
        }
        if (++reports > 16) {
            if (reports % 100 == 0) {
                LOG_WARNING(Render_Vulkan, "CbufProbe: {} stale dispatches so far", reports);
            }
            continue;
        }
        LOG_WARNING(Render_Vulkan,
                    "CbufProbe STALE cs {:#x}: [{:#x}]={:#x} > {:#x} (seen {}, stale {}) V# base "
                    "{:#x} size {:#x} words@{:#x} {:08x} cpu_modified={} gpu_modified={}",
                    cs.pgm_hash, addr, value, max_value, target.seen, target.stale,
                    vsharp.base_address, vsharp.GetSize(), head, fmt::join(words, " "),
                    buffer_cache.IsRegionCpuModified(head, sizeof(words)),
                    buffer_cache.IsRegionGpuModified(head, sizeof(words)));
        AmdGpu::CpHistory::Dump(96);
        if (!watcher_running.exchange(true)) {
            std::thread([memory, addr, value, hash = cs.pgm_hash] {
                const auto start = std::chrono::steady_clock::now();
                u32 last = value;
                u32 changes = 0;
                while (std::chrono::steady_clock::now() - start < std::chrono::seconds{2} &&
                       changes < 8) {
                    u32 now_value = last;
                    if (memory->TryReadBacking(addr, &now_value, sizeof(now_value)) &&
                        now_value != last) {
                        const auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                                            std::chrono::steady_clock::now() - start)
                                            .count();
                        LOG_WARNING(Render_Vulkan,
                                    "CbufProbe watch cs {:#x}: [{:#x}] {:#x} -> {:#x} after {} us",
                                    hash, addr, last, now_value, us);
                        last = now_value;
                        ++changes;
                    }
                    std::this_thread::sleep_for(std::chrono::microseconds{100});
                }
                LOG_WARNING(Render_Vulkan, "CbufProbe watch cs {:#x}: done, {} changes", hash,
                            changes);
                watcher_running = false;
            }).detach();
        }
    }
}

/// Returns why a texture layout cannot be sized, or null when it is safe
const char* UnsupportedImageLayout(const AmdGpu::Image& tsharp) {
    const auto tile_mode = tsharp.GetTileMode();
    if (!magic_enum::enum_contains(tile_mode)) {
        return "reserved tile mode"; // GetArrayMode: UNREACHABLE
    }
    const auto array_mode = AmdGpu::GetArrayMode(tile_mode);
    if (array_mode == AmdGpu::ArrayMode::ArrayLinearGeneral) {
        return "linear general"; // UpdateSize: UNREACHABLE
    }
    if (!AmdGpu::IsMacroTiled(array_mode)) {
        return nullptr;
    }
    const auto data_fmt = tsharp.GetDataFmt();
    if (AmdGpu::IsBlockCoded(data_fmt)) {
        return "block-compressed format on a macro-tiled mode"; // UpdateSize: ASSERT
    }
    // GetMacroTileExtents: asserts above 8 samples and indexes its table by log2(bpp) - 3
    const u32 samples = tsharp.NumSamples();
    const u32 bpp = AmdGpu::NumBitsPerBlock(data_fmt);
    if (samples > 8 || bpp < 8 || bpp > 128) {
        return "macro-tiled mode with an unsupported sample count or bpp";
    }
    const bool alt = Libraries::Kernel::sceKernelIsNeoMode() && tsharp.alt_tile_mode;
    const auto [pitch_align, height_align] =
        VideoCore::GetMacroTileExtents(tile_mode, bpp, samples, alt);
    if (pitch_align == 0 || height_align == 0) {
        return "macro-tiled mode without tile extents"; // ImageSizeMacroTiled: ASSERT
    }
    return nullptr;
}

} // Anonymous namespace

static Shader::PushData MakeUserData(const AmdGpu::Regs& regs) {
    // TODO(roamic): Add support for multiple viewports and geometry shaders when ViewportIndex
    // is encountered and implemented in the recompiler.
    Shader::PushData push_data{};
    push_data.xoffset = regs.viewport_control.xoffset_enable ? regs.viewports[0].xoffset : 0.f;
    push_data.xscale = regs.viewport_control.xscale_enable ? regs.viewports[0].xscale : 1.f;
    push_data.yoffset = regs.viewport_control.yoffset_enable ? regs.viewports[0].yoffset : 0.f;
    push_data.yscale = regs.viewport_control.yscale_enable ? regs.viewports[0].yscale : 1.f;
    return push_data;
}

Rasterizer::Rasterizer(const Instance& instance_, Scheduler& scheduler_, Runtime& runtime_,
                       AmdGpu::Liverpool* liverpool_)
    : instance{instance_}, scheduler{scheduler_}, runtime{runtime_}, page_manager{this},
      buffer_cache{instance, scheduler, runtime, liverpool_, texture_cache, page_manager},
      texture_cache{instance, scheduler, runtime, liverpool_, buffer_cache, page_manager},
      liverpool{liverpool_}, memory{Core::Memory::Instance()},
      pipeline_cache{instance, scheduler, liverpool, buffer_cache.GetSparsePageShift()},
      host_markers_enabled{EmulatorSettings.IsVkHostMarkersEnabled()},
      guest_markers_enabled{EmulatorSettings.IsVkGuestMarkersEnabled()},
      lod_stats_enabled{EmulatorSettings.IsLodStatsFromBindings()},
      periodic_flush_commands{EmulatorSettings.GetPeriodicFlushCommands()},
      recording_cuts{EmulatorSettings.IsCpRecordingCuts()},
      recording_cuts_toggle{Common::PerfStats::AbToggleFollows("cp_recording_cuts")},
      mapped_page_table{EmulatorSettings.IsMappedPageTable()
                            ? std::make_unique<VideoCore::MappedPageTable>()
                            : nullptr},
      mapped_page_table_toggle{Common::PerfStats::AbToggleFollows("mapped_page_table")},
      mapped_page_table_verify{mapped_page_table &&
                               std::getenv("SHADPS4_MAPPED_PAGE_TABLE_VERIFY") != nullptr},
      incremental_bind{EmulatorSettings.IsIncrementalBind()},
      incremental_bind_toggle{Common::PerfStats::AbToggleFollows("incremental_bind")},
      incremental_bind_verify{incremental_bind &&
                              std::getenv("SHADPS4_INCREMENTAL_BIND_VERIFY") != nullptr},
      tsharp_cache{EmulatorSettings.IsTsharpCache() ? std::make_unique<TsharpCache>() : nullptr},
      tsharp_cache_toggle{Common::PerfStats::AbToggleFollows("tsharp_cache")},
      tsharp_cache_verify{tsharp_cache && std::getenv("SHADPS4_TSHARP_CACHE_VERIFY") != nullptr},
      depth_target_sampled_layout{EmulatorSettings.IsDepthTargetSampledLayout()},
      depth_target_sampled_toggle{
          Common::PerfStats::AbToggleFollows("depth_target_sampled_layout")},
      gcn_unordered_dispatches{EmulatorSettings.IsGcnUnorderedDispatches()},
      gcn_unordered_toggle{Common::PerfStats::AbToggleFollows("gcn_unordered_dispatches")} {
    if (!EmulatorSettings.IsNullGPU()) {
        liverpool->BindRasterizer(this);
    }
    memory->SetRasterizer(this);
    if (mapped_page_table) {
        LOG_WARNING(Render_Vulkan, "Workaround mapped_page_table enabled{}",
                    mapped_page_table_verify ? " (verify)" : "");
    }
    if (incremental_bind) {
        LOG_WARNING(Render_Vulkan, "Workaround incremental_bind enabled{}",
                    incremental_bind_verify ? " (verify)" : "");
    }
    if (tsharp_cache) {
        // Enable page stamps before registering the first image so an unstamped page correctly
        // means that no image has changed there
        texture_cache.EnablePageStamps();
        LOG_WARNING(Render_Vulkan, "Workaround tsharp_cache enabled{}",
                    tsharp_cache_verify ? " (verify)" : "");
    }

    scheduler.SetSessionCallback([this] { buffer_cache.FlushSyncBatch(true); });
    pipeline_cache.read_clean_memory = [this](VAddr addr, void* out, u64 size) {
        return ReadCleanMemory(addr, out, size);
    };
    pipeline_cache.is_null_image = [this](const AmdGpu::Image& tsharp,
                                          const Shader::ImageResource& image_desc) {
        std::optional<VideoCore::TextureCache::ImageDesc> bound_desc;
        return !CheckBindableImage(tsharp, image_desc, bound_desc, false);
    };

    scheduler.SetSubmitCallback([this](Vulkan::SubmitInfo& info) {
        runtime.FlushBarriers();
        buffer_cache.SubmitPendingArenaBinds(info);
    });

    if (GpuCheckpoints::Enabled()) {
        scheduler.TrackGpuCommands();
        if (GpuCheckpoints::DiagEnabled() && GpuCheckpoints::DiagGpuCopies() != 0) {
            // Uncached host memory: the device loss report reads the words without an invalidate
            diag_capture_buffer = std::make_unique<VideoCore::Buffer>(
                instance, 0, DiagCaptureBufferSize, VideoCore::MemoryType::HostUncached,
                "GPU diag capture");
        }
    }
    scheduler.EnableGpuTiming();
    if (periodic_flush_commands != 0) {
        LOG_WARNING(Render_Vulkan, "Workaround periodic_flush_commands enabled: submit every {} commands",
                    periodic_flush_commands);
        if (!EmulatorSettings.IsReadbackAhead()) {
            LOG_WARNING(Render_Vulkan, "periodic_flush_commands without readback_ahead: one device "
                                       "loss (WriteInvalid) was seen in this combination");
        }
    }
    if (const u32 spin_us = EmulatorSettings.GetWaitSpinUs(); spin_us != 0) {
        LOG_WARNING(Render_Vulkan, "Workaround wait_spin_us enabled: poll up to {} us before waits",
                    spin_us);
    }
    if (const u32 aniso = EmulatorSettings.GetForceAnisotropy(); aniso != 0) {
        LOG_WARNING(Render_Vulkan, "Workaround force_anisotropy enabled: {}x (device max {}x)",
                    aniso, instance.MaxSamplerAnisotropy());
    }
    if (depth_target_sampled_layout) {
        LOG_WARNING(Render_Vulkan, "Workaround depth_target_sampled_layout enabled");
    }
    if (gcn_unordered_dispatches) {
        LOG_WARNING(Render_Vulkan, "Workaround gcn_unordered_dispatches enabled");
    }
    scheduler.SkipRedundantPipelineBinds(EmulatorSettings.IsGpuOverheadCuts());
}

Rasterizer::~Rasterizer() = default;

bool Rasterizer::FilterDraw() {
    const auto& regs = liverpool->regs;
    if (regs.color_control.mode == AmdGpu::ColorControl::OperationMode::EliminateFastClear) {
        // Clears the render target if FCE is launched before any draws
        EliminateFastClear();
        return false;
    }
    if (regs.color_control.mode == AmdGpu::ColorControl::OperationMode::FmaskDecompress) {
        // TODO: check for a valid MRT1 to promote the draw to the resolve pass.
        LOG_TRACE(Render_Vulkan, "FMask decompression pass skipped");
        ScopedMarkerInsert("FmaskDecompress");
        return false;
    }
    if (regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Resolve) {
        LOG_TRACE(Render_Vulkan, "Resolve pass");
        Resolve();
        return false;
    }
    if (regs.primitive_type == AmdGpu::PrimitiveType::None) {
        LOG_TRACE(Render_Vulkan, "Primitive type 'None' skipped");
        ScopedMarkerInsert("PrimitiveTypeNone");
        return false;
    }

    const bool cb_disabled =
        regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Disable;
    const auto depth_copy =
        regs.depth_render_override.force_z_dirty && regs.depth_render_override.force_z_valid &&
        regs.depth_buffer.DepthValid() && regs.depth_buffer.DepthWriteValid() &&
        regs.depth_buffer.DepthAddress() != regs.depth_buffer.DepthWriteAddress();
    const auto stencil_copy =
        regs.depth_render_override.force_stencil_dirty &&
        regs.depth_render_override.force_stencil_valid && regs.depth_buffer.StencilValid() &&
        regs.depth_buffer.StencilWriteValid() &&
        regs.depth_buffer.StencilAddress() != regs.depth_buffer.StencilWriteAddress();
    if (cb_disabled && (depth_copy || stencil_copy)) {
        // Games may disable color buffer and enable force depth/stencil dirty and valid to
        // do a copy from one depth-stencil surface to another, without a pixel shader.
        // We need to detect this case and perform the copy, otherwise it will have no effect.
        LOG_TRACE(Render_Vulkan, "Performing depth-stencil override copy");
        DepthStencilCopy(depth_copy, stencil_copy);
        return false;
    }

    return true;
}

void Rasterizer::PrepareRenderState(const GraphicsPipeline* pipeline) {
    depth_target_shared = depth_target_sampled_layout &&
                          Common::PerfStats::AbToggleOn(depth_target_sampled_toggle);
    // Prefetch render targets to handle overlaps with bound textures (e.g. mipgen)
    const auto& key = pipeline->GetGraphicsKey();
    const auto& regs = liverpool->regs;
    if (regs.color_control.degamma_enable) {
        LOG_WARNING(Render_Vulkan, "Color buffers require gamma correction");
    }

    const bool skip_cb_binding =
        regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Disable;
    for (s32 cb = 0; cb < std::bit_width(key.mrt_mask); ++cb) {
        auto& [image_id, desc] = cb_descs[cb];
        const auto& col_buf = regs.color_buffers[cb];
        const u32 target_mask = regs.color_target_mask.GetMask(cb);
        if (skip_cb_binding || !col_buf || !target_mask || (key.mrt_mask & (1 << cb)) == 0) {
            image_id = {};
            continue;
        }
        const auto& hint = liverpool->last_cb_extent[cb];
        std::construct_at(&desc, col_buf, hint);
        image_id = bound_images.emplace_back(texture_cache.FindImage(desc));
        auto& image = texture_cache.GetImage(image_id);
        image.binding.is_target = 1u;
    }

    if ((regs.depth_control.depth_enable && regs.depth_buffer.DepthValid()) ||
        (regs.depth_control.stencil_enable && regs.depth_buffer.StencilValid())) {
        const auto htile_address = regs.depth_htile_data_base.GetAddress();
        const auto& hint = liverpool->last_db_extent;
        auto& [image_id, desc] = db_desc;
        std::construct_at(&desc, regs.depth_buffer, regs.depth_view, regs.depth_control,
                          htile_address, hint);
        image_id = bound_images.emplace_back(texture_cache.FindImage(desc));
        auto& image = texture_cache.GetImage(image_id);
        image.binding.is_target = 1u;
    } else {
        db_desc.first = {};
    }
}

static std::pair<u32, u32> GetDrawOffsets(const AmdGpu::Regs& regs, const Shader::Info& info,
                                          const Shader::Gcn::FetchShaderData& fetch_shader) {
    u32 vertex_offset = regs.index_offset;
    u32 instance_offset = 0;
    if (!fetch_shader.Empty()) {
        if (vertex_offset == 0 && fetch_shader.vertex_offset_sgpr != -1) {
            vertex_offset = info.user_data[fetch_shader.vertex_offset_sgpr];
        }
        if (fetch_shader.instance_offset_sgpr != -1) {
            instance_offset = info.user_data[fetch_shader.instance_offset_sgpr];
        }
    }
    return {vertex_offset, instance_offset};
}

void Rasterizer::EliminateFastClear() {
    auto& col_buf = liverpool->regs.color_buffers[0];
    if (!col_buf || !col_buf.info.fast_clear) {
        return;
    }
    VideoCore::TextureCache::ImageDesc desc(col_buf, liverpool->last_cb_extent[0]);
    const auto image_id = texture_cache.FindImage(desc);
    const auto& image_view = texture_cache.FindRenderTarget(image_id, desc);
    if (!texture_cache.IsMetaCleared(col_buf.CmaskAddress(), col_buf.view.slice_start)) {
        return;
    }
    for (u32 slice = col_buf.view.slice_start; slice <= col_buf.view.slice_max; ++slice) {
        texture_cache.TouchMeta(col_buf.CmaskAddress(), slice, false);
    }
    auto& image = texture_cache.GetImage(image_id);
    const auto clear_value = LiverpoolToVK::ColorBufferClearValue(col_buf);

    ScopeMarkerBegin(fmt::format("EliminateFastClear:MRT={:#x}:M={:#x}", col_buf.Address(),
                                 col_buf.CmaskAddress()));
    runtime.ClearImage(&image, desc.view_info.range, clear_value);
    ScopeMarkerEnd();
}

void Rasterizer::Draw(bool is_indexed, u32 index_offset) {
    RENDERER_TRACE;
    const Common::PerfStats::ScopedTimer perf_timer{Common::PerfStats::Id::Draws,
                                                    Common::PerfStats::Id::DrawNs};
    NoteRingCommand(false);

    scheduler.PopPendingOperations();

    const auto& regs = liverpool->regs;
    const bool trace = DrawTrace::Instance().Active();
    const std::string_view trace_kind = is_indexed ? "DrawIndexed" : "Draw";
    if (!FilterDraw()) {
        if (trace) {
            TraceGraphics(regs, nullptr, trace_kind, "skipped: filtered", memory, buffer_cache,
                          page_manager);
        }
        return;
    }

    const GraphicsPipeline* pipeline = pipeline_cache.GetGraphicsPipeline();
    if (!pipeline) {
        if (trace) {
            TraceGraphics(regs, nullptr, trace_kind, "skipped: no pipeline", memory, buffer_cache,
                          page_manager);
        }
        return;
    }
    if (trace) {
        TraceGraphics(regs, pipeline, trace_kind, "", memory, buffer_cache, page_manager);
    }

    PrepareRenderState(pipeline);
    if (!BindResources(pipeline)) {
        return;
    }
    const auto state = BeginRendering(pipeline);

    BindVertexBuffers(pipeline);
    if (is_indexed) {
        BindIndexBuffer(index_offset);
    }

    if (needs_barrier) {
        runtime.FlushBarriers();
    }

    RefreshImageDescriptorLayouts();
    pipeline->BindResources(set_writes, push_data);
    UpdateDynamicState(pipeline, is_indexed);
    scheduler.BeginRendering(state);

    const auto& vs_info = pipeline->GetStage(Shader::SwStage::Vertex);
    const auto& fetch_shader = pipeline->GetFetchShader();
    const auto [vertex_offset, instance_offset] = GetDrawOffsets(regs, vs_info, fetch_shader);

    const auto cmdbuf = scheduler.CommandBuffer();
    scheduler.BindPipeline(vk::PipelineBindPoint::eGraphics, pipeline->Handle());
    CountPipelineBind(cmdbuf, vk::PipelineBindPoint::eGraphics, pipeline->Handle());

    if (GpuCheckpoints::Enabled()) {
        const auto* fs_info = TryGetStage(pipeline, Shader::SwStage::Fragment);
        const auto* record = GpuCheckpoints::Push(
            is_indexed ? GpuCheckpoints::Kind::DrawIndexed : GpuCheckpoints::Kind::Draw,
            vs_info.pgm_hash, fs_info ? fs_info->pgm_hash : 0, regs.num_indices,
            regs.num_instances.NumInstances(), 0);
        if (instance.IsNvDiagnosticCheckpointsSupported()) {
            cmdbuf.setCheckpointNV(record);
        }
    }

    if (is_indexed) {
        cmdbuf.drawIndexed(regs.num_indices, regs.num_instances.NumInstances(), 0,
                           s32(vertex_offset), instance_offset);
    } else {
        cmdbuf.draw(regs.num_indices, regs.num_instances.NumInstances(), vertex_offset,
                    instance_offset);
    }
    DebugState.IncDrawCall();

    OnGpuCommandRecorded();
    ResetBindings(false);
}

void Rasterizer::DrawIndirect(bool is_indexed, VAddr arg_address, u32 offset, u32 stride,
                              u32 max_count, VAddr count_address, u16 vertex_sgpr_offset,
                              u16 instance_sgpr_offset) {
    RENDERER_TRACE;
    const Common::PerfStats::ScopedTimer perf_timer{Common::PerfStats::Id::Draws,
                                                    Common::PerfStats::Id::DrawNs};
    NoteRingCommand(false);

    scheduler.PopPendingOperations();

    const bool trace = DrawTrace::Instance().Active();
    const std::string_view trace_kind = is_indexed ? "DrawIndexedIndirect" : "DrawIndirect";
    if (!FilterDraw()) {
        if (trace) {
            TraceGraphics(liverpool->regs, nullptr, trace_kind, "skipped: filtered", memory,
                          buffer_cache, page_manager);
        }
        return;
    }

    const DrawIndirectParams params = {
        .vertex_sgpr_offset = vertex_sgpr_offset,
        .instance_sgpr_offset = instance_sgpr_offset,
    };
    const GraphicsPipeline* pipeline = pipeline_cache.GetGraphicsPipeline(params);
    if (!pipeline) {
        if (trace) {
            TraceGraphics(liverpool->regs, nullptr, trace_kind, "skipped: no pipeline", memory,
                          buffer_cache, page_manager);
        }
        return;
    }
    if (trace) {
        TraceGraphics(liverpool->regs, pipeline, trace_kind, "", memory, buffer_cache,
                      page_manager);
    }

    PrepareRenderState(pipeline);
    if (!BindResources(pipeline)) {
        return;
    }
    const auto state = BeginRendering(pipeline);

    BindVertexBuffers(pipeline);
    if (is_indexed) {
        BindIndexBuffer();
    }

    const auto [buffer, base] =
        buffer_cache.ObtainBuffer(arg_address + offset, stride * max_count, false);
    needs_barrier |= runtime.IsBufferAccessed(buffer, base, stride * max_count);

    const VideoCore::Buffer* count_buffer;
    u64 count_offset;
    if (count_address != 0) {
        std::tie(count_buffer, count_offset) = buffer_cache.ObtainBuffer(count_address, 4, false);
        needs_barrier |= runtime.IsBufferAccessed(count_buffer, count_offset, 4);
    }

    if (needs_barrier) {
        runtime.FlushBarriers();
    }

    RefreshImageDescriptorLayouts();
    pipeline->BindResources(set_writes, push_data);
    UpdateDynamicState(pipeline, is_indexed);
    scheduler.BeginRendering(state);

    const auto cmdbuf = scheduler.CommandBuffer();
    scheduler.BindPipeline(vk::PipelineBindPoint::eGraphics, pipeline->Handle());
    CountPipelineBind(cmdbuf, vk::PipelineBindPoint::eGraphics, pipeline->Handle());

    if (GpuCheckpoints::Enabled()) {
        const auto* vs_info = TryGetStage(pipeline, Shader::SwStage::Vertex);
        const auto* fs_info = TryGetStage(pipeline, Shader::SwStage::Fragment);
        const auto* record = GpuCheckpoints::Push(
            is_indexed ? GpuCheckpoints::Kind::DrawIndexedIndirect
                       : GpuCheckpoints::Kind::DrawIndirect,
            vs_info ? vs_info->pgm_hash : 0, fs_info ? fs_info->pgm_hash : 0, max_count,
            count_address != 0, 0, arg_address + offset);
        if (instance.IsNvDiagnosticCheckpointsSupported()) {
            cmdbuf.setCheckpointNV(record);
        }
    }

    if (is_indexed) {
        ASSERT(sizeof(VkDrawIndexedIndirectCommand) == stride);

        if (count_address != 0) {
            cmdbuf.drawIndexedIndirectCount(buffer->Handle(), base, count_buffer->Handle(),
                                            count_offset, max_count, stride);
        } else {
            cmdbuf.drawIndexedIndirect(buffer->Handle(), base, max_count, stride);
        }
        DebugState.IncDrawCall();
    } else {
        ASSERT(sizeof(VkDrawIndirectCommand) == stride);

        if (count_address != 0) {
            cmdbuf.drawIndirectCount(buffer->Handle(), base, count_buffer->Handle(), count_offset,
                                     max_count, stride);
        } else {
            cmdbuf.drawIndirect(buffer->Handle(), base, max_count, stride);
        }
        DebugState.IncDrawCall();
    }

    OnGpuCommandRecorded();
    ResetBindings(false);
}

void Rasterizer::DispatchDirect() {
    RENDERER_TRACE;
    const Common::PerfStats::ScopedTimer perf_timer{Common::PerfStats::Id::Dispatches,
                                                    Common::PerfStats::Id::DispatchNs};
    unordered_ring = NoteRingCommand(true);

    scheduler.PopPendingOperations();

    const auto& cs_program = liverpool->GetCsRegs();
    const ComputePipeline* pipeline = pipeline_cache.GetComputePipeline();
    if (!pipeline) {
        return;
    }

    const auto& cs = pipeline->GetStage(Shader::SwStage::Compute);
    game_dispatch_accesses = gcn_unordered_dispatches && !cs.uses_dma;
    if (!game_dispatch_accesses) {
        unordered_ring = -1;
    }
    if (DrawTrace::Instance().Active()) {
        TraceCompute(cs, cs_program, "Dispatch", memory, buffer_cache, page_manager);
    }
    DiagProbeTable(cs, cs_program, memory);
    if (AmdGpu::CpHistory::Enabled()) {
        ProbeStaleCbuf(cs, memory, buffer_cache);
    }
    if (ExecuteShaderHLE(cs, liverpool->regs, cs_program, *this)) {
        return;
    }

    if (!BindResources(pipeline)) {
        return;
    }

    if (needs_barrier) {
        runtime.FlushBarriers(unordered_ring);
    }

    scheduler.EndRendering();
    RefreshImageDescriptorLayouts();
    pipeline->BindResources(set_writes, push_data);
    if (GpuCheckpoints::DiagEnabled()) {
        CaptureBufferHeads();
    }

    const auto cmdbuf = scheduler.CommandBuffer();
    scheduler.BindPipeline(vk::PipelineBindPoint::eCompute, pipeline->Handle());
    CountPipelineBind(cmdbuf, vk::PipelineBindPoint::eCompute, pipeline->Handle());
    if (GpuCheckpoints::Enabled()) {
        const auto* record =
            GpuCheckpoints::Push(GpuCheckpoints::Kind::Dispatch, cs.pgm_hash, 0, cs_program.dim_x,
                                 cs_program.dim_y, cs_program.dim_z);
        if (instance.IsNvDiagnosticCheckpointsSupported()) {
            cmdbuf.setCheckpointNV(record);
        }
    }
    cmdbuf.dispatch(cs_program.dim_x, cs_program.dim_y, cs_program.dim_z);
    DebugState.IncDispatch();

    OnGpuCommandRecorded();
    ResetBindings(true);
}

void Rasterizer::DispatchIndirect(VAddr address, u32 offset, u32 size) {
    RENDERER_TRACE;
    const Common::PerfStats::ScopedTimer perf_timer{Common::PerfStats::Id::Dispatches,
                                                    Common::PerfStats::Id::DispatchNs};
    unordered_ring = NoteRingCommand(true);

    scheduler.PopPendingOperations();

    const auto& cs_program = liverpool->GetCsRegs();
    const ComputePipeline* pipeline = pipeline_cache.GetComputePipeline();
    if (!pipeline) {
        return;
    }
    game_dispatch_accesses =
        gcn_unordered_dispatches && !pipeline->GetStage(Shader::SwStage::Compute).uses_dma;
    if (!game_dispatch_accesses) {
        unordered_ring = -1;
    }
    if (DrawTrace::Instance().Active()) {
        TraceCompute(pipeline->GetStage(Shader::SwStage::Compute), cs_program, "DispatchIndirect",
                     memory, buffer_cache, page_manager);
    }
    if (AmdGpu::CpHistory::Enabled()) {
        ProbeStaleCbuf(pipeline->GetStage(Shader::SwStage::Compute), memory, buffer_cache);
    }

    if (!BindResources(pipeline)) {
        return;
    }

    const auto [buffer, base] = buffer_cache.ObtainBuffer(address + offset, size, false);
    // Keep a barrier for writes to indirect dispatch arguments
    // The command processor reads those counts and a race can hang the GPU
    if (runtime.IsBufferAccessed(buffer, base, size)) {
        needs_barrier = true;
        unordered_ring = -1;
    }

    if (needs_barrier) {
        runtime.FlushBarriers(unordered_ring);
    }

    scheduler.EndRendering();
    RefreshImageDescriptorLayouts();
    pipeline->BindResources(set_writes, push_data);
    if (GpuCheckpoints::DiagEnabled()) {
        CaptureBufferHeads();
        if (GpuCheckpoints::DiagGpuCopies() > 1) {
            if (const u32* args = CaptureWords(buffer->Handle(), base, 3)) {
                diag_detail.args = args;
                diag_detail.num_args = 3;
            }
        }
    }

    const auto cmdbuf = scheduler.CommandBuffer();
    scheduler.BindPipeline(vk::PipelineBindPoint::eCompute, pipeline->Handle());
    CountPipelineBind(cmdbuf, vk::PipelineBindPoint::eCompute, pipeline->Handle());
    if (GpuCheckpoints::Enabled()) {
        const auto& cs = pipeline->GetStage(Shader::SwStage::Compute);
        const auto* record = GpuCheckpoints::Push(GpuCheckpoints::Kind::DispatchIndirect,
                                                  cs.pgm_hash, 0, 0, 0, 0, address + offset);
        if (instance.IsNvDiagnosticCheckpointsSupported()) {
            cmdbuf.setCheckpointNV(record);
        }
    }
    cmdbuf.dispatchIndirect(buffer->Handle(), base);
    DebugState.IncDispatch();

    OnGpuCommandRecorded();
    ResetBindings(true);
}

void Rasterizer::OnGpuCommandRecorded() {
    if (!GpuCheckpoints::Enabled()) {
        return;
    }
    if (GpuCheckpoints::DiagEnabled()) {
        GpuCheckpoints::AttachDetail(diag_detail);
    }
    // For diagnosis, wait every N guest commands so the ledger can locate the submit that lost the device
    static const u32 sync_every = [] {
        const char* value = std::getenv("SHADPS4_GPU_SYNC_EVERY");
        return value ? static_cast<u32>(std::strtoul(value, nullptr, 10)) : 0u;
    }();
    static const u32 sync_after_sec = [] {
        const char* value = std::getenv("SHADPS4_GPU_SYNC_AFTER_SEC");
        return value ? static_cast<u32>(std::strtoul(value, nullptr, 10)) : 0u;
    }();
    if (sync_every == 0 || ++diag_commands_since_sync < sync_every) {
        return;
    }
    if (std::chrono::steady_clock::now() - diag_start < std::chrono::seconds{sync_after_sec}) {
        return;
    }
    diag_commands_since_sync = 0;
    scheduler.Finish();
}

const u32* Rasterizer::CaptureWords(vk::Buffer buffer, u64 offset, u32 num_words) {
    // Capture the words the GPU is about to read so a device loss report shows their values
    // A remaining sentinel shows where execution stopped; the ring overwrites old captures
    if (!diag_capture_buffer) {
        return nullptr;
    }
    const u64 size = u64(num_words) * sizeof(u32);
    if (diag_capture_offset + size > DiagCaptureBufferSize) {
        diag_capture_offset = 0;
    }
    const u64 capture_offset = diag_capture_offset;
    diag_capture_offset += Common::AlignUp(size, 64);
    auto* words = reinterpret_cast<u32*>(diag_capture_buffer->mapped_data.data() + capture_offset);
    std::fill_n(words, num_words, GpuCheckpoints::CaptureSentinel);
    diag_capture_buffer->Flush(capture_offset, size);
    const auto cmdbuf = scheduler.CommandBuffer();
    const vk::MemoryBarrier2 before = {
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &before});
    cmdbuf.copyBuffer(buffer, diag_capture_buffer->Handle(),
                      vk::BufferCopy{offset, capture_offset, size});
    const vk::MemoryBarrier2 after = {
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eHost | vk::PipelineStageFlagBits2::eAllCommands,
        .dstAccessMask = vk::AccessFlagBits2::eHostRead | vk::AccessFlagBits2::eMemoryRead,
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &after});
    return words;
}

void Rasterizer::CaptureBufferHeads() {
    // Must run outside of a render pass, after the bindings were resolved
    const u32 count = std::min(diag_detail.num_buffers, GpuCheckpoints::DiagGpuCopies());
    for (u32 i = 0; i < count; ++i) {
        const auto& source = diag_sources[i];
        if (source.buffer && source.range >= GpuCheckpoints::CaptureWords * sizeof(u32)) {
            diag_detail.buffers[i].words =
                CaptureWords(source.buffer, source.offset, GpuCheckpoints::CaptureWords);
        }
    }
}

u64 Rasterizer::Flush() {
    const u64 current_tick = scheduler.CurrentTick();
    SubmitInfo info{};
    scheduler.Flush(info);
    return current_tick;
}

void Rasterizer::Finish() {
    scheduler.Finish();
}

void Rasterizer::OnSubmit() {
    buffer_cache.TickFrame();
    texture_cache.ProcessDownloadImages();
    texture_cache.RunGarbageCollector();
    runtime.TickFrame();
}

void Rasterizer::OnFence() {
    texture_cache.ProcessDownloadImages();
    buffer_cache.FlushSyncBatch();
}

bool Rasterizer::BindResources(const Pipeline* pipeline) {
    if (IsComputeImageCopy(pipeline) || IsComputeMetaClear(pipeline) ||
        IsComputeImageClear(pipeline)) {
        return false;
    }

    set_write_index = 0;
    set_writes.clear();
    buffer_infos.clear();
    image_infos.clear();
    image_descriptor_refs.clear();
    if (GpuCheckpoints::DiagEnabled()) {
        diag_detail.Reset();
    }
    bind_cache = nullptr;
    if (incremental_bind && Common::PerfStats::AbToggleOn(incremental_bind_toggle)) {
        auto& slot = pipeline->BindCacheSlot();
        if (!slot) {
            slot = std::make_shared<BindCache>();
        }
        bind_cache = static_cast<BindCache*>(slot.get());
        bind_generations = {
            .sync_flushes = buffer_cache.SyncBatchGeneration(),
            .image_sets = texture_cache.ImageSetGeneration(),
            .maps = buffer_cache.MappingGeneration(),
            .arenas = buffer_cache.ArenaGeneration(),
        };
        bind_cache_buffer_index = 0;
        bind_cache_image_index = 0;
    }
    tsharp_cache_on = tsharp_cache && Common::PerfStats::AbToggleOn(tsharp_cache_toggle);

    bool uses_dma = false;

    // Bind resource buffers and textures.
    Shader::Backend::Bindings binding{};
    push_data = MakeUserData(liverpool->regs);
    for (const auto* stage : pipeline->GetStages()) {
        if (!stage) {
            continue;
        }
        set_writes.resize(set_writes.size() + stage->buffers.size() + stage->images.size() +
                          stage->samplers.size());
        stage->PushUd(binding, push_data);
        BindBuffers(*stage, binding, push_data);
        BindTextures(*stage, binding);
        uses_dma |= stage->uses_dma;
    }

    if (uses_dma) {
        buffer_cache.SynchronizeDmaBuffers();
    }
    diag_detail.uses_dma = uses_dma;

    return true;
}

void Rasterizer::BindVertexBuffers(const GraphicsPipeline* pipeline) {
    const auto& regs = liverpool->regs;
    VertexInputs<vk::VertexInputAttributeDescription2EXT> attributes;
    VertexInputs<vk::VertexInputBindingDescription2EXT> bindings;
    VertexInputs<vk::VertexInputBindingDivisorDescriptionEXT> divisors;
    VertexInputs<AmdGpu::Buffer> guest_buffers;
    pipeline->GetVertexInputs(attributes, bindings, divisors, guest_buffers,
                              regs.vgt_instance_step_rate_0, regs.vgt_instance_step_rate_1);

    if (instance.IsVertexInputDynamicState()) {
        // Update current vertex inputs.
        const auto cmdbuf = scheduler.CommandBuffer();
        cmdbuf.setVertexInputEXT(bindings, attributes);
    }

    if (bindings.empty()) {
        // If there are no bindings, there is nothing further to do.
        return;
    }

    struct BufferRange {
        VAddr base_address;
        VAddr end_address;
        const VideoCore::Buffer* buffer;
        u64 offset;

        [[nodiscard]] size_t GetSize() const {
            return end_address - base_address;
        }
    };

    // Build list of ranges covering the requested buffers
    VertexInputs<BufferRange> ranges{};
    for (const auto& buffer : guest_buffers) {
        if (buffer.base_address != 0 && buffer.GetSize() > 0) {
            ranges.emplace_back(buffer.base_address, buffer.base_address + buffer.GetSize());
        }
    }

    // Merge connecting ranges together
    VertexInputs<BufferRange> ranges_merged{};
    if (!ranges.empty()) {
        std::ranges::sort(ranges, [](const BufferRange& lhv, const BufferRange& rhv) {
            return lhv.base_address < rhv.base_address;
        });
        ranges_merged.emplace_back(ranges[0]);
        for (auto range : ranges) {
            auto& prev_range = ranges_merged.back();
            if (prev_range.end_address < range.base_address) {
                ranges_merged.emplace_back(range);
            } else {
                prev_range.end_address = std::max(prev_range.end_address, range.end_address);
            }
        }
    }

    // Map buffers for merged ranges
    for (auto& range : ranges_merged) {
        const u64 size = memory->ClampRangeSize(range.base_address, range.GetSize());
        std::tie(range.buffer, range.offset) =
            buffer_cache.ObtainBuffer(range.base_address, size, false);
        needs_barrier |= runtime.IsBufferAccessed(range.buffer, range.offset, size);
    }

    // Bind vertex buffers
    VertexInputs<vk::Buffer> host_buffers;
    VertexInputs<vk::DeviceSize> host_offsets;
    VertexInputs<vk::DeviceSize> host_sizes;
    VertexInputs<vk::DeviceSize> host_strides;
    for (const auto& buffer : guest_buffers) {
        if (buffer.base_address != 0 && buffer.GetSize() > 0) {
            const auto host_buffer_info =
                std::ranges::find_if(ranges_merged, [&](const BufferRange& range) {
                    return buffer.base_address >= range.base_address &&
                           buffer.base_address < range.end_address;
                });
            ASSERT(host_buffer_info != ranges_merged.cend());
            host_buffers.emplace_back(host_buffer_info->buffer->Handle());
            host_offsets.push_back(host_buffer_info->offset + buffer.base_address -
                                   host_buffer_info->base_address);
        } else {
            host_buffers.emplace_back(VK_NULL_HANDLE);
            host_offsets.push_back(0);
        }
        host_sizes.push_back(buffer.GetSize());
        host_strides.push_back(buffer.GetStride());
    }

    const auto cmdbuf = scheduler.CommandBuffer();
    const auto num_buffers = guest_buffers.size();
    if (instance.IsVertexInputDynamicState()) {
        cmdbuf.bindVertexBuffers(0, num_buffers, host_buffers.data(), host_offsets.data());
    } else {
        cmdbuf.bindVertexBuffers2(0, num_buffers, host_buffers.data(), host_offsets.data(),
                                  host_sizes.data(), host_strides.data());
    }
}

void Rasterizer::BindIndexBuffer(u32 index_offset) {
    const auto& regs = liverpool->regs;

    // Figure out index type and size.
    const bool is_index16 = regs.index_buffer_type.index_type == AmdGpu::IndexType::Index16;
    const vk::IndexType index_type = is_index16 ? vk::IndexType::eUint16 : vk::IndexType::eUint32;
    const u32 index_size = is_index16 ? sizeof(u16) : sizeof(u32);
    const VAddr index_address =
        regs.index_base_address.Address<VAddr>() + index_offset * index_size;

    // Bind index buffer.
    const u32 index_buffer_size = regs.num_indices * index_size;
    const auto [buffer, offset] =
        buffer_cache.ObtainBuffer(index_address, index_buffer_size, false);
    needs_barrier |= runtime.IsBufferAccessed(buffer, offset, index_buffer_size);
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.bindIndexBuffer(buffer->Handle(), offset, index_type);
}

void Rasterizer::ResetBindings(bool is_compute) {
    for (auto& image_id : bound_images) {
        texture_cache.GetImage(image_id).binding = {};
    }
    for (const auto [buffer, offset, size, is_written] : bound_buffers) {
        const auto dst_stage = is_compute ? vk::PipelineStageFlagBits2::eComputeShader
                                          : vk::PipelineStageFlagBits2::eAllGraphics;
        const auto write_flag =
            is_written ? vk::AccessFlagBits2::eShaderWrite : vk::AccessFlagBits2::eNone;
        runtime.AccessBuffer(buffer, offset, size, dst_stage,
                             vk::AccessFlagBits2::eShaderRead | write_flag,
                             is_compute && game_dispatch_accesses);
    }
    unordered_ring = -1;
    game_dispatch_accesses = false;
    bound_images.clear();
    bound_buffers.clear();
    needs_barrier = false;
    // The draw or dispatch is recorded: its written bindings belong to the current command buffer
    buffer_cache.CloseGpuWrites();
    scheduler.CountRecordedCommand();
    FlushPeriodic();
}

// GCN can overlap consecutive dispatches until the game sends a sync packet
// Pass the ring to the runtime so it can skip unnecessary memory barriers
// Only guest dispatch accesses without DMA are eligible
s8 Rasterizer::NoteRingCommand(bool is_dispatch) {
    if (!gcn_unordered_dispatches) {
        return -1;
    }
    const u8 ring = command_ring;
    const bool unordered = is_dispatch && ring_last_dispatch[ring] && !ring_synced[ring] &&
                           Common::PerfStats::AbToggleOn(gcn_unordered_toggle);
    ring_last_dispatch[ring] = is_dispatch;
    ring_synced[ring] = false;
    return unordered ? static_cast<s8>(ring) : s8{-1};
}

void Rasterizer::OnGameSync() {
    if (!gcn_unordered_dispatches) {
        return;
    }
    ring_synced[command_ring] = true;
    runtime.OnGameSync(command_ring);
}

void Rasterizer::VerifyIncrementalBind(bool matches, const char* kind, u64 pgm_hash, VAddr address) {
    Common::PerfStats::Add(Common::PerfStats::Id::BindVerifyChecks);
    ++bind_verify_checks;
    if (!matches) {
        Common::PerfStats::Add(Common::PerfStats::Id::BindVerifyMismatches);
        if (++bind_verify_mismatches <= 32) {
            LOG_ERROR(Render_Vulkan,
                      "incremental_bind verify: reused {} of shader {:#x} at {:#x} differs from the "
                      "full bind",
                      kind, pgm_hash, address);
        }
    }
    const auto now = std::chrono::steady_clock::now();
    if (now - bind_verify_last_report >= std::chrono::seconds{60}) {
        bind_verify_last_report = now;
        LOG_WARNING(Render_Vulkan,
                    "incremental_bind verify: {} reusable slots checked, {} mismatches, {} T# raced "
                    "with a GPU unmap",
                    bind_verify_checks, bind_verify_mismatches, bind_verify_raced);
    }
}

void Rasterizer::FlushPeriodic() {
    // With readback_ahead most readbacks no longer submit, so the GPU would only receive the frame at the
    // remaining drains and sit idle while the command processor records it. Submitting every N commands
    // lets it run the frame during the recording. A submit inside a render pass cuts it, so it waits for
    // the next command outside one, up to PassOverrun * N commands
    constexpr u32 PassOverrun = 4;
    const u32 every = periodic_flush_commands;
    const u32 recorded = scheduler.CommandsSinceSubmit();
    if (every == 0 || recorded < every) {
        return;
    }
    const bool in_pass = scheduler.IsRendering();
    if (in_pass && recorded < u64{every} * PassOverrun) {
        return;
    }
    Common::PerfStats::Add(Common::PerfStats::Id::FlushPeriodic);
    if (in_pass) {
        Common::PerfStats::Add(Common::PerfStats::Id::FlushPeriodicInPass);
    }
    scheduler.Flush();
}

bool Rasterizer::IsComputeMetaClear(const Pipeline* pipeline) {
    if (!pipeline->IsCompute()) {
        return false;
    }

    // Most of the time when a metadata is updated with a shader it gets cleared. It means
    // we can skip the whole dispatch and update the tracked state instead. Also, it is not
    // intended to be consumed and in such rare cases (e.g. HTile introspection, CRAA) we
    // will need its full emulation anyways.
    const auto& info = pipeline->GetStage(Shader::SwStage::Compute);

    // Assume if a shader reads metadata, it is a copy shader.
    for (const auto& desc : info.buffers) {
        const VAddr address = desc.GetSharp(info).base_address;
        if (!desc.IsSpecial() && !desc.is_written && texture_cache.IsMeta(address)) {
            return false;
        }
    }

    // Metadata surfaces are tiled and thus need address calculation to be written properly.
    // If a shader wants to encode HTILE, for example, from a depth image it will have to compute
    // proper tile address from dispatch invocation id. This address calculation contains an xor
    // operation so use it as a heuristic for metadata writes that are probably not clears.
    if (!info.has_bitwise_xor) {
        // Assume if a shader writes metadata without address calculation, it is a clear shader.
        for (const auto& desc : info.buffers) {
            const VAddr address = desc.GetSharp(info).base_address;
            if (!desc.IsSpecial() && desc.is_written && texture_cache.ClearMeta(address)) {
                // Assume all slices were updates
                LOG_TRACE(Render_Vulkan, "Metadata update skipped");
                return true;
            }
        }
    }
    return false;
}

bool Rasterizer::IsComputeImageCopy(const Pipeline* pipeline) {
    if (!pipeline->IsCompute()) {
        return false;
    }

    // Ensure shader only has 2 bound buffers
    const auto& cs_pgm = liverpool->GetCsRegs();
    const auto& info = pipeline->GetStage(Shader::SwStage::Compute);
    if (cs_pgm.num_thread_x.full != 64 || info.buffers.size() != 2 || !info.images.empty()) {
        return false;
    }

    // Those 2 buffers must both be formatted. One must be source and another destination.
    const auto& desc0 = info.buffers[0];
    const auto& desc1 = info.buffers[1];
    if (!desc0.is_formatted || !desc1.is_formatted || desc0.is_written == desc1.is_written) {
        return false;
    }

    // Buffers must have the same size and each thread of the dispatch must copy 1 dword of data
    const AmdGpu::Buffer buf0 = desc0.GetSharp(info);
    const AmdGpu::Buffer buf1 = desc1.GetSharp(info);
    if (buf0.GetSize() != buf1.GetSize() || cs_pgm.dim_x != (buf0.GetSize() / 256)) {
        return false;
    }

    // Find images the buffer alias
    const auto image0_id = texture_cache.FindImageFromRange(buf0.base_address, buf0.GetSize());
    if (!image0_id) {
        return false;
    }
    const auto image1_id =
        texture_cache.FindImageFromRange(buf1.base_address, buf1.GetSize(), false);
    if (!image1_id) {
        return false;
    }

    // Image copy must be valid
    VideoCore::Image& image0 = texture_cache.GetImage(image0_id);
    VideoCore::Image& image1 = texture_cache.GetImage(image1_id);
    if (image0.info.guest_size != image1.info.guest_size ||
        image0.info.pitch != image1.info.pitch || image0.info.guest_size != buf0.GetSize() ||
        image0.info.num_bits != image1.info.num_bits) {
        return false;
    }

    // Perform image copy
    VideoCore::Image& src_image = desc0.is_written ? image1 : image0;
    VideoCore::Image& dst_image = desc0.is_written ? image0 : image1;
    runtime.CopyColorAndDepth(&src_image, &dst_image);
    return true;
}

bool Rasterizer::IsComputeImageClear(const Pipeline* pipeline) {
    if (!pipeline->IsCompute()) {
        return false;
    }

    // Ensure shader only has 2 bound buffers
    const auto& cs_pgm = liverpool->GetCsRegs();
    const auto& info = pipeline->GetStage(Shader::SwStage::Compute);
    if (cs_pgm.num_thread_x.full != 64 || info.buffers.size() != 2 || !info.images.empty()) {
        return false;
    }

    // From those 2 buffers, first must hold the clear vector and second the image being cleared
    const auto& desc0 = info.buffers[0];
    const auto& desc1 = info.buffers[1];
    if (desc0.is_formatted || !desc1.is_formatted || desc0.is_written || !desc1.is_written) {
        return false;
    }

    // First buffer must have size of vec4 and second the size of a single layer
    const AmdGpu::Buffer buf0 = desc0.GetSharp(info);
    const AmdGpu::Buffer buf1 = desc1.GetSharp(info);
    const u32 buf1_bpp = AmdGpu::NumBitsPerBlock(buf1.GetDataFmt());
    if (buf0.GetSize() != 16 || (cs_pgm.dim_x * 128ULL * (buf1_bpp / 8)) != buf1.GetSize()) {
        return false;
    }

    // Find image the buffer alias
    const auto image1_id =
        texture_cache.FindImageFromRange(buf1.base_address, buf1.GetSize(), false);
    if (!image1_id) {
        return false;
    }

    // Image clear must be valid
    VideoCore::Image& image1 = texture_cache.GetImage(image1_id);
    if (image1.info.guest_size != buf1.GetSize() || image1.info.num_bits != buf1_bpp ||
        image1.info.props.is_depth) {
        return false;
    }

    // Perform image clear
    const float* values = reinterpret_cast<float*>(buf0.base_address);
    const vk::ClearValue clear = {
        .color = {.float32 = std::array<float, 4>{values[0], values[1], values[2], values[3]}},
    };
    const VideoCore::SubresourceRange range = {
        .base =
            {
                .level = 0,
                .layer = 0,
            },
        .extent = image1.info.resources,
    };
    runtime.ClearImage(&image1, range, clear);
    return true;
}

void Rasterizer::BindBuffers(const Shader::Info& stage, Shader::Backend::Bindings& binding,
                             Shader::PushData& push_data) {
    const u64 alignment = instance.StorageMinAlignment();
    for (const auto& desc : stage.buffers) {
        if (desc.IsSpecial()) {
            if (desc.buffer_type == Shader::BufferType::GdsBuffer) {
                const auto* gds_buf = buffer_cache.GetGdsBuffer();
                buffer_infos.emplace_back(gds_buf->Handle(), 0, gds_buf->SizeBytes());
                // GDS fills/copies and earlier shader writes need a barrier before this access
                needs_barrier |=
                    runtime.IsBufferAccessed(gds_buf, 0, gds_buf->SizeBytes(), desc.is_written);
                bound_buffers.emplace_back(gds_buf, 0, gds_buf->SizeBytes(), desc.is_written);
            } else if (desc.buffer_type == Shader::BufferType::Flatbuf) {
                auto& vk_buffer = buffer_cache.GetStreamBuffer();
                const u32 ubo_size = stage.flattened_ud_buf.size() * sizeof(u32);
                const u64 offset =
                    vk_buffer.Copy(stage.flattened_ud_buf.data(), ubo_size, alignment);
                buffer_infos.emplace_back(vk_buffer.Handle(), offset, ubo_size);
            } else if (desc.buffer_type == Shader::BufferType::ClipPlanes) {
                // Permutations compiled without enabled planes never read the buffer, so the
                // declared binding is satisfied with a null descriptor instead of a copy.
                if (liverpool->regs.clipper_control.user_clip_plane_enable == 0) {
                    buffer_infos.emplace_back(VK_NULL_HANDLE, 0, VK_WHOLE_SIZE);
                } else {
                    auto& vk_buffer = buffer_cache.GetStreamBuffer();
                    std::array<float, AmdGpu::NUM_CLIP_PLANES * 4> planes{};
                    for (u32 i = 0; i < AmdGpu::NUM_CLIP_PLANES; ++i) {
                        const auto& plane = liverpool->regs.clip_user_data[i];
                        planes[i * 4 + 0] = std::bit_cast<float>(plane.data_x);
                        planes[i * 4 + 1] = std::bit_cast<float>(plane.data_y);
                        planes[i * 4 + 2] = std::bit_cast<float>(plane.data_z);
                        planes[i * 4 + 3] = std::bit_cast<float>(plane.data_w);
                    }
                    const u32 ubo_size = static_cast<u32>(sizeof(planes));
                    const u64 offset = vk_buffer.Copy(planes.data(), ubo_size, alignment);
                    buffer_infos.emplace_back(vk_buffer.Handle(), offset, ubo_size);
                }
            } else if (desc.buffer_type == Shader::BufferType::BdaPagetable) {
                const auto* bda_buffer = buffer_cache.GetBdaPageTableBuffer();
                buffer_infos.emplace_back(bda_buffer->Handle(), 0, bda_buffer->SizeBytes());
            } else if (desc.buffer_type == Shader::BufferType::FaultBuffer) {
                const auto* fault_buffer = buffer_cache.GetFaultBuffer();
                buffer_infos.emplace_back(fault_buffer->Handle(), 0, fault_buffer->SizeBytes());
            } else if (desc.buffer_type == Shader::BufferType::LoopCapBuffer) {
                const auto* loop_cap_buffer = buffer_cache.GetLoopCapBuffer();
                buffer_infos.emplace_back(loop_cap_buffer->Handle(), 0,
                                          loop_cap_buffer->SizeBytes());
            } else if (desc.buffer_type == Shader::BufferType::SharedMemory) {
                auto& lds_buffer = buffer_cache.GetStreamBuffer();
                const auto& cs_program = liverpool->GetCsRegs();
                const auto lds_size = cs_program.SharedMemSize() * cs_program.NumWorkgroups();
                const auto [data, offset] = lds_buffer.Map(lds_size, alignment);
                std::memset(data, 0, lds_size);
                lds_buffer.Commit();
                buffer_infos.emplace_back(lds_buffer.Handle(), offset, lds_size);
            } else {
                UNREACHABLE_MSG("Unexpected buffer type {}", u32(desc.buffer_type));
            }
        } else {
            const auto vsharp = desc.GetSharp(stage);
            const auto bind_range = [&](const VideoCore::Buffer* buffer, u64 offset, u64 size) {
                const u64 offset_aligned = Common::AlignDown(offset, alignment);
                const u64 adjust = offset - offset_aligned;
                if (adjust % 4 != 0) {
                    LOG_WARNING(Render_Vulkan, "Buffer binding in shader {:#x} isn't dword aligned",
                                stage.pgm_hash);
                }
                push_data.AddOffset(binding.buffer, adjust);
                buffer_infos.emplace_back(buffer->Handle(), offset_aligned, size + adjust);
                bound_buffers.emplace_back(buffer, offset, size, desc.is_written);
                if (desc.is_written) {
                    // Raw storage-buffer writes can also make an aliased cached image stale.
                    texture_cache.InvalidateMemoryFromGPU(vsharp.base_address, size);
                    Common::GuestWriteJournal::Record(
                        Common::GuestWriteJournal::Source::WrittenBinding, vsharp.base_address,
                        size, nullptr, stage.pgm_hash);
                    WatchGpuWrite("written binding", stage.pgm_hash, vsharp.base_address, size);
                }
                if (GpuCheckpoints::DiagEnabled() &&
                    diag_detail.num_buffers < GpuCheckpoints::Detail::MaxBuffers) {
                    diag_sources[diag_detail.num_buffers] = {buffer->Handle(), offset, size};
                    diag_detail.AddBuffer(vsharp.base_address, size, desc.is_written);
                    auto& ref = diag_detail.buffers[diag_detail.num_buffers - 1];
                    ref.overlaps_stack = memory->OverlapsStackRange(vsharp.base_address, size);
                    ref.has_cpu_words =
                        size >= sizeof(ref.cpu_words) &&
                        memory->TryReadBacking(vsharp.base_address, ref.cpu_words.data(),
                                               sizeof(ref.cpu_words));
                }
                needs_barrier |= runtime.IsBufferAccessed(buffer, offset, size, desc.is_written);
            };

            // Reuse the previous read-only V# binding from this pipeline when its descriptor,
            // arenas, sync batch and GPU mappings are unchanged
            // Those checks keep the same arena range valid for the next draw without repeating the
            // buffer lookups
            BindCache::BufferSlot* slot =
                bind_cache ? &bind_cache->Buffer(bind_cache_buffer_index++) : nullptr;
            const bool reuse = slot && !desc.is_written && slot->Matches(vsharp, bind_generations);
            if (reuse && !incremental_bind_verify) {
                buffer_cache.ReuseReadBuffer(slot->arena, vsharp.base_address, slot->size,
                                             desc.is_formatted);
                Common::PerfStats::Add(Common::PerfStats::Id::BindReusedBuffers);
                bind_range(slot->arena, slot->offset, slot->size);
            } else {
                // With verification enabled, run the full binding path for this slot and compare
                // its result with the binding we would have reused
                const std::optional<BindCache::BufferSlot> reused =
                    reuse ? std::optional{*slot} : std::nullopt;
                const bool reused_valid =
                    reuse && buffer_cache.CheckReusedBuffer(slot->arena, slot->offset,
                                                            vsharp.base_address, slot->size);
                if (slot) {
                    slot->arena = nullptr;
                }
                bool reuse_matches = false;
                // A V# with an unmapped base address cannot describe a usable resource
                // This can happen with stale descriptors in draws the game would predicate away, so
                // bind null instead of asserting or using unmapped memory
                // Do not pass that address to ClampRangeSize or create a buffer over a range the
                // guest has not mapped
                if (vsharp.base_address <= 1 || vsharp.GetSize() == 0 ||
                    !IsMapped(vsharp.base_address, 1)) {
                    buffer_infos.emplace_back(VK_NULL_HANDLE, 0, VK_WHOLE_SIZE);
                } else {
                    const u64 size = memory->ClampRangeSize(vsharp.base_address, vsharp.GetSize());
                    if (size != vsharp.GetSize()) {
                        LOG_DEBUG(Render, "Clamped size from {} to {} for stage {:#x}",
                                  vsharp.GetSize(), size, stage.pgm_hash);
                    }
                    const auto [buffer, offset] = buffer_cache.ObtainBuffer(
                        vsharp.base_address, size, desc.is_written, desc.is_formatted);
                    const bool stream = buffer == &buffer_cache.GetStreamBuffer();
                    if (slot && !desc.is_written && !stream) {
                        *slot = {
                            .sharp = vsharp,
                            .arena = buffer,
                            .offset = offset,
                            .size = static_cast<u32>(size),
                            .sync_flushes = bind_generations.sync_flushes,
                            .maps = bind_generations.maps,
                            .arenas = bind_generations.arenas,
                        };
                    }
                    // The full binding path may copy a small range into the stream buffer even
                    // though the same range is still valid in its arena
                    reuse_matches = reused && reused_valid && size == reused->size &&
                                    (stream ||
                                     (buffer == reused->arena && offset == reused->offset));
                    bind_range(buffer, offset, size);
                }
                if (reused) {
                    VerifyIncrementalBind(reuse_matches, "V#", stage.pgm_hash, vsharp.base_address);
                }
            }
        }

        auto& set_write = set_writes[set_write_index++];
        set_write.dstSet = VK_NULL_HANDLE;
        set_write.dstBinding = binding.unified++;
        set_write.dstArrayElement = 0;
        set_write.descriptorCount = 1;
        set_write.descriptorType = vk::DescriptorType::eStorageBuffer;
        set_write.pBufferInfo = &buffer_infos.back();
        ++binding.buffer;
    }
}

bool Rasterizer::CheckBindableImage(const AmdGpu::Image& tsharp,
                                    const Shader::ImageResource& image_desc,
                                    std::optional<VideoCore::TextureCache::ImageDesc>& bound_desc,
                                    bool log) {
    const auto data_fmt = tsharp.GetDataFmt();
    const auto num_fmt = tsharp.GetNumberFmt();
    if (tsharp.Address() == 0 || data_fmt == AmdGpu::DataFormat::FormatInvalid) {
        return false;
    }

    if (!memory->IsValidGpuMapping(tsharp.Address(), 0) ||
        !magic_enum::enum_contains(data_fmt) || !magic_enum::enum_contains(num_fmt)) {
        if (log) {
            LOG_WARNING(Render_Vulkan,
                        "Rejecting invalid T# address={:#x}, pitch={}, width={}, "
                        "data_format={}, num_format={}",
                        tsharp.Address(), tsharp.pitch, tsharp.width, static_cast<u32>(data_fmt),
                        static_cast<u32>(num_fmt));
        }
        return false;
    }

    // Reject format pairs with no Vulkan format before ImageInfo can assert
    // Bind them as null so reads return zero
    const auto vk_fmt = LiverpoolToVK::TrySurfaceFormat(data_fmt, num_fmt);
    const bool depth_ok = !image_desc.is_depth || vk_fmt == vk::Format::eR32Sfloat ||
                          vk_fmt == vk::Format::eR32Uint || vk_fmt == vk::Format::eR16Unorm;
    if (vk_fmt == vk::Format::eUndefined || !depth_ok) {
        if (log) {
            LOG_WARNING(Render_Vulkan,
                        "Rejecting T# with no host format address={:#x}, pitch={}, width={}, "
                        "data_format={}, num_format={}, depth={}",
                        tsharp.Address(), tsharp.pitch, tsharp.width, static_cast<u32>(data_fmt),
                        static_cast<u32>(num_fmt), image_desc.is_depth);
        }
        return false;
    }

    // Limit warnings because an invalid descriptor can be rebound every draw
    static std::atomic<u32> layout_rejections{0};
    static std::atomic<u32> low_address_rejections{0};
    constexpr u32 MaxRejectionWarnings = 32;
    if (const char* reason = UnsupportedImageLayout(tsharp)) {
        if (log && layout_rejections.fetch_add(1, std::memory_order_relaxed) < MaxRejectionWarnings) {
            LOG_WARNING(Render_Vulkan,
                        "Rejecting T# with an unsupported layout ({}) address={:#x}, width={}, "
                        "tiling_index={}, data_format={}, samples={}",
                        reason, tsharp.Address(), tsharp.width, u32(tsharp.tiling_index),
                        static_cast<u32>(data_fmt), tsharp.NumSamples());
        }
        return false;
    }

    const auto& desc = bound_desc.emplace(tsharp, image_desc);
    if (!IsPlausibleImage(desc.info, log)) {
        return false;
    }

    // Reject texture addresses below guest memory before page tracking
    // They can pass the 40-bit check but still fail Protect
    if (desc.info.guest_address < memory->SystemManagedVirtualBase()) {
        if (log &&
            low_address_rejections.fetch_add(1, std::memory_order_relaxed) < MaxRejectionWarnings) {
            LOG_WARNING(Render_Vulkan,
                        "Rejecting T# below the guest address space address={:#x}, size={:#x}, "
                        "width={}, data_format={}",
                        desc.info.guest_address, desc.info.guest_size, tsharp.width,
                        static_cast<u32>(data_fmt));
        }
        return false;
    }
    return true;
}

void Rasterizer::BindTextures(const Shader::Info& stage, Shader::Backend::Bindings& binding) {
    image_bindings.clear();
    const u32 first_image_idx = image_infos.size();
    // To emulate storing to explicit mip levels, build a descriptor
    // array with each mip level
    boost::container::small_vector<u32, 8> image_descriptor_array_sizes;

    using ImageDesc = VideoCore::TextureCache::ImageDesc;
    const auto count_lod_stats = [&](const AmdGpu::Image& tsharp, bool is_written) {
        if (tsharp.lod_hw_cnt_en && lod_stats_enabled) {
            u32& uses = lod_stats_uses[tsharp.counter_bank_id];
            uses += uses < 0xFFFFFFu;
        }
        if (GpuCheckpoints::DiagEnabled()) {
            diag_detail.AddImage(tsharp.Address(),
                                 static_cast<u32>(tsharp.GetDataFmt()) |
                                     static_cast<u32>(tsharp.GetNumberFmt()) << 8,
                                 is_written);
        }
    };
    const auto mark_bound = [&](VideoCore::ImageId& image_id,
                                const Shader::ImageResource& image_desc) {
        auto* image = &texture_cache.GetImage(image_id);
        if (auto depth_image_id = texture_cache.GetAssociatedDepth(*image)) {
            // If this image has an associated depth image, it's a stencil attachment. Redirect the access to
            // the actual depth-stencil buffer
            image_id = depth_image_id;
            image = &texture_cache.GetImage(image_id);
        }
        if (image->binding.is_bound) {
            // The image is already bound. In case if it is about to be used as storage we need to force
            // general layout on it
            image->binding.force_general |= image_desc.is_written;
        }
        image->binding.is_bound = 1u;
    };
    // Keep FindImage's result in the shared resolution before applying the depth redirect used by
    // incremental_bind and tsharp_cache
    const auto find_bound_image = [&](VideoCore::ImageId& image_id,
                                      VideoCore::TextureCache::ImageDesc& desc,
                                      const Shader::ImageResource& image_desc,
                                      BindCache::ImageResolution* resolution = nullptr) {
        image_id = texture_cache.FindImage(desc);
        if (resolution) {
            resolution->bindings.emplace_back(image_id, desc);
        }
        mark_bound(image_id, image_desc);
    };
    // A rejected T# still fills every descriptor the shader declares, with the declared type
    const auto append_null_binding = [&](const Shader::ImageResource& image_desc) {
        image_bindings.emplace_back(std::piecewise_construct, std::tuple{}, std::tuple{});
        image_bindings.back().second.type = image_desc.is_written
                                                ? VideoCore::TextureCache::BindingType::Storage
                                                : VideoCore::TextureCache::BindingType::Texture;
    };

    // The T# reuse counters go to the perf stats once per call
    struct ReuseCounts {
        u64 tsharp_cache{};
        u64 slots{};
        ~ReuseCounts() {
            if (tsharp_cache) {
                Common::PerfStats::Add(Common::PerfStats::Id::TsharpCacheHits, tsharp_cache);
            }
            if (slots) {
                Common::PerfStats::Add(Common::PerfStats::Id::BindReusedImages, slots);
            }
        }
    } reuse_counts;
    for (const auto& image_desc : stage.images) {
        const auto tsharp = image_desc.GetSharp(stage);
        const u32 num_bindings = image_desc.NumBindings(stage);
        if (image_desc.array_size != 0) {
            // Per-game dynamic_tsharp_array_size: one binding per table element, then a null one for
            // out-of-range indices. The shader was typed after `tsharp`, the first valid element, so elements
            // of another view type or number class stay null
            const auto view_type = tsharp.GetViewType(image_desc.is_array);
            const bool is_integer = AmdGpu::IsInteger(tsharp.GetNumberFmt());
            u32 null_elements = 0;
            for (u32 i = 0; i < num_bindings; ++i) {
                const auto element = i < image_desc.array_size
                                         ? image_desc.GetElementSharp(stage, i)
                                         : AmdGpu::Image::Null(image_desc.is_depth);
                const bool compatible =
                    element && element.GetViewType(image_desc.is_array) == view_type &&
                    AmdGpu::IsInteger(element.GetNumberFmt()) == is_integer &&
                    element.GetNumberConversion() == tsharp.GetNumberConversion();
                std::optional<ImageDesc> element_desc;
                if (!compatible || !CheckBindableImage(element, image_desc, element_desc, true)) {
                    null_elements += i < image_desc.array_size;
                    append_null_binding(image_desc);
                    continue;
                }
                count_lod_stats(element, image_desc.is_written);
                auto& [image_id, desc] = image_bindings.emplace_back(
                    std::piecewise_construct, std::tuple{}, std::forward_as_tuple(*element_desc));
                find_bound_image(image_id, desc, image_desc);
            }
            ReportTsharpArray(stage.pgm_hash, image_desc.array_size, null_elements);
            image_descriptor_array_sizes.push_back(num_bindings);
            continue;
        }
        const auto append_null_bindings = [&] {
            for (u32 i = 0; i < num_bindings; ++i) {
                append_null_binding(image_desc);
            }
            image_descriptor_array_sizes.push_back(num_bindings);
        };

        // incremental_bind keeps this slot's previous T# result, while tsharp_cache can reuse the
        // last result from any pipeline
        // Use either result only while its mapping and image checks still hold, including a
        // previously rejected descriptor
        // Read the image generation for each T# because a lookup earlier in this call may have
        // registered or removed an image
        // The rejection checks only read the T# and image resource, so a reusable rejected result
        // should keep binding the same null descriptors
        const u64 image_sets = texture_cache.ImageSetGeneration();
        const u64 maps = buffer_cache.MappingGeneration();
        bind_generations.image_sets = image_sets;
        BindCache::ImageSlot* slot =
            bind_cache ? &bind_cache->Image(bind_cache_image_index++) : nullptr;
        const auto holds = [&](const BindCache::ImageResolution& resolution) {
            return (resolution.rejected || resolution.bindings.size() == num_bindings) &&
                   (tsharp_cache_on ? resolution.PagesUnchanged(texture_cache)
                                    : resolution.SameGenerations(bind_generations));
        };
        std::optional<TsharpCache::Key> tsharp_key;
        const BindCache::ImageResolution* cached = nullptr;
        bool from_tsharp_cache = false;
        if (slot && slot->SameSharp(tsharp) && holds(slot->resolution)) {
            cached = &slot->resolution;
            if (tsharp_cache_on && !slot->resolution.SameGenerations(bind_generations)) {
                Common::PerfStats::Add(Common::PerfStats::Id::TsharpCacheSlotPages);
            }
        } else if (tsharp_cache_on) {
            tsharp_key = TsharpCache::MakeKey(tsharp, image_desc, num_bindings);
            const auto* entry = tsharp_cache->Find(*tsharp_key);
            if (entry && holds(*entry)) {
                cached = entry;
                from_tsharp_cache = true;
            } else {
                Common::PerfStats::Add(entry ? Common::PerfStats::Id::TsharpCacheStale
                                             : Common::PerfStats::Id::TsharpCacheNew);
            }
        }
        if (cached && !incremental_bind_verify && !tsharp_cache_verify) {
            ++(from_tsharp_cache ? reuse_counts.tsharp_cache : reuse_counts.slots);
            if (from_tsharp_cache && slot) {
                slot->sharp = tsharp;
                slot->valid = true;
                slot->resolution = *cached;
                cached = &slot->resolution;
            }
            if (cached->rejected) {
                append_null_bindings();
                continue;
            }
            count_lod_stats(tsharp, image_desc.is_written);
            for (const auto& [cached_id, cached_desc] : cached->bindings) {
                auto& image_id = image_bindings.emplace_back(cached_id, cached_desc).first;
                // Keep the LRU touch that FindImage normally performs even when we reuse its image
                // result and skip the lookup
                texture_cache.TouchReusedImage(image_id);
                mark_bound(image_id, image_desc);
            }
            image_descriptor_array_sizes.push_back(num_bindings);
            continue;
        }
        // With verification enabled, resolve the reused T# through the full lookup again and
        // compare the resulting images and views
        std::optional<BindCache::ImageResolution> reused;
        if (cached) {
            reused.emplace(*cached);
        }
        BindCache::ImageResolution fresh{.image_sets = image_sets, .maps = maps};
        const auto finish_slot = [&](bool rejected) {
            fresh.rejected = rejected;
            if (reused) {
                bool matches = reused->rejected == rejected &&
                               reused->bindings.size() == fresh.bindings.size();
                for (size_t i = 0; matches && i < fresh.bindings.size(); ++i) {
                    const auto& [old_id, old_desc] = reused->bindings[i];
                    const auto& [new_id, new_desc] = fresh.bindings[i];
                    matches = old_id == new_id && old_desc.type == new_desc.type &&
                              old_desc.view_info == new_desc.view_info &&
                              old_desc.info.guest_address == new_desc.info.guest_address &&
                              old_desc.info.guest_size == new_desc.info.guest_size &&
                              old_desc.info.pixel_format == new_desc.info.pixel_format;
                }
                // The guest may unmap GPU memory between the reused result and the verification
                // lookup
                // That changes the lookup result even without caching, just as it would if a normal
                // lookup happened immediately before the unmap
                const bool raced = !matches && buffer_cache.MappingGeneration() != maps;
                bind_verify_raced += raced;
                VerifyIncrementalBind(matches || raced,
                                      from_tsharp_cache ? "T# (tsharp_cache)" : "T#",
                                      stage.pgm_hash, tsharp.Address());
            }
            if (slot) {
                slot->sharp = tsharp;
                slot->valid = true;
                slot->resolution = fresh;
            }
            if (tsharp_cache_on) {
                if (!tsharp_key) {
                    tsharp_key = TsharpCache::MakeKey(tsharp, image_desc, num_bindings);
                }
                tsharp_cache->Store(*tsharp_key, fresh);
            }
        };

        if (texture_cache.IsMeta(tsharp.Address())) {
            LOG_WARNING(Render_Vulkan, "Unexpected metadata read by a shader (texture)");
        }

        std::optional<ImageDesc> bound_desc;
        if (!CheckBindableImage(tsharp, image_desc, bound_desc, true)) {
            append_null_bindings();
            finish_slot(true);
            continue;
        }
        fresh.address = bound_desc->info.guest_address;
        fresh.size = bound_desc->info.guest_size;
        count_lod_stats(tsharp, image_desc.is_written);

        const Shader::MipStorageFallbackMode mip_fallback_mode = image_desc.mip_fallback_mode;

        for (auto i = 0; i < num_bindings; i++) {
            auto& [image_id, desc] = image_bindings.emplace_back(
                std::piecewise_construct, std::tuple{}, std::forward_as_tuple(*bound_desc));

            if (mip_fallback_mode == Shader::MipStorageFallbackMode::ConstantIndex) {
                ASSERT(num_bindings == 1);
                desc.view_info.range.base.level += image_desc.constant_mip_index;
                desc.view_info.range.extent.levels = 1;
            } else if (mip_fallback_mode == Shader::MipStorageFallbackMode::DynamicIndex) {
                desc.view_info.range.base.level += i;
                desc.view_info.range.extent.levels = 1;
            }

            find_bound_image(image_id, desc, image_desc, &fresh);
        }

        image_descriptor_array_sizes.push_back(num_bindings);
        finish_slot(false);
    }

    // Second pass to re-bind images that were updated after binding
    for (auto& [image_id, desc] : image_bindings) {
        bool is_storage = desc.type == VideoCore::TextureCache::BindingType::Storage;
        if (!image_id) {
            image_infos.emplace_back(VK_NULL_HANDLE, VK_NULL_HANDLE, vk::ImageLayout::eGeneral);
        } else {
            if (auto& old_image = texture_cache.GetImage(image_id);
                old_image.binding.needs_rebind) {
                old_image.binding = {};
                image_id = texture_cache.FindImage(desc);
            }

            bound_images.emplace_back(image_id);

            auto& image = texture_cache.GetImage(image_id);
            auto& image_view = texture_cache.FindTexture(image_id, desc);
            const auto binding = image.binding;

            // The image is either bound as storage in a separate descriptor or bound as render
            // target in feedback loop. Depth images are excluded because they can't be bound as
            // storage and feedback loop doesn't make sense for them
            if ((binding.force_general || binding.is_target) && !image.info.props.is_depth) {
                if (instance.IsAttachmentFeedbackLoopLayoutSupported() && image.binding.is_target) {
                    needs_barrier |= runtime.Transit(
                        &image, vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT,
                        vk::PipelineStageFlagBits2::eAllGraphics, vk::AccessFlagBits2::eShaderRead);
                } else {
                    needs_barrier |= runtime.Transit(
                        &image, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits2::eAllCommands,
                        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
                        desc.view_info.range);
                }
            } else {
                if (is_storage) {
                    needs_barrier |= runtime.Transit(
                        &image, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits2::eAllCommands,
                        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
                        desc.view_info.range);
                } else {
                    auto new_layout = image.info.props.is_depth
                                          ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
                                          : vk::ImageLayout::eShaderReadOnlyOptimal;
                    auto new_access = vk::AccessFlags2{vk::AccessFlagBits2::eShaderRead};
                    // Use the attachment layout and access for both uses of
                    // this read-only depth target
                    // This avoids a barrier between sampling and attachment access
                    if (image.info.props.is_depth && binding.is_target &&
                        image_id == db_desc.first &&
                        LiverpoolToVK::IsFormatDepthCompatible(image_view.info.format)) {
                        if (const auto layout = DepthTargetLayout(image);
                            layout && IsReadOnlyDepthLayout(*layout)) {
                            Common::PerfStats::Add(Common::PerfStats::Id::DepthTargetSampled);
                            if (depth_target_shared) {
                                new_layout = *layout;
                                new_access |= SharedDepthTargetAccess;
                            }
                        }
                    }
                    needs_barrier |= runtime.Transit(&image, new_layout,
                                                     vk::PipelineStageFlagBits2::eAllCommands,
                                                     new_access, desc.view_info.range);
                }
            }
            image.usage.storage |= is_storage;
            image.usage.texture |= !is_storage;

            image_infos.emplace_back(VK_NULL_HANDLE, *image_view.image_view,
                                     image.backing->state.layout);
            if (!is_storage) {
                const auto& range = image_view.info.range;
                image_descriptor_refs.push_back({
                    .info_index = static_cast<u32>(image_infos.size() - 1),
                    .backing = image.backing,
                    .subres_idx = range.base.level * image.info.resources.layers + range.base.layer,
                });
            }
        }
    }

    u32 image_info_idx = first_image_idx;
    u32 image_binding_idx = 0;
    for (u32 array_size : image_descriptor_array_sizes) {
        const auto& [_, desc] = image_bindings[image_binding_idx];
        const bool is_storage = desc.type == VideoCore::TextureCache::BindingType::Storage;
        auto& set_write = set_writes[set_write_index++];
        set_write.dstSet = VK_NULL_HANDLE;
        set_write.dstBinding = binding.unified;
        set_write.dstArrayElement = 0;
        set_write.descriptorCount = array_size;
        set_write.descriptorType =
            is_storage ? vk::DescriptorType::eStorageImage : vk::DescriptorType::eSampledImage;
        set_write.pImageInfo = &image_infos[image_info_idx];

        image_info_idx += array_size;
        image_binding_idx += array_size;
        binding.unified += array_size;
    }

    for (const auto& sampler : stage.samplers) {
        auto ssharp = sampler.GetSharp(stage);
        // An S# whose fields have no defined meaning on GCN (filter mode 3, mip filter 3), or a
        // custom border color without a border color table, comes from an unused sampler slot
        // holding garbage, so bind the default sampler. Anisotropy ratios 5-7 stay accepted
        if (!ssharp.Valid() || (ssharp.border_color_type.Value() == AmdGpu::BorderColor::Custom &&
                                liverpool->regs.ta_bc_base.Address() == 0)) {
            LOG_RENDER_PROBLEM(Render_Vulkan, Warning,
                               "Rejecting invalid S# max_aniso={}, filter_mode={}, mip_filter={}, "
                               "border_color_type={}, border_color_base={:#x}",
                               static_cast<u32>(ssharp.max_aniso.Value()),
                               static_cast<u32>(ssharp.filter_mode.Value()),
                               static_cast<u32>(ssharp.mip_filter.Value()),
                               static_cast<u32>(ssharp.border_color_type.Value()),
                               liverpool->regs.ta_bc_base.Address());
            ssharp = AmdGpu::Sampler{};
        }
        const auto vk_sampler =
            texture_cache.GetSampler(ssharp, liverpool->regs.ta_bc_base, sampler.is_depth);
        image_infos.emplace_back(vk_sampler, VK_NULL_HANDLE, vk::ImageLayout::eGeneral);
        auto& set_write = set_writes[set_write_index++];
        set_write.dstSet = VK_NULL_HANDLE;
        set_write.dstBinding = binding.unified++;
        set_write.dstArrayElement = 0;
        set_write.descriptorCount = 1;
        set_write.descriptorType = vk::DescriptorType::eSampler;
        set_write.pImageInfo = &image_infos.back();
    }
}

static bool IsSampledImageLayout(vk::ImageLayout layout) {
    switch (layout) {
    case vk::ImageLayout::eGeneral:
    case vk::ImageLayout::eShaderReadOnlyOptimal:
    case vk::ImageLayout::eDepthStencilReadOnlyOptimal:
    case vk::ImageLayout::eDepthReadOnlyStencilAttachmentOptimal:
    case vk::ImageLayout::eDepthAttachmentStencilReadOnlyOptimal:
    case vk::ImageLayout::eDepthReadOnlyOptimal:
    case vk::ImageLayout::eStencilReadOnlyOptimal:
    case vk::ImageLayout::eReadOnlyOptimal:
    case vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT:
        return true;
    default:
        return false;
    }
}

void Rasterizer::RefreshImageDescriptorLayouts() {
    // Refresh sampled descriptors after all transitions in the draw
    for (const auto& ref : image_descriptor_refs) {
        const auto layout = ref.backing->Layout(ref.subres_idx);
        if (IsSampledImageLayout(layout)) {
            image_infos[ref.info_index].imageLayout = layout;
        }
    }
}

std::optional<vk::ImageLayout> Rasterizer::DepthTargetLayout(const VideoCore::Image& image) const {
    if (!db_desc.first) {
        return std::nullopt;
    }
    const auto& desc = db_desc.second;
    const bool has_stencil = image.info.props.has_stencil;
    // Stencil writes can be enabled while depth writes are off
    const bool stencil_write =
        has_stencil && liverpool->regs.depth_control.stencil_enable && !desc.view_info.is_storage;
    return desc.view_info.is_storage
               ? has_stencil ? vk::ImageLayout::eDepthStencilAttachmentOptimal
                             : vk::ImageLayout::eDepthAttachmentOptimal
           : stencil_write ? vk::ImageLayout::eDepthReadOnlyStencilAttachmentOptimal
           : has_stencil   ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
                           : vk::ImageLayout::eDepthReadOnlyOptimal;
}

bool Rasterizer::IsReadOnlyDepthLayout(vk::ImageLayout layout) {
    return layout == vk::ImageLayout::eDepthReadOnlyStencilAttachmentOptimal ||
           layout == vk::ImageLayout::eDepthStencilReadOnlyOptimal ||
           layout == vk::ImageLayout::eDepthReadOnlyOptimal;
}

RenderState Rasterizer::BeginRendering(const GraphicsPipeline* pipeline) {
    attachment_feedback_loop = false;
    const auto& regs = liverpool->regs;
    const auto& key = pipeline->GetGraphicsKey();
    RenderState state;
    state.width = instance.GetMaxFramebufferWidth();
    state.height = instance.GetMaxFramebufferHeight();
    state.num_layers = std::numeric_limits<u16>::max();
    state.num_color_attachments = std::bit_width(key.mrt_mask);
    for (auto cb = 0u; cb < state.num_color_attachments; ++cb) {
        auto& [image_id, desc] = cb_descs[cb];
        if (!image_id) {
            state.color_attachments[cb] = {};
            continue;
        }
        auto* image = &texture_cache.GetImage(image_id);
        if (image->binding.needs_rebind) {
            image_id = bound_images.emplace_back(texture_cache.FindImage(desc));
            image = &texture_cache.GetImage(image_id);
        }
        texture_cache.UpdateTarget(image_id);
        runtime.SetBackingSamples(image, key.color_samples[cb]);
        const auto& image_view = texture_cache.FindRenderTarget(image_id, desc);
        const auto slice = image_view.info.range.base.layer;
        const auto mip = image_view.info.range.base.level;

        const auto& col_buf = regs.color_buffers[cb];
        const bool is_clear = texture_cache.IsMetaCleared(col_buf.CmaskAddress(), slice);
        texture_cache.TouchMeta(col_buf.CmaskAddress(), slice, false);

        if (image->binding.is_bound) {
            ASSERT_MSG(!image->binding.force_general,
                       "Having image both as storage and render target is unsupported");
            runtime.FlushBarriers();
            needs_barrier |=
                runtime.Transit(image,
                                instance.IsAttachmentFeedbackLoopLayoutSupported()
                                    ? vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT
                                    : vk::ImageLayout::eGeneral,
                                vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                                vk::AccessFlagBits2::eColorAttachmentWrite |
                                    vk::AccessFlagBits2::eColorAttachmentRead);
            attachment_feedback_loop = true;
        } else {
            needs_barrier |= runtime.Transit(image, vk::ImageLayout::eColorAttachmentOptimal,
                                             vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                                             vk::AccessFlagBits2::eColorAttachmentWrite |
                                                 vk::AccessFlagBits2::eColorAttachmentRead,
                                             desc.view_info.range);
        }

        state.width = std::min<u32>(state.width, std::max(image->info.size.width >> mip, 1u));
        state.height = std::min<u32>(state.height, std::max(image->info.size.height >> mip, 1u));
        state.num_layers = std::min<u32>(state.num_layers, image_view.info.range.extent.layers);

        const auto clear_value =
            is_clear ? LiverpoolToVK::ColorBufferClearValue(col_buf) : vk::ClearValue{};
        auto& attachment = state.color_attachments[cb];
        attachment.image_view = *image_view.image_view;
        attachment.image_layout = image->backing->state.layout;
        attachment.clear_value = clear_value.color.uint32;
        attachment.is_clear = is_clear;

        image->usage.render_target = 1u;
    }
    for (u32 cb = state.num_color_attachments; cb < state.color_attachments.size(); ++cb) {
        state.color_attachments[cb] = {};
    }

    if (auto image_id = db_desc.first; image_id) {
        auto& desc = db_desc.second;
        const auto htile_address = regs.depth_htile_data_base.GetAddress();
        const auto& image_view = texture_cache.FindDepthTarget(image_id, desc);
        auto& image = texture_cache.GetImage(image_id);

        const auto slice = image_view.info.range.base.layer;
        const bool is_depth_clear =
            (regs.depth_render_control.depth_clear_enable && regs.depth_control.depth_enable &&
             regs.depth_control.depth_write_enable) ||
            texture_cache.IsMetaCleared(htile_address, slice);
        const bool is_stencil_clear = regs.depth_render_control.stencil_clear_enable;
        texture_cache.TouchMeta(htile_address, slice, false);
        ASSERT(desc.view_info.range.extent.levels == 1 && !image.binding.needs_rebind);

        const auto new_layout = *DepthTargetLayout(image);
        // depth_target_sampled_layout: a read-only depth target keeps the access of a
        // sampled one, so draws that sample it and draws that do not share one state
        const bool shared = depth_target_shared && IsReadOnlyDepthLayout(new_layout);
        needs_barrier |= runtime.Transit(
            &image, new_layout,
            shared ? vk::PipelineStageFlags2{vk::PipelineStageFlagBits2::eAllCommands}
                   : vk::PipelineStageFlagBits2::eEarlyFragmentTests |
                         vk::PipelineStageFlagBits2::eLateFragmentTests,
            shared ? vk::AccessFlagBits2::eShaderRead | SharedDepthTargetAccess
                   : SharedDepthTargetAccess,
            desc.view_info.range);

        state.width = std::min<u32>(state.width, image.info.size.width);
        state.height = std::min<u32>(state.height, image.info.size.height);
        state.num_layers = std::min<u32>(state.num_layers, image_view.info.range.extent.layers);

        auto& attachment = state.depth_stencil_attachment;
        attachment.image_view = *image_view.image_view;
        attachment.image_layout = image.backing->state.layout;
        attachment.clear_value = {};
        attachment.is_clear = 0;

        if (regs.depth_buffer.DepthValid()) {
            attachment.clear_value[0] = is_depth_clear ? std::bit_cast<u32>(regs.depth_clear) : 0u;
            attachment.has_depth = true;
            attachment.depth_clear = is_depth_clear;
        }
        if (regs.depth_buffer.StencilValid()) {
            attachment.clear_value[1] = is_stencil_clear ? regs.stencil_clear : 0u;
            attachment.has_stencil = true;
            attachment.stencil_clear = is_stencil_clear;
        }

        image.usage.depth_target = true;
    } else {
        state.depth_stencil_attachment = {};
    }

    if (state.num_layers == std::numeric_limits<u16>::max()) {
        state.num_layers = 1;
    }

    return state;
}

void Rasterizer::Resolve() {
    const auto& mrt0_hint = liverpool->last_cb_extent[0];
    const auto& mrt1_hint = liverpool->last_cb_extent[1];
    VideoCore::TextureCache::ImageDesc mrt0_desc{liverpool->regs.color_buffers[0], mrt0_hint};
    VideoCore::TextureCache::ImageDesc mrt1_desc{liverpool->regs.color_buffers[1], mrt1_hint};
    auto& mrt0_image = texture_cache.GetImage(texture_cache.FindImage(mrt0_desc, true));
    auto& mrt1_image = texture_cache.GetImage(texture_cache.FindImage(mrt1_desc, true));

    ScopeMarkerBegin(fmt::format("Resolve:MRT0={:#x}:MRT1={:#x}",
                                 liverpool->regs.color_buffers[0].Address(),
                                 liverpool->regs.color_buffers[1].Address()));
    runtime.ResolveImage(&mrt0_image, &mrt1_image, mrt0_desc.view_info.range,
                         mrt1_desc.view_info.range);
    ScopeMarkerEnd();
}

void Rasterizer::DepthStencilCopy(bool is_depth, bool is_stencil) {
    auto& regs = liverpool->regs;

    auto read_desc = VideoCore::TextureCache::ImageDesc(
        regs.depth_buffer, regs.depth_view, regs.depth_control,
        regs.depth_htile_data_base.GetAddress(), liverpool->last_db_extent, false);
    auto write_desc = VideoCore::TextureCache::ImageDesc(
        regs.depth_buffer, regs.depth_view, regs.depth_control,
        regs.depth_htile_data_base.GetAddress(), liverpool->last_db_extent, true);

    auto& read_image = texture_cache.GetImage(texture_cache.FindImage(read_desc));
    auto& write_image = texture_cache.GetImage(texture_cache.FindImage(write_desc));

    VideoCore::SubresourceRange sub_range;
    sub_range.base.layer = liverpool->regs.depth_view.slice_start;
    sub_range.extent.layers = liverpool->regs.depth_view.NumSlices() - sub_range.base.layer;

    ScopeMarkerBegin(fmt::format(
        "DepthStencilCopy:DR={:#x}:SR={:#x}:DW={:#x}:SW={:#x}", regs.depth_buffer.DepthAddress(),
        regs.depth_buffer.StencilAddress(), regs.depth_buffer.DepthWriteAddress(),
        regs.depth_buffer.StencilWriteAddress()));

    runtime.CopyDepthStencil(&read_image, &write_image, sub_range);

    ScopeMarkerEnd();
}

void Rasterizer::FillBuffer(VAddr address, u32 num_bytes, u32 value, bool is_gds) {
    ASSERT_MSG(address % 4 == 0 && num_bytes % 4 == 0,
               "FillBuffer address and size must be a multiple of 4 bytes");
    if (!is_gds) {
        texture_cache.ClearMeta(address);
        // The journal copies up to 16 bytes of data
        const u32 pattern[4] = {value, value, value, value};
        if (!buffer_cache.IsRegionGpuModified(address, num_bytes)) {
            Common::GuestWriteJournal::Record(Common::GuestWriteJournal::Source::FillCpu, address,
                                              num_bytes, pattern, value);
            u32* buffer = std::bit_cast<u32*>(address);
            std::fill(buffer, buffer + (num_bytes / sizeof(u32)), value);
            return;
        }
        Common::GuestWriteJournal::Record(Common::GuestWriteJournal::Source::FillGpu, address,
                                          num_bytes, pattern, value);
        WatchGpuWrite("FillBuffer", 0, address, num_bytes);
    }
    const auto [buffer, offset] = [&] -> std::pair<const VideoCore::Buffer*, u64> {
        if (is_gds) {
            return {buffer_cache.GetGdsBuffer(), address};
        }
        return buffer_cache.ObtainBuffer(address, num_bytes, true);
    }();
    runtime.FillBuffer(buffer, offset, num_bytes, value);
    buffer_cache.CloseGpuWrites();
}

void Rasterizer::CopyBuffer(VAddr dst, VAddr src, u32 num_bytes, bool dst_gds, bool src_gds) {
    if (!dst_gds && !buffer_cache.IsRegionGpuModified(dst, num_bytes) &&
        !buffer_cache.IsRegionInSyncBatch(dst, num_bytes)) {
        if (!src_gds && !buffer_cache.IsRegionGpuModified(src, num_bytes) &&
            !texture_cache.FindImageFromRange(src, num_bytes)) {
            // Both buffers were not transferred to GPU yet. Can safely copy in host memory.
            Common::GuestWriteJournal::Record(Common::GuestWriteJournal::Source::CopyCpu, dst,
                                              num_bytes, std::bit_cast<const void*>(src), src);
            std::memcpy(std::bit_cast<void*>(dst), std::bit_cast<void*>(src), num_bytes);
            return;
        }
    }
    if (!dst_gds) {
        Common::GuestWriteJournal::Record(Common::GuestWriteJournal::Source::CopyGpu, dst,
                                          num_bytes, nullptr, src_gds ? 0 : src);
        WatchGpuWrite("CopyBuffer", 0, dst, num_bytes);
    }
    texture_cache.InvalidateMemoryFromGPU(dst, num_bytes);
    const auto* gds_buffer = buffer_cache.GetGdsBuffer();
    const auto [src_buffer, src_offset] = [&] -> std::pair<const VideoCore::Buffer*, u64> {
        if (src_gds) {
            return {gds_buffer, src};
        }
        return buffer_cache.ObtainBuffer(src, num_bytes, false, true);
    }();
    const auto [dst_buffer, dst_offset] = [&] -> std::pair<const VideoCore::Buffer*, u64> {
        if (dst_gds) {
            return {gds_buffer, dst};
        }
        return buffer_cache.ObtainBuffer(dst, num_bytes, true, true);
    }();
    const vk::BufferCopy copy = {
        .srcOffset = src_offset,
        .dstOffset = dst_offset,
        .size = num_bytes,
    };
    runtime.CopyBuffer(src_buffer, dst_buffer, std::span{&copy, 1});
    buffer_cache.CloseGpuWrites();
}

u32 Rasterizer::ReadDataFromGds(u32 gds_offset) {
    auto* gds_buf = buffer_cache.GetGdsBuffer();
    u32 value;
    std::memcpy(&value, gds_buf->mapped_data.data() + gds_offset, sizeof(u32));
    return value;
}

bool Rasterizer::WriteLodStats(VAddr address, u32 num_bytes) {
    if (!lod_stats_enabled) {
        return false;
    }
    // Hardware reports sample counts below min_lod_warn in bits 32..55 and the finest requested LOD in bits
    // 56..59 (15 means not sampled). We cannot see those LODs, so report LOD 0 and the binding count for each
    // bank used since the last packet; this lets SotC stream finer mips within its memory budget
    // Leave the 64-byte header intact and map each entry to bank (index % 256)
    constexpr u32 HeaderBytes = 0x40;
    constexpr u32 MaxBytes = 16_MB;
    constexpr u64 NotSampled = u64{0xF} << 56;
    // Reject unmapped buffers and unreasonable sizes before staging entries on the host
    if (num_bytes <= HeaderBytes || num_bytes > MaxBytes || !IsMapped(address, num_bytes)) {
        LOG_WARNING(Render_Vulkan,
                    "Workaround lod_stats_from_bindings: IT_GET_LOD_STATS with invalid buffer "
                    "{:#x}+{:#x}, ignored",
                    address, num_bytes);
        return true;
    }
    const u32 num_entries = (num_bytes - HeaderBytes) / sizeof(u64);
    std::vector<u64> entries(num_entries);
    u32 banks_used = 0;
    for (const u32 uses : lod_stats_uses) {
        banks_used += uses != 0;
    }
    for (u32 i = 0; i < num_entries; ++i) {
        const u32 uses = lod_stats_uses[i % lod_stats_uses.size()];
        entries[i] = uses != 0 ? u64{uses} << 32 : NotSampled;
    }
    const VAddr dst = address + HeaderBytes;
    Common::GuestWriteJournal::Record(Common::GuestWriteJournal::Source::WriteData, dst,
                                      num_entries * sizeof(u64), entries.data(), 2);
    std::memcpy(reinterpret_cast<void*>(dst), entries.data(), num_entries * sizeof(u64));
    lod_stats_uses.fill(0);

    const auto now = std::chrono::steady_clock::now();
    if (lod_stats_packets_minute == 0 && lod_stats_last_report == decltype(now){}) {
        LOG_WARNING(Render_Vulkan,
                    "Workaround lod_stats_from_bindings: IT_GET_LOD_STATS at {:#x} ({} bytes) "
                    "answered with {} bound counter banks",
                    address, num_bytes, banks_used);
        lod_stats_last_report = now;
    }
    ++lod_stats_packets_minute;
    lod_stats_banks_minute += banks_used;
    if (now - lod_stats_last_report >= std::chrono::minutes{1}) {
        LOG_WARNING(Render_Vulkan,
                    "Workaround lod_stats_from_bindings: {} IT_GET_LOD_STATS in the last minute, "
                    "{} bound counter banks reported",
                    lod_stats_packets_minute, lod_stats_banks_minute);
        lod_stats_packets_minute = 0;
        lod_stats_banks_minute = 0;
        lod_stats_last_report = now;
    }
    return true;
}

void Rasterizer::ReportTsharpArray(u64 pgm_hash, u32 num_elements, u32 null_elements) {
    const auto now = std::chrono::steady_clock::now();
    if (tsharp_array_binds_minute == 0 && tsharp_array_last_report == decltype(now){}) {
        LOG_WARNING(Render_Vulkan,
                    "Workaround dynamic_tsharp_array_size: shader {:#x} binds {} T#s, {} null",
                    pgm_hash, num_elements, null_elements);
        tsharp_array_last_report = now;
    }
    ++tsharp_array_binds_minute;
    tsharp_array_null_minute += null_elements;
    if (now - tsharp_array_last_report >= std::chrono::minutes{1}) {
        LOG_WARNING(Render_Vulkan,
                    "Workaround dynamic_tsharp_array_size: {} T# arrays bound in the last minute, "
                    "{} null elements",
                    tsharp_array_binds_minute, tsharp_array_null_minute);
        tsharp_array_binds_minute = 0;
        tsharp_array_null_minute = 0;
        tsharp_array_last_report = now;
    }
}

bool Rasterizer::InvalidateMemory(VAddr addr, u64 size, bool assume_locks) {
    if (!IsMapped(addr, size)) {
        // Not GPU mapped memory, can skip invalidation logic entirely.
        Common::GuestWriteJournal::Record(
            Common::GuestWriteJournal::Source::FaultIgnoredNotGpuMapped, addr, size, nullptr, 1);
        return false;
    }
    Common::GuestWriteJournal::Record(Common::GuestWriteJournal::Source::FaultWrite, addr, size);
    {
        const Common::PerfStats::ScopedTimer perf_timer{Common::PerfStats::Id::Invalidates,
                                                        Common::PerfStats::Id::InvalidateBufferNs};
        buffer_cache.InvalidateMemory(addr, size, assume_locks);
    }
    const Common::PerfStats::ScopedNs perf_timer{Common::PerfStats::Id::InvalidateTextureNs};
    texture_cache.InvalidateMemory(addr, size);
    return true;
}

void Rasterizer::ReleaseCpuAuthoritativeRange(VAddr addr, u64 size) {
    if (IsMapped(addr, size)) {
        buffer_cache.ReleaseCpuAuthoritativeRange(addr, size);
    }
}

bool Rasterizer::ReadMemory(VAddr addr, u64 size, bool assume_locks) {
    if (!IsMapped(addr, size)) {
        // Not GPU mapped memory, can skip invalidation logic entirely.
        Common::GuestWriteJournal::Record(
            Common::GuestWriteJournal::Source::FaultIgnoredNotGpuMapped, addr, size, nullptr, 0);
        return false;
    }
    Common::GuestWriteJournal::Record(Common::GuestWriteJournal::Source::FaultRead, addr, size);
    buffer_cache.ReadMemory(addr, size, false, assume_locks);
    return true;
}

bool Rasterizer::ReadCleanMemory(VAddr addr, void* out, u64 size) {
    // Page by page through the clean page cache of the buffer cache: SotC serves tens of thousands of these
    // per frame, too many for the mapped-range and backing lookups with their locks
    using CleanRead = VideoCore::BufferCache::CleanRead;
    constexpr u64 PageSize = 4_KB;
    auto* dst = static_cast<u8*>(out);
    for (VAddr current = addr, end = addr + size; current < end;) {
        const VAddr page_addr = Common::AlignDown(current, PageSize);
        const u32 chunk = static_cast<u32>(std::min(end, page_addr + PageSize) - current);
        auto result = buffer_cache.ReadCleanPage(current, dst, chunk);
        if (result == CleanRead::Miss) {
            // The generation is read first: an unmap racing with the checks leaves a stale entry
            const u64 generation = buffer_cache.MappingGeneration();
            if (!IsMapped(page_addr, PageSize) ||
                !buffer_cache.FillCleanPage(page_addr, generation)) {
                return false;
            }
            result = buffer_cache.ReadCleanPage(current, dst, chunk);
        }
        if (result != CleanRead::Served) {
            return false;
        }
        current += chunk;
        dst += chunk;
    }
    static const bool verify = std::getenv("SHADPS4_CLEAN_READ_VERIFY") != nullptr;
    if (verify) {
        VerifyCleanRead(addr, out, size);
    }
    return true;
}

void Rasterizer::VerifyCleanRead(VAddr addr, const void* served, u64 size) {
    // Gate only: every clean read pays the readback it was meant to avoid. The readback runs as a fault on
    // this thread would, and afterwards the backing must still hold the bytes the clean read served
    static std::vector<u8> read_back;
    static u64 checks = 0;
    static u64 mismatches = 0;
    static auto last_report = std::chrono::steady_clock::now();
    buffer_cache.ReadMemory(addr, size, false, true);
    read_back.resize(size);
    if (!memory->TryReadBacking(addr, read_back.data(), size)) {
        return;
    }
    ++checks;
    const auto* served_bytes = static_cast<const u8*>(served);
    u64 offset = 0;
    while (offset < size && served_bytes[offset] == read_back[offset]) {
        ++offset;
    }
    if (offset != size) {
        ++mismatches;
        if (mismatches <= 32) {
            LOG_ERROR(Render_Vulkan,
                      "Clean read verify: byte {:#x} of {} at {:#x} was served as {:#04x}, the "
                      "readback left {:#04x}",
                      offset, size, addr, served_bytes[offset], read_back[offset]);
        }
    }
    const auto now = std::chrono::steady_clock::now();
    if (checks == 1 || now - last_report >= std::chrono::minutes{1}) {
        LOG_WARNING(Render_Vulkan, "Clean read verify: {} clean reads checked, {} mismatches",
                    checks, mismatches);
        last_report = now;
    }
}

bool Rasterizer::IsPlausibleImage(const VideoCore::ImageInfo& info, bool log) {
    // A garbage T# (e.g. read from a constant buffer the guest has not written yet) can pass the address and
    // format checks and still describe an image that Vulkan cannot create. Creating it exhausts device memory
    // (image.cpp allocation assert), so it is bound as null instead
    const auto& limits = instance.GetLimits();
    const bool is_3d = info.type == AmdGpu::ImageType::Color3D;
    const u32 max_dim = is_3d ? limits.maxImageDimension3D : limits.maxImageDimension2D;
    // For block-compressed formats num_bits is per 4x4 block
    const u32 block_dim = info.props.is_block ? 4 : 1;
    const u64 host_bytes = u64{(info.size.width + block_dim - 1) / block_dim} *
                           ((info.size.height + block_dim - 1) / block_dim) * info.size.depth *
                           info.resources.layers * std::max(info.num_samples, 1u) *
                           std::max(info.num_bits / 8, 1u);
    constexpr u64 MaxPlausibleImageBytes = 2ULL << 30;
    const bool plausible = info.size.width <= max_dim && info.size.height <= max_dim &&
                           (!is_3d || info.size.depth <= max_dim) &&
                           info.resources.layers <= limits.maxImageArrayLayers &&
                           host_bytes <= MaxPlausibleImageBytes;
    if (!plausible && log) {
        LOG_WARNING(Render_Vulkan,
                    "Rejecting implausible T# address={:#x} {}x{}x{} layers={} levels={} "
                    "bits={} (~{} MiB)",
                    info.guest_address, info.size.width, info.size.height, info.size.depth,
                    info.resources.layers, info.resources.levels, info.num_bits,
                    host_bytes >> 20);
    }
    return plausible;
}

bool Rasterizer::IsMapped(VAddr addr, u64 size) {
    if (size == 0) {
        // There is no memory, so not mapped.
        return false;
    }
    if (static_cast<u64>(addr) > std::numeric_limits<u64>::max() - size) {
        // Memory range wrapped the address space, cannot be mapped.
        return false;
    }
    if (mapped_page_table && Common::PerfStats::AbToggleOn(mapped_page_table_toggle)) {
        // BindBuffers checks every V# on every draw, and taking the shared lock then walking the
        // interval set cost about 6 ms per frame in SotC
        // The table answers directly when the range uses pages that are fully mapped or completely
        // empty, which covers most buffer bindings
        // Other ranges still use the interval set to get the exact byte-level answer
        const auto answer = mapped_page_table->Query(addr, addr + size);
        if (mapped_page_table_verify) [[unlikely]] {
            VerifyMappedPageTable(addr, size, answer);
        }
        if (answer != VideoCore::MappedPageTable::Answer::Unknown) {
            return answer == VideoCore::MappedPageTable::Answer::Mapped;
        }
    }
    if (recording_cuts && Common::PerfStats::AbToggleOn(recording_cuts_toggle)) {
        return IsMappedCached(addr, addr + size);
    }
    const auto range = decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);

    Common::RecursiveSharedLock lock{mapped_ranges_mutex};
    return boost::icl::contains(mapped_ranges, range);
}

void Rasterizer::VerifyMappedPageTable(VAddr addr, u64 size,
                                       VideoCore::MappedPageTable::Answer answer) {
    using Answer = VideoCore::MappedPageTable::Answer;
    static std::atomic<u64> queries{};
    static std::atomic<u64> unknown{};
    static std::atomic<u64> mismatches{};
    static std::atomic<s64> last_report_ms{};

    const u64 total = queries.fetch_add(1, std::memory_order_relaxed) + 1;
    if (answer == Answer::Unknown) {
        unknown.fetch_add(1, std::memory_order_relaxed);
    } else {
        bool exact;
        {
            const auto range =
                decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);
            Common::RecursiveSharedLock lock{mapped_ranges_mutex};
            exact = boost::icl::contains(mapped_ranges, range);
        }
        if (exact != (answer == Answer::Mapped)) {
            const u64 count = mismatches.fetch_add(1, std::memory_order_relaxed) + 1;
            if (count <= 16) {
                LOG_ERROR(Render_Vulkan,
                          "mapped_page_table: {:#x}+{:#x} table={} interval set={} (a map/unmap "
                          "between the two lookups also explains it)",
                          addr, size, answer == Answer::Mapped, exact);
            }
        }
    }

    const s64 now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                           std::chrono::steady_clock::now().time_since_epoch())
                           .count();
    s64 last_ms = last_report_ms.load(std::memory_order_relaxed);
    if (last_ms == 0) {
        last_report_ms.compare_exchange_strong(last_ms, now_ms, std::memory_order_relaxed);
        return;
    }
    if (now_ms - last_ms >= 60'000 &&
        last_report_ms.compare_exchange_strong(last_ms, now_ms, std::memory_order_relaxed)) {
        const u64 num_unknown = unknown.load(std::memory_order_relaxed);
        LOG_WARNING(Render_Vulkan,
                    "mapped_page_table verify: {} queries, {} unknown ({:.3f} %), {} mismatches",
                    total, num_unknown,
                    100.0 * static_cast<double>(num_unknown) / static_cast<double>(total),
                    mismatches.load(std::memory_order_relaxed));
    }
}

void Rasterizer::UpdateMappedPageTable(VAddr addr, u64 size, bool mapped) {
    if (!mapped_page_table) {
        return;
    }
    mapped_page_table->Update(addr, addr + size, mapped, [this](VAddr start, VAddr end) {
        u64 bytes = 0;
        for (const auto& range :
             mapped_ranges & decltype(mapped_ranges)::interval_type::right_open(start, end)) {
            bytes += boost::icl::length(range);
        }
        return bytes;
    });
}

bool Rasterizer::IsMappedCached(VAddr addr, VAddr end) {
    // BindBuffers asks this for every V# of every draw, tens of thousands of times per frame, and the
    // shared lock plus the interval tree walk cost several ms. The runs asked about change rarely: keep the
    // last few mapped runs per thread, stamped with the mapping generation read before they were looked up.
    // MapMemory/UnmapMemory bump the generation after changing mapped_ranges, so a stamped run is exact
    // until the next change. Only positive answers are cached
    struct MappedRun {
        u64 generation = ~u64{0};
        VAddr start = 0;
        VAddr last = 0;
    };
    static constexpr size_t NumRuns = 4;
    thread_local std::array<MappedRun, NumRuns> runs{};
    thread_local size_t next_run{};

    const u64 generation = buffer_cache.MappingGeneration();
    for (const auto& run : runs) {
        if (run.generation == generation && addr >= run.start && end - 1 <= run.last) {
            return true;
        }
    }

    Common::RecursiveSharedLock lock{mapped_ranges_mutex};
    // interval_set joins touching intervals, so the interval holding addr is the whole mapped run
    const auto it = mapped_ranges.find(addr);
    if (it == mapped_ranges.end()) {
        return false;
    }
    const VAddr run_start = boost::icl::first(*it);
    const VAddr run_last = boost::icl::last(*it);
    if (end - 1 > run_last) {
        return false;
    }
    runs[next_run] = {generation, run_start, run_last};
    next_run = (next_run + 1) % NumRuns;
    return true;
}

void Rasterizer::MapMemory(VAddr addr, u64 size) {
    Common::GuestWriteJournal::Record(Common::GuestWriteJournal::Source::GpuMap, addr, size);
    {
        std::scoped_lock lock{mapped_ranges_mutex};
        mapped_ranges += decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);
        UpdateMappedPageTable(addr, size, true);
    }
    buffer_cache.OnMappingChanged();
}

void Rasterizer::RegisterMemory(VAddr addr, u64 size) {
    page_manager.OnGpuMap(addr, size);
}

void Rasterizer::UnmapMemory(VAddr addr, u64 size) {
    Common::GuestWriteJournal::Record(Common::GuestWriteJournal::Source::GpuUnmap, addr, size);
    // Bump the mapping generation before removing the range as well as afterward
    // A bind reading it during removal must discard its cached answer, and an extra bump only
    // causes another safe invalidation
    buffer_cache.OnMappingChanged();
    buffer_cache.InvalidateMemory(addr, size);
    texture_cache.UnmapMemory(addr, size);
    {
        std::scoped_lock lock{mapped_ranges_mutex};
        mapped_ranges -= decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);
        UpdateMappedPageTable(addr, size, false);
    }
    buffer_cache.OnMappingChanged();
}

void Rasterizer::UpdateDynamicState(const GraphicsPipeline* pipeline, const bool is_indexed) const {
    UpdateViewportScissorState();
    UpdateDepthStencilState();
    UpdatePrimitiveState(is_indexed);
    UpdateRasterizationState();
    UpdateColorBlendingState(pipeline);

    auto& dynamic_state = scheduler.GetDynamicState();
    dynamic_state.Commit(instance, scheduler.CommandBuffer());
}

void Rasterizer::UpdateViewportScissorState() const {
    const auto& regs = liverpool->regs;

    const auto combined_scissor_value_tl = [](s16 scr, s16 win, s16 gen, s16 win_offset) {
        return std::max({scr, s16(win + win_offset), s16(gen + win_offset)});
    };
    const auto combined_scissor_value_br = [](s16 scr, s16 win, s16 gen, s16 win_offset) {
        return std::min({scr, s16(win + win_offset), s16(gen + win_offset)});
    };
    const bool enable_offset = !regs.window_scissor.window_offset_disable;

    AmdGpu::Scissor scsr{};
    scsr.top_left_x = combined_scissor_value_tl(
        regs.screen_scissor.top_left_x, s16(regs.window_scissor.top_left_x),
        s16(regs.generic_scissor.top_left_x),
        enable_offset ? regs.window_offset.window_x_offset : 0);
    scsr.top_left_y = combined_scissor_value_tl(
        regs.screen_scissor.top_left_y, s16(regs.window_scissor.top_left_y),
        s16(regs.generic_scissor.top_left_y),
        enable_offset ? regs.window_offset.window_y_offset : 0);
    scsr.bottom_right_x = combined_scissor_value_br(
        regs.screen_scissor.bottom_right_x, regs.window_scissor.bottom_right_x,
        regs.generic_scissor.bottom_right_x,
        enable_offset ? regs.window_offset.window_x_offset : 0);
    scsr.bottom_right_y = combined_scissor_value_br(
        regs.screen_scissor.bottom_right_y, regs.window_scissor.bottom_right_y,
        regs.generic_scissor.bottom_right_y,
        enable_offset ? regs.window_offset.window_y_offset : 0);

    boost::container::static_vector<vk::Viewport, AmdGpu::NUM_VIEWPORTS> viewports;
    boost::container::static_vector<vk::Rect2D, AmdGpu::NUM_VIEWPORTS> scissors;

    if (regs.polygon_control.enable_window_offset &&
        (regs.window_offset.window_x_offset != 0 || regs.window_offset.window_y_offset != 0)) {
        LOG_ERROR(Render_Vulkan,
                  "PA_SU_SC_MODE_CNTL.VTX_WINDOW_OFFSET_ENABLE support is not yet implemented.");
    }

    const auto& vp_ctl = regs.viewport_control;
    for (u32 i = 0; i < AmdGpu::NUM_VIEWPORTS; i++) {
        const auto& vp = regs.viewports[i];
        const auto& vp_d = regs.viewport_depths[i];
        if (vp.xscale == 0) {
            continue;
        }

        const auto zoffset = vp_ctl.zoffset_enable ? vp.zoffset : 0.f;
        const auto zscale = vp_ctl.zscale_enable ? vp.zscale : 1.f;

        vk::Viewport viewport{};

        // https://gitlab.freedesktop.org/mesa/mesa/-/blob/209a0ed/src/amd/vulkan/radv_pipeline_graphics.c#L688-689
        // https://gitlab.freedesktop.org/mesa/mesa/-/blob/209a0ed/src/amd/vulkan/radv_cmd_buffer.c#L3103-3109
        // When the clip space is ranged [-1...1], the zoffset is centered.
        // By reversing the above viewport calculations, we get the following:
        if (regs.clipper_control.clip_space == AmdGpu::ClipSpace::MinusWToW) {
            viewport.minDepth = zoffset - zscale;
            viewport.maxDepth = zoffset + zscale;
        } else {
            viewport.minDepth = zoffset;
            viewport.maxDepth = zoffset + zscale;
        }

        if (!instance.IsDepthRangeUnrestrictedSupported()) {
            // Unrestricted depth range not supported by device. Restrict to valid range.
            viewport.minDepth = std::max(viewport.minDepth, 0.f);
            viewport.maxDepth = std::min(viewport.maxDepth, 1.f);
        }

        if (regs.IsClipDisabled()) {
            // In case if clipping is disabled we patch the shader to convert vertex position
            // from screen space coordinates to NDC by defining a render space as full hardware
            // window range [0..16383, 0..16383] and setting the viewport to its size.
            viewport.x = 0.f;
            viewport.y = 0.f;
            viewport.width = float(std::min<u32>(instance.GetMaxViewportWidth(), 16_KB));
            viewport.height = float(std::min<u32>(instance.GetMaxViewportHeight(), 16_KB));
        } else {
            const auto xoffset = vp_ctl.xoffset_enable ? vp.xoffset : 0.f;
            const auto xscale = vp_ctl.xscale_enable ? vp.xscale : 1.f;
            const auto yoffset = vp_ctl.yoffset_enable ? vp.yoffset : 0.f;
            const auto yscale = vp_ctl.yscale_enable ? vp.yscale : 1.f;

            viewport.x = xoffset - xscale;
            viewport.y = yoffset - yscale;
            viewport.width = xscale * 2.0f;
            viewport.height = yscale * 2.0f;
        }

        viewports.push_back(viewport);

        auto vp_scsr = scsr;
        if (regs.mode_control.vport_scissor_enable) {
            vp_scsr.top_left_x =
                std::max(vp_scsr.top_left_x, s16(regs.viewport_scissors[i].top_left_x));
            vp_scsr.top_left_y =
                std::max(vp_scsr.top_left_y, s16(regs.viewport_scissors[i].top_left_y));
            vp_scsr.bottom_right_x = std::min(AmdGpu::Scissor::Clamp(vp_scsr.bottom_right_x),
                                              regs.viewport_scissors[i].bottom_right_x);
            vp_scsr.bottom_right_y = std::min(AmdGpu::Scissor::Clamp(vp_scsr.bottom_right_y),
                                              regs.viewport_scissors[i].bottom_right_y);
        }
        scissors.push_back({
            .offset = {vp_scsr.top_left_x, vp_scsr.top_left_y},
            .extent = {vp_scsr.GetWidth(), vp_scsr.GetHeight()},
        });
    }

    if (viewports.empty()) {
        // Vulkan requires providing at least one viewport.
        constexpr vk::Viewport empty_viewport = {
            .x = -1.0f,
            .y = -1.0f,
            .width = 1.0f,
            .height = 1.0f,
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };
        constexpr vk::Rect2D empty_scissor = {
            .offset = {0, 0},
            .extent = {1, 1},
        };
        viewports.push_back(empty_viewport);
        scissors.push_back(empty_scissor);
    }

    auto& dynamic_state = scheduler.GetDynamicState();
    dynamic_state.SetViewports(viewports);
    dynamic_state.SetScissors(scissors);
}

void Rasterizer::UpdateDepthStencilState() const {
    const auto& regs = liverpool->regs;
    auto& dynamic_state = scheduler.GetDynamicState();

    const auto depth_test_enabled =
        regs.depth_control.depth_enable && regs.depth_buffer.DepthValid();
    dynamic_state.SetDepthTestEnabled(depth_test_enabled);
    if (depth_test_enabled) {
        dynamic_state.SetDepthWriteEnabled(regs.depth_control.depth_write_enable &&
                                           !regs.depth_render_control.depth_clear_enable);
        dynamic_state.SetDepthCompareOp(LiverpoolToVK::CompareOp(regs.depth_control.depth_func));
    }

    const auto depth_bounds_test_enabled = regs.depth_control.depth_bounds_enable;
    dynamic_state.SetDepthBoundsTestEnabled(depth_bounds_test_enabled);
    if (depth_bounds_test_enabled) {
        dynamic_state.SetDepthBounds(regs.depth_bounds_min, regs.depth_bounds_max);
    }

    const auto depth_bias_enabled = regs.polygon_control.NeedsBias();
    dynamic_state.SetDepthBiasEnabled(depth_bias_enabled);
    if (depth_bias_enabled) {
        const bool front = regs.polygon_control.enable_polygon_offset_front;
        dynamic_state.SetDepthBias(
            front ? regs.poly_offset.front_offset : regs.poly_offset.back_offset,
            regs.poly_offset.depth_bias,
            (front ? regs.poly_offset.front_scale : regs.poly_offset.back_scale) / 16.f);
    }

    const auto stencil_test_enabled =
        regs.depth_control.stencil_enable && regs.depth_buffer.StencilValid();
    dynamic_state.SetStencilTestEnabled(stencil_test_enabled);
    if (stencil_test_enabled) {
        const StencilOps front_ops{
            .fail_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_fail_front),
            .pass_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_zpass_front),
            .depth_fail_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_zfail_front),
            .compare_op = LiverpoolToVK::CompareOp(regs.depth_control.stencil_ref_func),
        };
        const StencilOps back_ops = regs.depth_control.backface_enable ? StencilOps{
            .fail_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_fail_back),
            .pass_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_zpass_back),
            .depth_fail_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_zfail_back),
            .compare_op = LiverpoolToVK::CompareOp(regs.depth_control.stencil_bf_func),
        } : front_ops;
        dynamic_state.SetStencilOps(front_ops, back_ops);

        const bool stencil_clear = regs.depth_render_control.stencil_clear_enable;
        const auto front = regs.stencil_ref_front;
        const auto back =
            regs.depth_control.backface_enable ? regs.stencil_ref_back : regs.stencil_ref_front;
        // GCN REPLACE_OP writes DB_STENCILREFMASK.STENCILOPVAL, so a face whose stencil ops
        // include ReplaceOp takes its Vulkan reference from op_val.
        const auto& sc = regs.stencil_control;
        const auto uses_op_val = [](AmdGpu::StencilFunc fail, AmdGpu::StencilFunc zpass,
                                    AmdGpu::StencilFunc zfail) {
            return fail == AmdGpu::StencilFunc::ReplaceOp ||
                   zpass == AmdGpu::StencilFunc::ReplaceOp ||
                   zfail == AmdGpu::StencilFunc::ReplaceOp;
        };
        const bool front_op =
            uses_op_val(sc.stencil_fail_front, sc.stencil_zpass_front, sc.stencil_zfail_front);
        const bool back_op =
            regs.depth_control.backface_enable
                ? uses_op_val(sc.stencil_fail_back, sc.stencil_zpass_back, sc.stencil_zfail_back)
                : front_op;
        const auto ref_conflict = [](AmdGpu::CompareFunc func, const AmdGpu::StencilRefMask& ref) {
            return func != AmdGpu::CompareFunc::Always && func != AmdGpu::CompareFunc::Never &&
                   ref.stencil_test_val != ref.stencil_op_val;
        };
        if ((front_op && ref_conflict(regs.depth_control.stencil_ref_func, front)) ||
            (back_op && regs.depth_control.backface_enable &&
             ref_conflict(regs.depth_control.stencil_bf_func, back))) {
            LOG_WARNING(Render_Vulkan, "Stencil test requires test_val while ReplaceOp requires "
                                       "op_val; the stencil test will use op_val");
        }
        dynamic_state.SetStencilReferences(front_op ? front.stencil_op_val : front.stencil_test_val,
                                           back_op ? back.stencil_op_val : back.stencil_test_val);
        dynamic_state.SetStencilWriteMasks(!stencil_clear ? front.stencil_write_mask : 0U,
                                           !stencil_clear ? back.stencil_write_mask : 0U);
        dynamic_state.SetStencilCompareMasks(front.stencil_mask, back.stencil_mask);
    }
}

void Rasterizer::UpdatePrimitiveState(const bool is_indexed) const {
    const auto& regs = liverpool->regs;
    auto& dynamic_state = scheduler.GetDynamicState();

    const auto is_list_topology = [](const AmdGpu::PrimitiveType type) {
        const auto topology = LiverpoolToVK::PrimitiveType(type);
        return topology == vk::PrimitiveTopology::ePointList ||
               topology == vk::PrimitiveTopology::eLineList ||
               topology == vk::PrimitiveTopology::eTriangleList ||
               topology == vk::PrimitiveTopology::eLineListWithAdjacency ||
               topology == vk::PrimitiveTopology::eTriangleListWithAdjacency;
    };
    const auto is_patch_list_topology = [](const AmdGpu::PrimitiveType type) {
        // Quad and rect lists are emulated using tessellation.
        return type == AmdGpu::PrimitiveType::PatchPrimitive ||
               type == AmdGpu::PrimitiveType::QuadList || type == AmdGpu::PrimitiveType::RectList;
    };

    const auto prim_restart =
        (regs.enable_primitive_restart & 1) != 0 &&
        (instance.IsListRestartSupported() || !is_list_topology(regs.primitive_type)) &&
        (instance.IsPatchListRestartSupported() || !is_patch_list_topology(regs.primitive_type));
    ASSERT_MSG(!is_indexed || !prim_restart || regs.primitive_restart_index == 0xFFFF ||
                   regs.primitive_restart_index == 0xFFFFFFFF,
               "Primitive restart index other than -1 is not supported yet");

    const auto cull_mode = LiverpoolToVK::IsPrimitiveCulled(regs.primitive_type)
                               ? LiverpoolToVK::CullMode(regs.polygon_control.CullingMode())
                               : vk::CullModeFlagBits::eNone;
    const auto front_face = LiverpoolToVK::FrontFace(regs.polygon_control.front_face);

    dynamic_state.SetPrimitiveRestartEnabled(prim_restart);
    dynamic_state.SetRasterizerDiscardEnabled(regs.clipper_control.dx_rasterization_kill);
    dynamic_state.SetCullMode(cull_mode);
    dynamic_state.SetFrontFace(front_face);
}

void Rasterizer::UpdateRasterizationState() const {
    const auto& regs = liverpool->regs;
    auto& dynamic_state = scheduler.GetDynamicState();
    dynamic_state.SetLineWidth(regs.line_control.Width());
}

void Rasterizer::UpdateColorBlendingState(const GraphicsPipeline* pipeline) const {
    const auto& regs = liverpool->regs;
    auto& dynamic_state = scheduler.GetDynamicState();
    dynamic_state.SetBlendConstants(regs.blend_constants);
    dynamic_state.SetColorWriteMasks(pipeline->GetGraphicsKey().write_masks);
    dynamic_state.SetAttachmentFeedbackLoopEnabled(attachment_feedback_loop);
}

void Rasterizer::ScopeMarkerBegin(const std::string_view& str, bool from_guest) {
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.beginDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{
        .pLabelName = str.data(),
    });
}

void Rasterizer::ScopeMarkerEnd(bool from_guest) {
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.endDebugUtilsLabelEXT();
}

void Rasterizer::ScopedMarkerInsert(const std::string_view& str, bool from_guest) {
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.insertDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{
        .pLabelName = str.data(),
    });
}

void Rasterizer::ScopedMarkerInsertColor(const std::string_view& str, const u32 color,
                                         bool from_guest) {
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.insertDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{
        .pLabelName = str.data(),
        .color = std::array<f32, 4>(
            {(f32)((color >> 16) & 0xff) / 255.0f, (f32)((color >> 8) & 0xff) / 255.0f,
             (f32)(color & 0xff) / 255.0f, (f32)((color >> 24) & 0xff) / 255.0f})});
}

std::thread::id Rasterizer::GetGpuCommandProcessorThread() {
    return liverpool->GetGpuCommandProcessorThread();
}

#ifdef __linux__
u32 Rasterizer::GetGpuCommandProcessorThreadId() {
    return liverpool->GetGpuCommandProcessorThreadId();
}
#endif

} // namespace Vulkan
