// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <bit>
#include <chrono>
#include <limits>

#include "common/assert.h"
#include "common/debug.h"
#include "common/perf_stats.h"
#include "common/thread.h"
#include "core/emulator_settings.h"
#include "imgui/renderer/texture_manager.h"
#include "video_core/renderer_vulkan/vk_gpu_checkpoints.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

namespace Vulkan {

std::mutex Scheduler::submit_mutex;

Scheduler::Scheduler(const Instance& instance)
    : instance{instance}, work_semaphore{instance}, command_pool{instance, &work_semaphore},
      wait_spin_us{EmulatorSettings.GetWaitSpinUs()} {
#if TRACY_GPU_ENABLED
    profiler_scope = reinterpret_cast<tracy::VkCtxScope*>(std::malloc(sizeof(tracy::VkCtxScope)));
#endif
    BeginSession();
    priority_pending_ops_thread =
        std::jthread(std::bind_front(&Scheduler::PriorityPendingOpsThread, this));
}

Scheduler::~Scheduler() {
#if TRACY_GPU_ENABLED
    std::free(profiler_scope);
#endif
}

void Scheduler::BeginRendering(const RenderState& new_state) {
    if (is_rendering && render_state == new_state) {
        return;
    }
    EndRendering();
    is_rendering = true;
    render_state = new_state;

    std::array<vk::RenderingAttachmentInfo, 8> color_attachments;
    for (u32 i = 0; i < render_state.num_color_attachments; ++i) {
        const auto& cb = render_state.color_attachments[i];
        color_attachments[i] = vk::RenderingAttachmentInfo{
            .imageView = cb.image_view,
            .imageLayout = cb.image_layout,
            .loadOp = cb.is_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue = vk::ClearValue{.color = vk::ClearColorValue{.uint32 = cb.clear_value}},
        };
    }

    const auto& db = render_state.depth_stencil_attachment;
    const vk::RenderingAttachmentInfo depth_attachment = {
        .imageView = db.image_view,
        .imageLayout = db.image_layout,
        .loadOp = db.depth_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue =
            vk::ClearValue{.depthStencil = vk::ClearDepthStencilValue{.depth = std::bit_cast<float>(
                                                                          db.clear_value[0])}},
    };
    const vk::RenderingAttachmentInfo stencil_attachment = {
        .imageView = db.image_view,
        .imageLayout = db.image_layout,
        .loadOp = db.stencil_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = vk::ClearValue{.depthStencil =
                                         vk::ClearDepthStencilValue{.stencil = db.clear_value[1]}},
    };

    const vk::RenderingInfo rendering_info = {
        .renderArea =
            {
                .offset = {0, 0},
                .extent = {render_state.width, render_state.height},
            },
        .layerCount = render_state.num_layers,
        .colorAttachmentCount = render_state.num_color_attachments,
        .pColorAttachments = color_attachments.data(),
        .pDepthAttachment = db.has_depth ? &depth_attachment : nullptr,
        .pStencilAttachment = db.has_stencil ? &stencil_attachment : nullptr,
    };

    CommandBuffer().beginRendering(rendering_info);
}

void Scheduler::EndRendering() {
    if (!is_rendering) {
        return;
    }
    is_rendering = false;
    CommandBuffer().endRendering();
}

vk::CommandBuffer Scheduler::UploadCommandBuffer() {
    auto& upload_cmdbuf = sessions.back().upload;
    if (upload_cmdbuf) {
        return upload_cmdbuf;
    }
    const vk::CommandBufferBeginInfo begin_info = {
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
    };
    upload_cmdbuf = command_pool.Commit();
    Check(upload_cmdbuf.begin(begin_info));
    return upload_cmdbuf;
}

void Scheduler::Flush(SubmitInfo& info) {
    // When flushing, we only send data to the driver; no waiting is necessary.
    SubmitExecution(info);
}

void Scheduler::Flush() {
    SubmitInfo info{};
    Flush(info);
}

void Scheduler::Finish() {
    const Common::PerfStats::ScopedTimer perf_timer{Common::PerfStats::Id::VkFinish,
                                                    Common::PerfStats::Id::VkFinishNs};
    // When finishing, we need to wait for the submission to have executed on the device.
    const u64 presubmit_tick = CurrentTick();
    SubmitInfo info{};
    SubmitExecution(info);
    // The tick was just submitted, so Wait() would never flush; waiting on the semaphore directly keeps
    // Finish out of the vk_wait counters
    work_semaphore.Wait(presubmit_tick);
}

vk::CommandBuffer Scheduler::BeginAhead() {
    const vk::Device device = instance.GetDevice();
    // readback_ahead_transfer_queue honours SHADPS4_AB_TOGGLE: the off intervals use a graphics-family command
    // buffer on the graphics queue, as without the key
    static const bool follows_toggle =
        Common::PerfStats::AbToggleFollows("readback_ahead_transfer_queue");
    ahead_on_transfer =
        instance.GetTransferQueue() && Common::PerfStats::AbToggleOn(follows_toggle);
    auto& slot = ahead_slots[ahead_on_transfer ? 1 : 0];
    if (!slot.pool) {
        const vk::CommandPoolCreateInfo pool_info = {
            .flags = vk::CommandPoolCreateFlagBits::eTransient |
                     vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            .queueFamilyIndex = ahead_on_transfer ? instance.GetTransferQueueFamilyIndex()
                                                  : instance.GetGraphicsQueueFamilyIndex(),
        };
        auto [pool_result, pool] = device.createCommandPoolUnique(pool_info);
        ASSERT_MSG(pool_result == vk::Result::eSuccess, "Failed to create the ahead pool: {}",
                   vk::to_string(pool_result));
        slot.pool = std::move(pool);
        const vk::CommandBufferAllocateInfo alloc_info = {
            .commandPool = *slot.pool,
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = 1,
        };
        auto [alloc_result, cmdbufs] = device.allocateCommandBuffers(alloc_info);
        ASSERT_MSG(alloc_result == vk::Result::eSuccess,
                   "Failed to allocate the ahead command buffer: {}", vk::to_string(alloc_result));
        slot.cmdbuf = cmdbufs[0];
    }
    if (!ahead_fence) {
        auto [fence_result, fence] = device.createFenceUnique({});
        ASSERT_MSG(fence_result == vk::Result::eSuccess, "Failed to create the ahead fence: {}",
                   vk::to_string(fence_result));
        ahead_fence = std::move(fence);
    } else {
        Check(device.resetFences(*ahead_fence));
    }
    ahead_cmdbuf = slot.cmdbuf;
    Check(ahead_cmdbuf.reset());
    Check(ahead_cmdbuf.begin(vk::CommandBufferBeginInfo{
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
    }));
    return ahead_cmdbuf;
}

void Scheduler::SubmitAhead(u64 wait_tick) {
    Check(ahead_cmdbuf.end());
    // On the transfer queue, the timeline wait orders the copy after its writer and makes the writes visible;
    // on the graphics queue, submission order and the copy's barrier do
    const vk::Queue transfer_queue = ahead_on_transfer ? instance.GetTransferQueue() : vk::Queue{};
    const vk::Semaphore timeline = work_semaphore.Handle();
    static constexpr vk::PipelineStageFlags wait_stage = vk::PipelineStageFlagBits::eTransfer;
    const vk::TimelineSemaphoreSubmitInfo timeline_si = {
        .waitSemaphoreValueCount = 1U,
        .pWaitSemaphoreValues = &wait_tick,
    };
    const vk::SubmitInfo submit_info = {
        .pNext = transfer_queue ? &timeline_si : nullptr,
        .waitSemaphoreCount = transfer_queue ? 1U : 0U,
        .pWaitSemaphores = &timeline,
        .pWaitDstStageMask = &wait_stage,
        .commandBufferCount = 1U,
        .pCommandBuffers = &ahead_cmdbuf,
    };
    std::scoped_lock lk{submit_mutex};
    if (tracks_gpu_commands) {
        // gpu_checkpoints: the ahead copy carries no guest command and is queued after the submitted ones (on
        // the transfer queue it only waits for tick wait_tick of them). The previous one was waited for, so it
        // is the last known completed
        GpuCheckpoints::RecordSubmit(std::bit_cast<u64>(static_cast<VkFence>(*ahead_fence)),
                                     ahead_submits + 1, submitted_seq + 1, submitted_seq,
                                     ahead_submits, true);
    }
    ++ahead_submits;
    const auto submit_result = (transfer_queue ? transfer_queue : instance.GetGraphicsQueue())
                                   .submit(submit_info, *ahead_fence);
    if (submit_result == vk::Result::eErrorDeviceLost) {
        instance.ReportDeviceFault("SubmitAhead");
    }
    ASSERT_MSG(submit_result == vk::Result::eSuccess, "Ahead submit failed: {}",
               vk::to_string(submit_result));
}

void Scheduler::WaitAhead(bool writers_done) {
    using namespace Common::PerfStats;
    const bool perf = Enabled();
    const auto start = perf ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const vk::Device device = instance.GetDevice();
    const bool spun = SpinUntil(wait_spin_us, [&] {
        return device.getFenceStatus(*ahead_fence) != vk::Result::eNotReady;
    });
    const auto wait_result =
        spun ? device.getFenceStatus(*ahead_fence)
             : device.waitForFences(*ahead_fence, true, std::numeric_limits<u64>::max());
    if (wait_result == vk::Result::eErrorDeviceLost) {
        instance.ReportDeviceFault("WaitAhead");
    }
    ASSERT_MSG(wait_result == vk::Result::eSuccess, "Ahead wait failed: {}",
               vk::to_string(wait_result));
    if (tracks_gpu_commands) {
        GpuCheckpoints::g_ahead_completed.store(ahead_submits, std::memory_order_relaxed);
    }
    if (perf) {
        const u64 ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                           std::chrono::steady_clock::now() - start)
                           .count();
        Add(Id::ReadbackAhead);
        Add(Id::ReadbackAheadNs, ns);
        if (writers_done) {
            Add(Id::ReadbackAheadIdle);
            Add(Id::ReadbackAheadIdleNs, ns);
        }
    }
}

void Scheduler::Wait(u64 tick) {
    const Common::PerfStats::ScopedTimer perf_timer{Common::PerfStats::Id::VkWait,
                                                    Common::PerfStats::Id::VkWaitNs};
    if (tick >= work_semaphore.CurrentTick()) {
        // Make sure we are not waiting for the current tick without signalling
        SubmitInfo info{};
        Flush(info);
    }
    work_semaphore.Wait(tick);
}

void Scheduler::PopPendingOperations() {
    std::unique_lock lk(pending_ops_mutex);
    work_semaphore.Refresh();
    while (!pending_ops.empty() && work_semaphore.IsFree(pending_ops.front().gpu_tick)) {
        pending_ops.front().callback();
        pending_ops.pop();
    }
}

void Scheduler::BeginSession() {
    EndSession();

    auto& session = sessions.emplace_back();

    const vk::CommandBufferBeginInfo begin_info = {
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
    };
    session.primary = command_pool.Commit();
    Check(session.primary.begin(begin_info));
    if (sessions.size() == 1) {
        // First session of a submit
        BeginGpuTiming();
    }

    // Invalidate dynamic state so it gets applied to the new command buffer.
    dynamic_state.Invalidate();
    InvalidatePipelineBinds();

#if TRACY_GPU_ENABLED
    auto* profiler_ctx = instance.GetProfilerContext();
    if (profiler_ctx) {
        static const auto scope_loc =
            GPU_SCOPE_LOCATION("Guest Frame", MarkersPalette::GpuMarkerColor);
        new (profiler_scope) tracy::VkCtxScope{profiler_ctx, &scope_loc, current_cmdbuf, true};
    }
#endif
}

void Scheduler::EndSession(u64 submit_tick) {
    if (sessions.empty()) {
        return;
    }

    if (on_session) {
        on_session();
    }

    const auto& session = sessions.back();
    if (session.upload) {
        Check(session.upload.end());
    }

    EndRendering();
    if (submit_tick != 0) {
        EndGpuTiming(submit_tick);
    }
    Check(session.primary.end());
}

void Scheduler::SubmitExecution(SubmitInfo& info) {
    std::scoped_lock lk{submit_mutex};
    const u64 signal_value = work_semaphore.NextTick();

#if TRACY_GPU_ENABLED
    auto* profiler_ctx = instance.GetProfilerContext();
    if (profiler_ctx) {
        profiler_scope->~VkCtxScope();
        TracyVkCollect(profiler_ctx, current_cmdbuf);
    }
#endif

    if (on_submit) {
        on_submit(info);
    }

    EndSession(signal_value);

    std::vector<vk::CommandBuffer> cmd_buffers;
    cmd_buffers.reserve(sessions.size() * 2);

    for (const auto& session : sessions) {
        if (session.upload) {
            cmd_buffers.push_back(session.upload);
        }
        cmd_buffers.push_back(session.primary);
    }
    sessions.clear();
    commands_since_submit = 0;

    const vk::Semaphore timeline = work_semaphore.Handle();
    info.AddSignal(timeline, signal_value);

    static constexpr std::array<vk::PipelineStageFlags, 2> wait_stage_masks = {
        vk::PipelineStageFlagBits::eAllCommands,
        vk::PipelineStageFlagBits::eColorAttachmentOutput,
    };

    const vk::TimelineSemaphoreSubmitInfo timeline_si = {
        .waitSemaphoreValueCount = info.num_wait_semas,
        .pWaitSemaphoreValues = info.wait_ticks.data(),
        .signalSemaphoreValueCount = info.num_signal_semas,
        .pSignalSemaphoreValues = info.signal_ticks.data(),
    };

    const vk::SubmitInfo submit_info = {
        .pNext = &timeline_si,
        .waitSemaphoreCount = info.num_wait_semas,
        .pWaitSemaphores = info.wait_semas.data(),
        .pWaitDstStageMask = wait_stage_masks.data(),
        .commandBufferCount = static_cast<u32>(cmd_buffers.size()),
        .pCommandBuffers = cmd_buffers.data(),
        .signalSemaphoreCount = info.num_signal_semas,
        .pSignalSemaphores = info.signal_semas.data(),
    };

    if (tracks_gpu_commands) {
        const u64 last_seq = GpuCheckpoints::g_seq.load(std::memory_order_relaxed);
        GpuCheckpoints::RecordSubmit(std::bit_cast<u64>(static_cast<VkSemaphore>(timeline)),
                                     signal_value, submitted_seq + 1, last_seq,
                                     work_semaphore.KnownGpuTick());
        submitted_seq = last_seq;
    }

    ImGui::Core::TextureManager::Submit();
    auto submit_result = instance.GetGraphicsQueue().submit(submit_info, info.fence);
    if (submit_result == vk::Result::eErrorDeviceLost) {
        instance.ReportDeviceFault("SubmitExecution");
    }
    ASSERT_MSG(submit_result != vk::Result::eErrorDeviceLost, "Device lost during submit");

    work_semaphore.Refresh();
    CollectGpuTiming();
    BeginSession();

    // Apply pending operations
    PopPendingOperations();
}

void Scheduler::EnableGpuTiming() {
    if (!Common::PerfStats::Enabled()) {
        return;
    }
    std::scoped_lock lk{submit_mutex};
    const vk::QueryPoolCreateInfo pool_info = {
        .queryType = vk::QueryType::eTimestamp,
        .queryCount = GpuTimingSlots * 2,
    };
    auto [result, pool] = instance.GetDevice().createQueryPoolUnique(pool_info);
    if (result != vk::Result::eSuccess) {
        LOG_WARNING(Render_Vulkan, "Perf stats: no timestamp query pool ({}), gpu_ms stays 0",
                    vk::to_string(result));
        return;
    }
    gpu_timing_pool = std::move(pool);
    gpu_timestamp_period_ns = instance.GetLimits().timestampPeriod;
    // The command buffer already open has no start timestamp; timing begins with the next submit
}

void Scheduler::BeginGpuTiming() {
    gpu_timing_open = -1;
    if (!gpu_timing_pool) {
        return;
    }
    const u32 slot = gpu_timing_next;
    if (gpu_timing_busy[slot]) {
        // The GPU is more than GpuTimingSlots submits behind; skip this one
        return;
    }
    gpu_timing_next = (gpu_timing_next + 1) % GpuTimingSlots;
    const auto cmdbuf = sessions.back().primary;
    cmdbuf.resetQueryPool(*gpu_timing_pool, slot * 2, 2);
    cmdbuf.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, *gpu_timing_pool, slot * 2);
    gpu_timing_open = static_cast<s32>(slot);
}

void Scheduler::EndGpuTiming(u64 submit_tick) {
    if (gpu_timing_open < 0) {
        return;
    }
    const u32 slot = static_cast<u32>(gpu_timing_open);
    sessions.back().primary.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe,
                                           *gpu_timing_pool, slot * 2 + 1);
    gpu_timing_busy[slot] = true;
    gpu_timing_pending.push_back({slot, submit_tick});
    gpu_timing_open = -1;
    Common::PerfStats::Add(Common::PerfStats::Id::VkSubmits);
}

void Scheduler::CollectGpuTiming() {
    while (!gpu_timing_pending.empty() &&
           work_semaphore.IsFree(gpu_timing_pending.front().tick)) {
        const u32 slot = gpu_timing_pending.front().slot;
        std::array<u64, 2> stamps{};
        const vk::Result result = instance.GetDevice().getQueryPoolResults(
            *gpu_timing_pool, slot * 2, 2, sizeof(stamps), stamps.data(), sizeof(u64),
            vk::QueryResultFlagBits::e64);
        if (result == vk::Result::eSuccess && stamps[1] >= stamps[0]) {
            Common::PerfStats::Add(
                Common::PerfStats::Id::GpuNs,
                static_cast<u64>(static_cast<double>(stamps[1] - stamps[0]) *
                                 gpu_timestamp_period_ns));
        }
        gpu_timing_busy[slot] = false;
        gpu_timing_pending.pop_front();
    }
}

void Scheduler::PriorityPendingOpsThread(std::stop_token stoken) {
    Common::SetCurrentThreadName("shadPS4:GpuSchedPriorityPendingOpsRunner");

    while (!stoken.stop_requested()) {
        PendingOp op;
        {
            std::unique_lock lk(priority_pending_ops_mutex);
            priority_pending_ops_cv.wait(lk, stoken,
                                         [this] { return !priority_pending_ops.empty(); });
            if (stoken.stop_requested()) {
                break;
            }

            op = std::move(priority_pending_ops.front());
            priority_pending_ops.pop();
        }

        work_semaphore.Wait(op.gpu_tick);
        if (stoken.stop_requested()) {
            break;
        }

        op.callback();
    }
}

void DynamicState::Commit(const Instance& instance, const vk::CommandBuffer& cmdbuf) {
    if (dirty_state.viewports) {
        dirty_state.viewports = false;
        cmdbuf.setViewportWithCount(viewports);
    }
    if (dirty_state.scissors) {
        dirty_state.scissors = false;
        cmdbuf.setScissorWithCount(scissors);
    }
    if (dirty_state.depth_test_enabled) {
        dirty_state.depth_test_enabled = false;
        cmdbuf.setDepthTestEnable(depth_test_enabled);
    }
    if (dirty_state.depth_write_enabled) {
        dirty_state.depth_write_enabled = false;
        // Note that this must be set in a command buffer even if depth test is disabled.
        cmdbuf.setDepthWriteEnable(depth_write_enabled);
    }
    if (depth_test_enabled && dirty_state.depth_compare_op) {
        dirty_state.depth_compare_op = false;
        cmdbuf.setDepthCompareOp(depth_compare_op);
    }
    if (dirty_state.depth_bounds_test_enabled) {
        dirty_state.depth_bounds_test_enabled = false;
        if (instance.IsDepthBoundsSupported()) {
            cmdbuf.setDepthBoundsTestEnable(depth_bounds_test_enabled);
        }
    }
    if (depth_bounds_test_enabled && dirty_state.depth_bounds) {
        dirty_state.depth_bounds = false;
        if (instance.IsDepthBoundsSupported()) {
            cmdbuf.setDepthBounds(depth_bounds_min, depth_bounds_max);
        }
    }
    if (dirty_state.depth_bias_enabled) {
        dirty_state.depth_bias_enabled = false;
        cmdbuf.setDepthBiasEnable(depth_bias_enabled);
    }
    if (depth_bias_enabled && dirty_state.depth_bias) {
        dirty_state.depth_bias = false;
        cmdbuf.setDepthBias(depth_bias_constant, depth_bias_clamp, depth_bias_slope);
    }
    if (dirty_state.stencil_test_enabled) {
        dirty_state.stencil_test_enabled = false;
        cmdbuf.setStencilTestEnable(stencil_test_enabled);
    }
    if (stencil_test_enabled) {
        if (dirty_state.stencil_front_ops && dirty_state.stencil_back_ops &&
            stencil_front_ops == stencil_back_ops) {
            dirty_state.stencil_front_ops = false;
            dirty_state.stencil_back_ops = false;
            cmdbuf.setStencilOp(vk::StencilFaceFlagBits::eFrontAndBack, stencil_front_ops.fail_op,
                                stencil_front_ops.pass_op, stencil_front_ops.depth_fail_op,
                                stencil_front_ops.compare_op);
        } else {
            if (dirty_state.stencil_front_ops) {
                dirty_state.stencil_front_ops = false;
                cmdbuf.setStencilOp(vk::StencilFaceFlagBits::eFront, stencil_front_ops.fail_op,
                                    stencil_front_ops.pass_op, stencil_front_ops.depth_fail_op,
                                    stencil_front_ops.compare_op);
            }
            if (dirty_state.stencil_back_ops) {
                dirty_state.stencil_back_ops = false;
                cmdbuf.setStencilOp(vk::StencilFaceFlagBits::eBack, stencil_back_ops.fail_op,
                                    stencil_back_ops.pass_op, stencil_back_ops.depth_fail_op,
                                    stencil_back_ops.compare_op);
            }
        }
        if (dirty_state.stencil_front_reference && dirty_state.stencil_back_reference &&
            stencil_front_reference == stencil_back_reference) {
            dirty_state.stencil_front_reference = false;
            dirty_state.stencil_back_reference = false;
            cmdbuf.setStencilReference(vk::StencilFaceFlagBits::eFrontAndBack,
                                       stencil_front_reference);
        } else {
            if (dirty_state.stencil_front_reference) {
                dirty_state.stencil_front_reference = false;
                cmdbuf.setStencilReference(vk::StencilFaceFlagBits::eFront,
                                           stencil_front_reference);
            }
            if (dirty_state.stencil_back_reference) {
                dirty_state.stencil_back_reference = false;
                cmdbuf.setStencilReference(vk::StencilFaceFlagBits::eBack, stencil_back_reference);
            }
        }
        if (dirty_state.stencil_front_write_mask && dirty_state.stencil_back_write_mask &&
            stencil_front_write_mask == stencil_back_write_mask) {
            dirty_state.stencil_front_write_mask = false;
            dirty_state.stencil_back_write_mask = false;
            cmdbuf.setStencilWriteMask(vk::StencilFaceFlagBits::eFrontAndBack,
                                       stencil_front_write_mask);
        } else {
            if (dirty_state.stencil_front_write_mask) {
                dirty_state.stencil_front_write_mask = false;
                cmdbuf.setStencilWriteMask(vk::StencilFaceFlagBits::eFront,
                                           stencil_front_write_mask);
            }
            if (dirty_state.stencil_back_write_mask) {
                dirty_state.stencil_back_write_mask = false;
                cmdbuf.setStencilWriteMask(vk::StencilFaceFlagBits::eBack, stencil_back_write_mask);
            }
        }
        if (dirty_state.stencil_front_compare_mask && dirty_state.stencil_back_compare_mask &&
            stencil_front_compare_mask == stencil_back_compare_mask) {
            dirty_state.stencil_front_compare_mask = false;
            dirty_state.stencil_back_compare_mask = false;
            cmdbuf.setStencilCompareMask(vk::StencilFaceFlagBits::eFrontAndBack,
                                         stencil_front_compare_mask);
        } else {
            if (dirty_state.stencil_front_compare_mask) {
                dirty_state.stencil_front_compare_mask = false;
                cmdbuf.setStencilCompareMask(vk::StencilFaceFlagBits::eFront,
                                             stencil_front_compare_mask);
            }
            if (dirty_state.stencil_back_compare_mask) {
                dirty_state.stencil_back_compare_mask = false;
                cmdbuf.setStencilCompareMask(vk::StencilFaceFlagBits::eBack,
                                             stencil_back_compare_mask);
            }
        }
    }
    if (dirty_state.primitive_restart_enable) {
        dirty_state.primitive_restart_enable = false;
        cmdbuf.setPrimitiveRestartEnable(primitive_restart_enable);
    }
    if (dirty_state.rasterizer_discard_enable) {
        dirty_state.rasterizer_discard_enable = false;
        cmdbuf.setRasterizerDiscardEnable(rasterizer_discard_enable);
    }
    if (dirty_state.cull_mode) {
        dirty_state.cull_mode = false;
        cmdbuf.setCullMode(cull_mode);
    }
    if (dirty_state.front_face) {
        dirty_state.front_face = false;
        cmdbuf.setFrontFace(front_face);
    }
    if (dirty_state.blend_constants) {
        dirty_state.blend_constants = false;
        cmdbuf.setBlendConstants(blend_constants.data());
    }
    if (dirty_state.color_write_masks) {
        dirty_state.color_write_masks = false;
        if (instance.IsDynamicColorWriteMaskSupported()) {
            cmdbuf.setColorWriteMaskEXT(0, color_write_masks);
        }
    }
    if (dirty_state.line_width) {
        dirty_state.line_width = false;
        cmdbuf.setLineWidth(line_width);
    }
    if (dirty_state.feedback_loop_enabled && instance.IsAttachmentFeedbackLoopLayoutSupported()) {
        dirty_state.feedback_loop_enabled = false;
        cmdbuf.setAttachmentFeedbackLoopEnableEXT(feedback_loop_enabled
                                                      ? vk::ImageAspectFlagBits::eColor
                                                      : vk::ImageAspectFlagBits::eNone);
    }
}

} // namespace Vulkan
