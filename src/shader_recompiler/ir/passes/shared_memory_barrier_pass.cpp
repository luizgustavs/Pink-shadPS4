// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <queue>
#include <unordered_set>
#include "common/logging/log.h"
#include "core/emulator_settings.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/ir/program.h"
#include "shader_recompiler/profile.h"

namespace Shader::Optimization {

static bool IsLoadShared(const IR::Inst& inst) {
    return inst.GetOpcode() == IR::Opcode::LoadSharedU16 ||
           inst.GetOpcode() == IR::Opcode::LoadSharedU32 ||
           inst.GetOpcode() == IR::Opcode::LoadSharedU64;
}

static bool IsWriteShared(const IR::Inst& inst) {
    const IR::Opcode opcode = inst.GetOpcode();
    if (opcode >= IR::Opcode::SharedAtomicIAdd32 && opcode <= IR::Opcode::SharedAtomicCmpSwap64) {
        return !inst.Flags<bool>();
    }
    return opcode == IR::Opcode::WriteSharedU16 || opcode == IR::Opcode::WriteSharedU32 ||
           opcode == IR::Opcode::WriteSharedU64;
}

// Inserts barriers when a shared memory write and read occur in the same basic block.
static void EmitBarrierInBlock(IR::Block* block) {
    enum class BarrierAction : u32 {
        None,
        BarrierOnWrite,
        BarrierOnRead,
    };
    BarrierAction action{};
    for (IR::Inst& inst : block->Instructions()) {
        if (IsLoadShared(inst)) {
            if (action == BarrierAction::BarrierOnRead) {
                IR::IREmitter ir{*block, IR::Block::InstructionList::s_iterator_to(inst)};
                ir.Barrier();
            }
            action = BarrierAction::BarrierOnWrite;
            continue;
        }
        if (IsWriteShared(inst)) {
            if (action == BarrierAction::BarrierOnWrite) {
                IR::IREmitter ir{*block, IR::Block::InstructionList::s_iterator_to(inst)};
                ir.Barrier();
            }
            action = BarrierAction::BarrierOnRead;
        }
    }
    if (action != BarrierAction::None) {
        IR::IREmitter ir{*block, --block->end()};
        ir.Barrier();
    }
}

using NodeSet = std::unordered_set<const IR::Block*>;

static void EmitBarrierAtBlockStart(IR::Block* block) {
    auto insert_point = std::ranges::find_if_not(block->Instructions(), IR::IsPhi);
    IR::IREmitter ir{*block, insert_point};
    ir.Barrier();
}

struct DivergenceContext {
    // Per-game lds_barrier_uniform_readlane: blocks whose ReadLane with a constant lane the wave64 lowering
    // turns into a workgroup-uniform value (an exchange through LDS)
    std::unordered_set<const IR::Block*> uniform_readlane_blocks;
    // Set when treating such a ReadLane as uniform changed a decision of this pass
    bool changed_decision{};
};

// A condition is divergent when it depends on LocalInvocationId. With uniform_readlane_blocks, the search
// does not continue through uniform ReadLanes: a wave-wide reduction (a tile min/max depth, for example) is
// the same in every invocation even though its inputs are not
static bool SearchDivergence(const IR::U1& cond,
                             const std::unordered_set<const IR::Block*>* uniform_readlane_blocks) {
    if (cond.IsImmediate()) {
        return false;
    }
    std::unordered_set<const IR::Inst*> visited{cond.Inst()};
    std::queue<const IR::Inst*> queue;
    queue.push(cond.Inst());
    while (!queue.empty()) {
        const IR::Inst* inst = queue.front();
        queue.pop();
        if (inst->GetOpcode() == IR::Opcode::GetAttributeU32 &&
            inst->Arg(0).Attribute() == IR::Attribute::LocalInvocationId) {
            return true;
        }
        if (uniform_readlane_blocks && inst->GetOpcode() == IR::Opcode::ReadLane &&
            inst->Arg(1).IsImmediate() && uniform_readlane_blocks->contains(inst->GetParent())) {
            continue;
        }
        for (size_t arg = inst->NumArgs(); arg--;) {
            const IR::Value value = inst->Arg(arg);
            if (value.IsImmediate()) {
                continue;
            }
            const IR::Inst* arg_inst = value.Inst();
            if (visited.insert(arg_inst).second) {
                queue.push(arg_inst);
            }
        }
    }
    return false;
}

static bool IsDivergent(const IR::U1& cond, DivergenceContext& ctx) {
    if (!SearchDivergence(cond, nullptr)) {
        return false;
    }
    if (ctx.uniform_readlane_blocks.empty() ||
        SearchDivergence(cond, &ctx.uniform_readlane_blocks)) {
        return true;
    }
    ctx.changed_decision = true;
    return false;
}

// Inserts a barrier after divergent conditional blocks to avoid undefined
// behavior when some threads write and others read from shared memory.
static void EmitBarrierInMergeBlock(const IR::AbstractSyntaxNode::Data& data,
                                    NodeSet& divergence_end, u32& divergence_depth,
                                    DivergenceContext& ctx) {
    const IR::U1 cond = data.if_node.cond;
    if (IsDivergent(cond, ctx)) {
        if (divergence_depth == 0) {
            EmitBarrierAtBlockStart(data.if_node.merge);
        }
        ++divergence_depth;
        divergence_end.emplace(data.if_node.merge);
    }
}

// A barrier inside a loop is invalid when different invocations leave on different iterations.
// Mark such loops so their shared-memory synchronization can be deferred to the merge block.
static NodeSet FindDivergentLoops(const IR::AbstractSyntaxList& syntax_list,
                                  DivergenceContext& ctx) {
    NodeSet divergent_loops;
    for (const IR::AbstractSyntaxNode& node : syntax_list) {
        switch (node.type) {
        case IR::AbstractSyntaxNode::Type::Repeat:
            if (IsDivergent(node.data.repeat.cond, ctx)) {
                divergent_loops.emplace(node.data.repeat.merge);
            }
            break;
        case IR::AbstractSyntaxNode::Type::Break:
            if (IsDivergent(node.data.break_node.cond, ctx)) {
                divergent_loops.emplace(node.data.break_node.merge);
            }
            break;
        default:
            break;
        }
    }
    return divergent_loops;
}

static constexpr u32 GcnSubgroupSize = 64;

void SharedMemoryBarrierPass(IR::Program& program, const RuntimeInfo& runtime_info,
                             const Profile& profile) {
    if (program.info.hw_stage != HwStage::Compute) {
        return;
    }
    const auto& cs_info = runtime_info.hw.cs;
    const u32 shared_memory_size = cs_info.shared_memory_size;
    const u32 threadgroup_size =
        cs_info.workgroup_size[0] * cs_info.workgroup_size[1] * cs_info.workgroup_size[2];
    // The compiler can skip barriers when the workgroup matches the hardware subgroup size
    // Larger groups may still have a single-wave LDS reduction tail, such as if (tid < 32) followed
    // by if (tid < 16), that assumes 64 lanes run together
    // Host subgroups narrower than 64 lanes do not guarantee that, which can break SotC auto-
    // exposure and make the sun too bright on NVIDIA
    // With lds_barriers_large_groups, add barriers at divergent merge blocks that touch LDS and
    // keep the guest barriers for the other waves
    // Read this setting like lds_barrier_uniform_readlane because it is included in the shader
    // codegen key
    // The setting changes compiled compute code, so cached versions need to keep the value that
    // produced their barriers
    const bool large_group =
        threadgroup_size > GcnSubgroupSize && EmulatorSettings.IsLdsBarriersLargeGroups();
    if (shared_memory_size == 0 || (threadgroup_size != GcnSubgroupSize && !large_group) ||
        !profile.needs_lds_barriers) {
        return;
    }
    using Type = IR::AbstractSyntaxNode::Type;
    DivergenceContext ctx;
    // Read from the settings rather than the Profile, which is stored in the pipeline cache; the setting is
    // part of the cache's codegen key instead
    if (EmulatorSettings.IsLdsBarrierUniformReadlane()) {
        const auto blocks = FindWave64UniformBlocks(program);
        ctx.uniform_readlane_blocks.insert(blocks.begin(), blocks.end());
    }
    u32 divergence_depth{};
    NodeSet divergence_end;
    const NodeSet divergent_loops = FindDivergentLoops(program.syntax_list, ctx);
    for (const IR::AbstractSyntaxNode& node : program.syntax_list) {
        if (node.type == Type::EndIf) {
            if (divergence_end.contains(node.data.end_if.merge)) {
                --divergence_depth;
            }
            continue;
        }
        // Check if branch depth is zero, we don't want to insert barrier in potentially divergent
        // code.
        if (node.type == Type::If) {
            EmitBarrierInMergeBlock(node.data, divergence_end, divergence_depth, ctx);
            continue;
        }
        if (node.type == Type::Loop && divergent_loops.contains(node.data.loop.merge)) {
            ++divergence_depth;
            continue;
        }
        if (node.type == Type::Repeat && divergent_loops.contains(node.data.repeat.merge)) {
            ASSERT(divergence_depth > 0);
            --divergence_depth;
            if (divergence_depth == 0) {
                EmitBarrierAtBlockStart(node.data.repeat.merge);
            }
            continue;
        }
        if (node.type == Type::Block && divergence_depth == 0 && !large_group) {
            EmitBarrierInBlock(node.data.block);
        }
    }
    if (large_group) {
        // Log this compile-time workaround once for each affected shader and include the running
        // count so its use is easy to track
        static std::atomic<u32> large_group_shaders{0};
        LOG_WARNING(Render_Recompiler,
                    "Workaround lds_barriers_large_groups: shader {:#x} ({} threads) gets LDS "
                    "barriers after divergent branches (affected shaders: {})",
                    program.info.pgm_hash, threadgroup_size, large_group_shaders.fetch_add(1) + 1);
    }
    if (ctx.changed_decision) {
        // Log this compile-time workaround once for each affected shader and include the running
        // count so its use is easy to track
        static std::atomic<u32> affected_shaders{0};
        LOG_WARNING(Render_Recompiler,
                    "Workaround lds_barrier_uniform_readlane: shader {:#x} keeps LDS barriers "
                    "inside a wave-uniform branch (affected shaders: {})",
                    program.info.pgm_hash, affected_shaders.fetch_add(1) + 1);
    }
}

} // namespace Shader::Optimization
