// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <memory>
#include <string>

#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {
class Instance;
class Scheduler;
struct SubmitInfo;
} // namespace Vulkan

namespace Vulkan::HostPasses {

/// Run DLSS Neural Rendering over the final game frame without motion vectors or depth
/// Share the frame with D3D12 on the same GPU and copy the result back before drawing overlays
/// Require Windows, NVIDIA RTX and nvngx_dlssnr.dll next to shadps4.exe
class DlssNrPass {
public:
    struct Settings {
        u32 style{0};
        float intensity{1.0f};
        bool ui_correction{true};

        bool operator==(const Settings&) const = default;
    };

    DlssNrPass();
    ~DlssNrPass();

    void Create(const Instance& instance, Scheduler& scheduler);

    /// Process a width x height SDR frame and leave it in the general layout
    /// Include the wait added to info in the final submit when this returns true
    bool Render(vk::Image image, vk::Format format, u32 width, u32 height, bool hdr,
                const Settings& settings, SubmitInfo& info);

    /// Release shared images and the model with the GPU idle
    void Destroy();

    static void SetEnabled(bool enabled);
    static void Toggle();
    static bool IsEnabled();
    /// Report the current state or failure reason for the overlay
    static std::string StatusText();

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace Vulkan::HostPasses
