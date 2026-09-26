// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/assert.h"
#include "common/logging/formatter.h"
#include "core/emulator_settings.h"
#include "video_core/renderdoc.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <string>
#include <vector>
#include <renderdoc_app.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <filesystem>

namespace VideoCore {

/// Offline RenderDoc loading: the setting, or SHADPS4_RDOC_CAPTURE_AT for scripted captures (so harness runs
/// keep the reference configuration)
static bool WantsRenderDoc() {
    return EmulatorSettings.IsRenderdocEnabled() ||
           std::getenv("SHADPS4_RDOC_CAPTURE_AT") != nullptr;
}

enum class CaptureState {
    Idle,
    Triggered,
    InProgress,
};
static CaptureState capture_state{CaptureState::Idle};
static std::atomic<u32> screenshot_game_only_count{0};
static std::atomic<u32> screenshot_with_overlays_count{0};
// Reference for SHADPS4_RDOC_CAPTURE_AT; initialized during static init, i.e. process launch
static const auto process_start = std::chrono::steady_clock::now();

RENDERDOC_API_1_6_0* rdoc_api{};

void LoadRenderDoc() {
#ifdef WIN32

    // Check if we are running by RDoc GUI
    HMODULE mod = GetModuleHandleA("renderdoc.dll");
    if (!mod && WantsRenderDoc()) {
        // If enabled in config, try to load RDoc runtime in offline mode
        HKEY h_reg_key;
        LONG result = RegOpenKeyExW(HKEY_LOCAL_MACHINE,
                                    L"SOFTWARE\\Classes\\RenderDoc.RDCCapture.1\\DefaultIcon\\", 0,
                                    KEY_READ, &h_reg_key);
        if (result != ERROR_SUCCESS) {
            return;
        }
        std::array<wchar_t, MAX_PATH> key_str{};
        DWORD str_sz_out{key_str.size()};
        result = RegQueryValueExW(h_reg_key, L"", 0, NULL, (LPBYTE)key_str.data(), &str_sz_out);
        if (result != ERROR_SUCCESS) {
            return;
        }

        std::filesystem::path path{key_str.cbegin(), key_str.cend()};
        path = path.parent_path().append("renderdoc.dll");
        const auto path_to_lib = path.generic_string();
        mod = LoadLibraryA(path_to_lib.c_str());
    }

    if (mod) {
        const auto RENDERDOC_GetAPI =
            reinterpret_cast<pRENDERDOC_GetAPI>(GetProcAddress(mod, "RENDERDOC_GetAPI"));
        const s32 ret = RENDERDOC_GetAPI(eRENDERDOC_API_Version_1_6_0, (void**)&rdoc_api);
        ASSERT(ret == 1);
    }
#else
#ifdef ANDROID
    static constexpr const char RENDERDOC_LIB[] = "libVkLayer_GLES_RenderDoc.so";
#else
    static constexpr const char RENDERDOC_LIB[] = "librenderdoc.so";
#endif
    // Check if we are running by RDoc GUI
    void* mod = dlopen(RENDERDOC_LIB, RTLD_NOW | RTLD_NOLOAD);
    if (!mod && WantsRenderDoc()) {
        // If enabled in config, try to load RDoc runtime in offline mode
        if ((mod = dlopen(RENDERDOC_LIB, RTLD_NOW))) {
            const auto RENDERDOC_GetAPI =
                reinterpret_cast<pRENDERDOC_GetAPI>(dlsym(mod, "RENDERDOC_GetAPI"));
            const s32 ret = RENDERDOC_GetAPI(eRENDERDOC_API_Version_1_6_0, (void**)&rdoc_api);
            ASSERT(ret == 1);
        } else {
            LOG_ERROR(Render, "Cannot load RenderDoc: {}", dlerror());
        }
    }
#endif
    if (rdoc_api) {
        // Disable default capture keys as they suppose to trigger present-to-present capturing
        // and it is not what we want
        rdoc_api->SetCaptureKeys(nullptr, 0);

        // Also remove rdoc crash handler
        rdoc_api->UnloadCrashHandler();
    }
}

void StartCapture() {
    if (!rdoc_api) {
        return;
    }

    if (capture_state == CaptureState::Triggered) {
        rdoc_api->StartFrameCapture(nullptr, nullptr);
        capture_state = CaptureState::InProgress;
    }
}

void EndCapture() {
    if (!rdoc_api) {
        return;
    }

    if (capture_state == CaptureState::InProgress) {
        const u32 ok = rdoc_api->EndFrameCapture(nullptr, nullptr);
        capture_state = CaptureState::Idle;
        // Report the file so scripts can collect it
        const u32 count = rdoc_api->GetNumCaptures();
        // GetCapture writes the whole path, so ask for its length (with the terminator) first
        u32 length = 0;
        std::string path;
        if (ok && count > 0 && rdoc_api->GetCapture(count - 1, nullptr, &length, nullptr) == 1 &&
            length != 0) {
            path.resize(length);
        }
        if (!path.empty() && rdoc_api->GetCapture(count - 1, path.data(), &length, nullptr) == 1) {
            path.erase(path.find_last_not_of('\0') + 1);
            LOG_WARNING(Render, "RenderDoc capture written: {}", path);
        } else {
            LOG_WARNING(Render, "RenderDoc capture ended (ok={}, captures={})", ok, count);
        }
    }
}

void PollScheduledCaptures() {
    // SHADPS4_RDOC_CAPTURE_AT=<sec>[,<sec>...]: trigger a capture of the next guest frame once each time mark
    // (seconds since launch) passes. Loads RenderDoc offline when it is installed
    static std::vector<double> marks = [] {
        std::vector<double> values;
        if (const char* env = std::getenv("SHADPS4_RDOC_CAPTURE_AT")) {
            const std::string text{env};
            size_t start = 0;
            while (start < text.size()) {
                const size_t end = text.find(',', start);
                const std::string item = text.substr(start, end - start);
                if (!item.empty()) {
                    values.push_back(std::strtod(item.c_str(), nullptr));
                }
                if (end == std::string::npos) {
                    break;
                }
                start = end + 1;
            }
            std::sort(values.begin(), values.end(), std::greater<>{});
        }
        return values;
    }();
    if (marks.empty() || !rdoc_api || capture_state != CaptureState::Idle) {
        return;
    }
    const double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - process_start).count();
    if (elapsed < marks.back()) {
        return;
    }
    // Skip marks that passed while a capture was running; one capture per poll
    while (!marks.empty() && marks.back() <= elapsed) {
        marks.pop_back();
    }
    LOG_WARNING(Render, "RenderDoc scheduled capture triggered at {:.1f}s", elapsed);
    TriggerCapture();
}

void TriggerCapture() {
    if (capture_state == CaptureState::Idle) {
        capture_state = CaptureState::Triggered;
    }
}

void SetOutputDir(const std::filesystem::path& path, const std::string& prefix) {
    if (!rdoc_api) {
        return;
    }
    LOG_WARNING(Common, "RenderDoc capture path: {}", (path / prefix).string());
    rdoc_api->SetCaptureFilePathTemplate(fmt::UTF((path / prefix).u8string()).data.data());
}

bool IsRenderDocLoaded() {
    return rdoc_api != nullptr;
}

void RequestScreenshot(const ScreenshotRequest request) {
    switch (request) {
    case ScreenshotRequest::GameOnly:
        screenshot_game_only_count.fetch_add(1, std::memory_order_relaxed);
        break;
    case ScreenshotRequest::WithOverlays:
        screenshot_with_overlays_count.fetch_add(1, std::memory_order_relaxed);
        break;
    case ScreenshotRequest::None:
    default:
        break;
    }
}

u32 ConsumeGameOnlyScreenshotRequests() {
    return screenshot_game_only_count.exchange(0, std::memory_order_acq_rel);
}

u32 ConsumeWithOverlaysScreenshotRequests() {
    return screenshot_with_overlays_count.exchange(0, std::memory_order_acq_rel);
}

ScreenshotRequests ConsumeScreenshotRequests() {
    return ScreenshotRequests{
        .game_only_count = ConsumeGameOnlyScreenshotRequests(),
        .with_overlays_count = ConsumeWithOverlaysScreenshotRequests(),
    };
}

} // namespace VideoCore
