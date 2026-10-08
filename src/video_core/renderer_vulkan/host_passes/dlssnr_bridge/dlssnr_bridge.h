// SPDX-FileCopyrightText: Copyright 2026 Daniil Sokolyuk (video2dlssnr)
// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: MIT

// C interface of nvngx.dll_dlssnr.dll, the bridge between the emulator and DLSS Neural Rendering
// (DLSS 5, NGX feature 18). Every call into NVIDIA code goes through the bridge: the model
// (nvngx_dlssnr.dll) only accepts callers whose module path contains "nvngx.dll", and the bridge
// catches faults inside NVIDIA code instead of taking the emulator down

#pragma once

struct ID3D12Device;
struct ID3D12GraphicsCommandList;
struct ID3D12Resource;

#define DLSSNR_BRIDGE_DLL_NAME L"nvngx.dll_dlssnr.dll"
#define DLSSNR_MODEL_DLL_NAME L"nvngx_dlssnr.dll"

extern "C" {

// The model's own parameters, latched when the feature is created
struct DlssNrModelParams {
    unsigned preset;       // DLSSNR.Hint.Render.Preset: 0 Default, 1..3 Preset 1..3
    float intensity;       // DLSSNR.Intensity, 0..2
    unsigned style;        // DLSSNR.Style: 0 Default, 1 Natural, 2 Cinematic
    float local_structure; // DLSSNR.LocalStructureStrength, 0..2
    float local_tone;      // DLSSNR.LocalToneStrength, 0..2
    float skin_structure;  // DLSSNR.SkinStructureStrength, below 0 keeps the model default
    unsigned auto_mask;    // DLSSNR.UseAutoMask
    unsigned ui_correction; // DLSSNR.UICorrection
};

struct DlssNrBridgeInitInfo {
    ID3D12Device* device;
    const wchar_t* model_path; // absolute path of nvngx_dlssnr.dll
    const wchar_t* data_path;  // folder for NGX logs
    int arch_spoof;            // report RTX 20/30/40 cards to the model as RTX 50
};

// NGX results: 1 is success, failures have the 0xBAD00000 bits set
#define DLSSNR_BRIDGE_SUCCESS 1
// A structured exception was caught inside NVIDIA code; the bridge refuses further calls
#define DLSSNR_BRIDGE_FAULTED ((int)0xBAD000FF)

// Loads the driver's NGX core and the model and initializes both on the device
typedef int (*PFN_DlssNrBridgeInit)(const DlssNrBridgeInitInfo* info);
// Records feature creation on cmd for a width x height frame; nullptr on failure
typedef void* (*PFN_DlssNrBridgeCreate)(ID3D12GraphicsCommandList* cmd, unsigned width,
                                        unsigned height, const DlssNrModelParams* params);
// Records an evaluation on cmd. color is read as a shader resource, output is written as UAV
typedef int (*PFN_DlssNrBridgeEvaluate)(ID3D12GraphicsCommandList* cmd, void* feature,
                                        ID3D12Resource* color, ID3D12Resource* depth,
                                        ID3D12Resource* motion, ID3D12Resource* output,
                                        unsigned width, unsigned height, int reset);
typedef void (*PFN_DlssNrBridgeRelease)(void* feature);
// Releases the parameter block and shuts the NGX core down for the device
typedef void (*PFN_DlssNrBridgeShutdown)();
// Copies the messages logged since the last call into buffer and clears them
typedef unsigned (*PFN_DlssNrBridgeReadLog)(char* buffer, unsigned size);

} // extern "C"
