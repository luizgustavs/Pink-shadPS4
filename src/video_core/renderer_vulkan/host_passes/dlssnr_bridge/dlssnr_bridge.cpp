// SPDX-FileCopyrightText: Copyright 2026 Daniil Sokolyuk (video2dlssnr)
// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: MIT

// Run DLSS Neural Rendering (NGX feature 18) through nvngx.dll_dlssnr.dll
// Port the forwarder and architecture redirect from video2dlssnr
// https://github.com/DaniilSokolyuk/video2dlssnr
// This bridge contains no NVIDIA code
//
// Keep NVIDIA calls in this module because the model requires "nvngx.dll" in the caller's path
// Use the driver's NGX core for initialization and parameters, then call the model directly
// Creating feature 18 through driver 616.64 or newer can fault inside D3D12
// Report Blackwell only to nvngx_dlssnr.dll when arch_spoof is enabled
// Preserve the real architecture for other callers, including DLSS Super Resolution
// Guard NVIDIA entry points with SEH and refuse further calls after a fault

#include <windows.h>

#include <intrin.h>

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>

#include "dlssnr_bridge.h"

namespace {

// NVSDK_NGX_Parameter as the NGX runtime lays it out. Only the virtual table matters; the
// declaration order below is the order of the NGX SDK, which fixes the MSVC table layout
struct NgxParameter {
    virtual void Set(const char* name, unsigned long long value) = 0;
    virtual void Set(const char* name, float value) = 0;
    virtual void Set(const char* name, double value) = 0;
    virtual void Set(const char* name, unsigned int value) = 0;
    virtual void Set(const char* name, int value) = 0;
    virtual void Set(const char* name, struct ID3D11Resource* value) = 0;
    virtual void Set(const char* name, ID3D12Resource* value) = 0;
    virtual void Set(const char* name, void* value) = 0;

    virtual int Get(const char* name, unsigned long long* value) const = 0;
    virtual int Get(const char* name, float* value) const = 0;
    virtual int Get(const char* name, double* value) const = 0;
    virtual int Get(const char* name, unsigned int* value) const = 0;
    virtual int Get(const char* name, int* value) const = 0;
    virtual int Get(const char* name, struct ID3D11Resource** value) const = 0;
    virtual int Get(const char* name, ID3D12Resource** value) const = 0;
    virtual int Get(const char* name, void** value) const = 0;

    virtual void Reset() = 0;
};

constexpr int kNgxSuccess = 1;
constexpr int kNgxFail = static_cast<int>(0xBAD00000u);
constexpr int kNgxVersionApi = 0x15;
// Generic application id, the one video2dlssnr uses
constexpr unsigned long long kApplicationId = 0x24480451ull;
constexpr int kFeatureNeuralRendering = 18;

// Log

std::mutex g_log_lock;
std::string g_log;

void Log(const char* format, ...) {
    char line[1024];
    va_list args;
    va_start(args, format);
    vsnprintf(line, sizeof(line), format, args);
    va_end(args);
    std::scoped_lock lock{g_log_lock};
    g_log += line;
    g_log += '\n';
}

std::string Narrow(const wchar_t* text) {
    if (!text) {
        return {};
    }
    const int size = WideCharToMultiByte(CP_UTF8, 0, text, -1, nullptr, 0, nullptr, nullptr);
    if (size <= 1) {
        return {};
    }
    std::string out(static_cast<size_t>(size - 1), '\0');
    WideCharToMultiByte(CP_UTF8, 0, text, -1, out.data(), size, nullptr, nullptr);
    return out;
}

// Keep C++ objects with destructors out of __try guards
// Use a volatile result to prevent tail calls so the return address stays in this module

using PfnCoreInit = int(__cdecl*)(unsigned long long, const wchar_t*, ID3D12Device*, int);
using PfnCoreAllocateParameters = int(__cdecl*)(NgxParameter**);
using PfnCoreDestroyParameters = int(__cdecl*)(NgxParameter*);
using PfnCoreShutdown1 = int(__cdecl*)(ID3D12Device*);
using PfnModelInitExt = int(__cdecl*)(unsigned long long, const wchar_t*, ID3D12Device*, int,
                                      const void*);
using PfnModelCreate = int(__cdecl*)(ID3D12GraphicsCommandList*, int, const void*, void**);
using PfnModelEvaluate = int(__cdecl*)(ID3D12GraphicsCommandList*, const void*, const void*,
                                       void*);
using PfnModelRelease = int(__cdecl*)(void*);

struct Fault {
    const char* stage;
    unsigned code;
    unsigned long long address;
    unsigned long long module_base;
    wchar_t module[MAX_PATH];
};
Fault g_fault{};

int RecordFault(EXCEPTION_POINTERS* pointers, const char* stage) {
    g_fault.stage = stage;
    g_fault.code = 0;
    g_fault.address = 0;
    g_fault.module_base = 0;
    g_fault.module[0] = 0;
    if (pointers && pointers->ExceptionRecord) {
        g_fault.code = pointers->ExceptionRecord->ExceptionCode;
        g_fault.address =
            reinterpret_cast<unsigned long long>(pointers->ExceptionRecord->ExceptionAddress);
        HMODULE module = nullptr;
        if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                   GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                               reinterpret_cast<LPCWSTR>(
                                   pointers->ExceptionRecord->ExceptionAddress),
                               &module) &&
            module) {
            g_fault.module_base = reinterpret_cast<unsigned long long>(module);
            GetModuleFileNameW(module, g_fault.module, MAX_PATH);
        }
    }
    return EXCEPTION_EXECUTE_HANDLER;
}

int GuardedCoreInit(PfnCoreInit fn, const wchar_t* data_path, ID3D12Device* device) {
    __try {
        volatile int result = fn(kApplicationId, data_path, device, kNgxVersionApi);
        return result;
    } __except (RecordFault(GetExceptionInformation(), "NGX core Init")) {
        return DLSSNR_BRIDGE_FAULTED;
    }
}

int GuardedCoreAllocate(PfnCoreAllocateParameters fn, NgxParameter** out) {
    __try {
        volatile int result = fn(out);
        return result;
    } __except (RecordFault(GetExceptionInformation(), "NGX core AllocateParameters")) {
        return DLSSNR_BRIDGE_FAULTED;
    }
}

int GuardedCoreDestroy(PfnCoreDestroyParameters fn, NgxParameter* params) {
    __try {
        volatile int result = fn(params);
        return result;
    } __except (RecordFault(GetExceptionInformation(), "NGX core DestroyParameters")) {
        return DLSSNR_BRIDGE_FAULTED;
    }
}

int GuardedCoreShutdown(PfnCoreShutdown1 fn, ID3D12Device* device) {
    __try {
        volatile int result = fn(device);
        return result;
    } __except (RecordFault(GetExceptionInformation(), "NGX core Shutdown1")) {
        return DLSSNR_BRIDGE_FAULTED;
    }
}

int GuardedModelInit(PfnModelInitExt fn, const wchar_t* data_path, ID3D12Device* device,
                     const void* params) {
    __try {
        volatile int result = fn(kApplicationId, data_path, device, kNgxVersionApi, params);
        return result;
    } __except (RecordFault(GetExceptionInformation(), "model Init_Ext")) {
        return DLSSNR_BRIDGE_FAULTED;
    }
}

int GuardedModelCreate(PfnModelCreate fn, ID3D12GraphicsCommandList* cmd, const void* params,
                       void** handle) {
    __try {
        volatile int result = fn(cmd, kFeatureNeuralRendering, params, handle);
        return result;
    } __except (RecordFault(GetExceptionInformation(), "model CreateFeature")) {
        return DLSSNR_BRIDGE_FAULTED;
    }
}

int GuardedModelEvaluate(PfnModelEvaluate fn, ID3D12GraphicsCommandList* cmd,
                         const void* feature, const void* params) {
    __try {
        volatile int result = fn(cmd, feature, params, nullptr);
        return result;
    } __except (RecordFault(GetExceptionInformation(), "model EvaluateFeature")) {
        return DLSSNR_BRIDGE_FAULTED;
    }
}

int GuardedModelRelease(PfnModelRelease fn, void* feature) {
    __try {
        volatile int result = fn(feature);
        return result;
    } __except (RecordFault(GetExceptionInformation(), "model ReleaseFeature")) {
        return DLSSNR_BRIDGE_FAULTED;
    }
}

int GuardedParamsReset(NgxParameter* params) {
    __try {
        params->Reset();
        return kNgxSuccess;
    } __except (RecordFault(GetExceptionInformation(), "parameter block Reset")) {
        return DLSSNR_BRIDGE_FAULTED;
    }
}

void LogFault() {
    Log("DLSS NR: %s faulted with exception 0x%08X at 0x%llX (%s+0x%llX); DLSS NR stays off "
        "for this session",
        g_fault.stage ? g_fault.stage : "?", g_fault.code, g_fault.address,
        g_fault.module[0] ? Narrow(g_fault.module).c_str() : "?",
        g_fault.module_base ? g_fault.address - g_fault.module_base : 0ull);
}

// GPU architecture redirect

namespace ArchSpoof {

// nvapi_QueryInterface ids
constexpr unsigned kNvapiInitialize = 0x0150E828u;
constexpr unsigned kNvapiEnumPhysicalGpus = 0xE5AC921Fu;
constexpr unsigned kNvapiGpuGetArchInfo = 0xD8265D24u;

// NV_GPU_ARCHITECTURE_ID groups; the low nibble carries the implementation
constexpr unsigned kArchTuring = 0x160u;
constexpr unsigned kArchAmpere = 0x170u;
constexpr unsigned kArchAda = 0x190u;
// What the model accepts: GB2XX. GB1XX (0x1A0) is still refused
constexpr unsigned kArchBlackwell = 0x1B0u;

struct NvArchInfo {
    unsigned version;
    unsigned architecture;
    unsigned implementation;
    unsigned revision;
};

using PfnQueryInterface = void*(__cdecl*)(unsigned);
using PfnInitialize = int(__cdecl*)();
using PfnEnumPhysicalGpus = int(__cdecl*)(void**, unsigned*);
using PfnGetArchInfo = int(__cdecl*)(void*, NvArchInfo*);

constexpr int kMaxGpus = 64;
void* g_handles[kMaxGpus];
NvArchInfo g_cache[kMaxGpus];
int g_count = 0;
bool g_done = false;

unsigned Group(unsigned arch) {
    return arch & 0xFFFFFFF0u;
}

bool NeedsSpoof(unsigned arch) {
    const unsigned group = Group(arch);
    return group == kArchTuring || group == kArchAmpere || group == kArchAda;
}

const char* Name(unsigned arch) {
    switch (Group(arch)) {
    case kArchTuring:
        return "Turing";
    case kArchAmpere:
        return "Ampere";
    case 0x180u:
        return "Hopper";
    case kArchAda:
        return "Ada";
    case 0x1A0u:
    case kArchBlackwell:
        return "Blackwell";
    default:
        return "unknown";
    }
}

struct CallerVerdict {
    HMODULE module;
    bool spoof;
};
constexpr int kMaxCallers = 16;
CallerVerdict g_callers[kMaxCallers];
int g_caller_count = 0;
std::mutex g_caller_lock;

// Report Blackwell only to nvngx_dlssnr.dll
// DLSS Super Resolution can crash on Ampere if it receives the redirected architecture
bool CallerIsModel(void* return_address) {
    HMODULE module = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            static_cast<LPCWSTR>(return_address), &module) ||
        !module) {
        return false;
    }
    std::scoped_lock lock{g_caller_lock};
    for (int i = 0; i < g_caller_count; ++i) {
        if (g_callers[i].module == module) {
            return g_callers[i].spoof;
        }
    }
    wchar_t path[MAX_PATH]{};
    GetModuleFileNameW(module, path, MAX_PATH);
    const wchar_t* base = wcsrchr(path, L'\\');
    base = base ? base + 1 : path;
    const bool spoof = _wcsicmp(base, DLSSNR_MODEL_DLL_NAME) == 0;
    if (g_caller_count < kMaxCallers) {
        g_callers[g_caller_count++] = {module, spoof};
    }
    return spoof;
}

// Replaces NvAPI_GPU_GetArchInfo for the whole process and answers from the cache taken before
// the redirect went in. An unknown handle gets the first card's answer
int __cdecl ArchInfoHook(void* gpu, NvArchInfo* info) {
    void* const caller = _ReturnAddress();
    if (!info) {
        return -1; // NVAPI_ERROR
    }
    const unsigned version = info->version;
    int index = 0;
    for (int i = 0; i < g_count; ++i) {
        if (g_handles[i] == gpu) {
            index = i;
            break;
        }
    }
    *info = g_cache[index];
    info->version = version;
    if (NeedsSpoof(info->architecture) && CallerIsModel(caller)) {
        // Implementation and revision as a GB2XX board reports them
        info->architecture = kArchBlackwell;
        info->implementation = 0x3u;
        info->revision = 0xA1u;
    }
    return 0; // NVAPI_OK
}

bool Redirect(void* target, void* to) {
    // mov rax, imm64 ; jmp rax
    unsigned char code[12] = {0x48, 0xB8};
    memcpy(code + 2, &to, sizeof(to));
    code[10] = 0xFF;
    code[11] = 0xE0;
    DWORD old_protect = 0;
    if (!VirtualProtect(target, sizeof(code), PAGE_EXECUTE_READWRITE, &old_protect)) {
        return false;
    }
    memcpy(target, code, sizeof(code));
    DWORD unused = 0;
    VirtualProtect(target, sizeof(code), old_protect, &unused);
    FlushInstructionCache(GetCurrentProcess(), target, sizeof(code));
    return true;
}

void Setup() {
    if (g_done) {
        return;
    }
    g_done = true;
    const auto fail = [](const char* why) {
        Log("DLSS NR: GPU architecture redirect not installed: %s; the model refuses GPUs older "
            "than RTX 50",
            why);
    };

    HMODULE nvapi = LoadLibraryW(L"nvapi64.dll");
    if (!nvapi) {
        return fail("nvapi64.dll did not load");
    }
    const auto query =
        reinterpret_cast<PfnQueryInterface>(GetProcAddress(nvapi, "nvapi_QueryInterface"));
    if (!query) {
        return fail("no nvapi_QueryInterface");
    }
    const auto initialize = reinterpret_cast<PfnInitialize>(query(kNvapiInitialize));
    const auto enum_gpus = reinterpret_cast<PfnEnumPhysicalGpus>(query(kNvapiEnumPhysicalGpus));
    const auto get_arch = reinterpret_cast<PfnGetArchInfo>(query(kNvapiGpuGetArchInfo));
    if (!initialize || !enum_gpus || !get_arch) {
        return fail("the NvAPI entry points did not resolve");
    }
    if (initialize() != 0) {
        return fail("NvAPI_Initialize failed");
    }

    void* handles[kMaxGpus] = {};
    unsigned count = 0;
    if (enum_gpus(handles, &count) != 0 || count == 0) {
        return fail("no NVIDIA GPU enumerated");
    }
    if (count > static_cast<unsigned>(kMaxGpus)) {
        count = kMaxGpus;
    }

    // Every card's real answer, taken while the real function is still in place
    g_count = 0;
    for (unsigned i = 0; i < count; ++i) {
        NvArchInfo info{};
        info.version = static_cast<unsigned>(sizeof(NvArchInfo)) | (2u << 16);
        if (get_arch(handles[i], &info) != 0) {
            info.version = static_cast<unsigned>(sizeof(NvArchInfo)) | (1u << 16);
            if (get_arch(handles[i], &info) != 0) {
                continue;
            }
        }
        g_handles[g_count] = handles[i];
        g_cache[g_count] = info;
        ++g_count;
    }
    if (g_count == 0) {
        return fail("GetArchInfo answered for no GPU");
    }

    bool any_old = false;
    for (int i = 0; i < g_count; ++i) {
        any_old = any_old || NeedsSpoof(g_cache[i].architecture);
    }
    if (!any_old) {
        Log("DLSS NR: GPU architecture 0x%X (%s) is accepted by the model as is",
            g_cache[0].architecture, Name(g_cache[0].architecture));
        return;
    }
    if (!Redirect(reinterpret_cast<void*>(get_arch), reinterpret_cast<void*>(&ArchInfoHook))) {
        return fail("could not make the NvAPI entry writable");
    }
    Log("DLSS NR: GPU architecture 0x%X (%s) is reported to the model as Blackwell",
        g_cache[0].architecture, Name(g_cache[0].architecture));
}

} // namespace ArchSpoof

// State

struct Bridge {
    HMODULE core = nullptr;
    PfnCoreInit core_init = nullptr;
    PfnCoreAllocateParameters core_allocate = nullptr;
    PfnCoreDestroyParameters core_destroy = nullptr;
    PfnCoreShutdown1 core_shutdown = nullptr;

    HMODULE model = nullptr;
    PfnModelInitExt model_init = nullptr;
    PfnModelCreate model_create = nullptr;
    PfnModelEvaluate model_evaluate = nullptr;
    PfnModelRelease model_release = nullptr;

    ID3D12Device* device = nullptr;
    NgxParameter* params = nullptr;
    bool core_initialized = false;
    bool initialized = false;
    bool faulted = false;
};
Bridge g;

// The driver's NGX core: _nvngx.dll in the folder the display driver registers
std::wstring CorePath() {
    wchar_t folder[MAX_PATH]{};
    DWORD size = sizeof(folder);
    if (RegGetValueW(HKEY_LOCAL_MACHINE, L"SYSTEM\\CurrentControlSet\\Services\\nvlddmkm\\NGXCore",
                     L"NGXPath", RRF_RT_REG_SZ, nullptr, folder, &size) != ERROR_SUCCESS) {
        return {};
    }
    return std::wstring(folder) + L"\\_nvngx.dll";
}

int Fail(int result, const char* what) {
    if (result == DLSSNR_BRIDGE_FAULTED) {
        g.faulted = true;
        LogFault();
    } else {
        Log("DLSS NR: %s failed (0x%08X)", what, static_cast<unsigned>(result));
    }
    return result == kNgxSuccess ? kNgxFail : result;
}

int Init(const DlssNrBridgeInitInfo* info) {
    if (g.faulted) {
        return DLSSNR_BRIDGE_FAULTED;
    }
    if (g.initialized) {
        return kNgxSuccess;
    }
    if (!info || !info->device || !info->model_path || !info->data_path) {
        return kNgxFail;
    }
    if (info->arch_spoof) {
        ArchSpoof::Setup();
    }

    if (!g.core) {
        const std::wstring core_path = CorePath();
        if (core_path.empty()) {
            Log("DLSS NR: the NVIDIA driver registers no NGX core (NGXCore\\NGXPath); DLSS NR "
                "needs an NVIDIA RTX GPU");
            return kNgxFail;
        }
        g.core = LoadLibraryExW(core_path.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
        if (!g.core) {
            Log("DLSS NR: could not load %s (error %lu)", Narrow(core_path.c_str()).c_str(),
                GetLastError());
            return kNgxFail;
        }
        g.core_init =
            reinterpret_cast<PfnCoreInit>(GetProcAddress(g.core, "NVSDK_NGX_D3D12_Init"));
        g.core_allocate = reinterpret_cast<PfnCoreAllocateParameters>(
            GetProcAddress(g.core, "NVSDK_NGX_D3D12_AllocateParameters"));
        g.core_destroy = reinterpret_cast<PfnCoreDestroyParameters>(
            GetProcAddress(g.core, "NVSDK_NGX_D3D12_DestroyParameters"));
        g.core_shutdown = reinterpret_cast<PfnCoreShutdown1>(
            GetProcAddress(g.core, "NVSDK_NGX_D3D12_Shutdown1"));
        if (!g.core_init || !g.core_allocate) {
            Log("DLSS NR: the NGX core exports no D3D12 Init / AllocateParameters");
            return kNgxFail;
        }
    }

    if (!g.model) {
        g.model = LoadLibraryExW(info->model_path, nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
        if (!g.model) {
            Log("DLSS NR: could not load %s (error %lu)", Narrow(info->model_path).c_str(),
                GetLastError());
            return kNgxFail;
        }
        g.model_init = reinterpret_cast<PfnModelInitExt>(
            GetProcAddress(g.model, "NVSDK_NGX_D3D12_Init_Ext"));
        g.model_create = reinterpret_cast<PfnModelCreate>(
            GetProcAddress(g.model, "NVSDK_NGX_D3D12_CreateFeature"));
        g.model_evaluate = reinterpret_cast<PfnModelEvaluate>(
            GetProcAddress(g.model, "NVSDK_NGX_D3D12_EvaluateFeature"));
        g.model_release = reinterpret_cast<PfnModelRelease>(
            GetProcAddress(g.model, "NVSDK_NGX_D3D12_ReleaseFeature"));
        if (!g.model_init || !g.model_create || !g.model_evaluate) {
            Log("DLSS NR: %s lacks the NGX D3D12 entry points", Narrow(info->model_path).c_str());
            return kNgxFail;
        }
    }

    CreateDirectoryW(info->data_path, nullptr);
    int result = GuardedCoreInit(g.core_init, info->data_path, info->device);
    if (result != kNgxSuccess) {
        return Fail(result, "NGX core Init");
    }
    g.core_initialized = true;
    g.device = info->device;

    result = GuardedCoreAllocate(g.core_allocate, &g.params);
    if (result != kNgxSuccess || !g.params) {
        return Fail(result, "NGX core AllocateParameters");
    }

    result = GuardedModelInit(g.model_init, info->data_path, info->device, g.params);
    if (result != kNgxSuccess) {
        return Fail(result, "model Init_Ext");
    }
    g.initialized = true;
    return kNgxSuccess;
}

// Everything the model reads is latched at creation; values written only at evaluation are
// ignored. "Upscaling" does not make the model enlarge anything, so the feature is created at the
// frame size and colour and output share it
int SetCreateParams(unsigned width, unsigned height, const DlssNrModelParams* mp) {
    const int reset = GuardedParamsReset(g.params);
    if (reset != kNgxSuccess) {
        return reset;
    }
    NgxParameter* p = g.params;
    p->Set("Width", width);
    p->Set("Height", height);
    p->Set("OutWidth", width);
    p->Set("OutHeight", height);
    p->Set("PerfQualityValue", 2u);
    p->Set("CreationNodeMask", 1u);
    p->Set("VisibilityNodeMask", 1u);
    p->Set("DLSSNR.Enabled", 1u);
    p->Set("DLSSNR.Width", width);
    p->Set("DLSSNR.Height", height);
    p->Set("DLSSNR.Hint.Render.Preset", mp->preset);
    p->Set("DLSSNR.Intensity", mp->intensity);
    p->Set("DLSSNR.Style", mp->style);
    p->Set("DLSSNR.LocalStructureStrength", mp->local_structure);
    p->Set("DLSSNR.LocalToneStrength", mp->local_tone);
    if (mp->skin_structure >= 0.0f) {
        p->Set("DLSSNR.SkinStructureStrength", mp->skin_structure);
    }
    p->Set("DLSSNR.UseAutoMask", mp->auto_mask);
    p->Set("DLSSNR.UICorrection", mp->ui_correction);
    return kNgxSuccess;
}

void SetEvaluateParams(ID3D12Resource* color, ID3D12Resource* depth, ID3D12Resource* motion,
                       ID3D12Resource* output, unsigned width, unsigned height, int reset) {
    NgxParameter* p = g.params;
    p->Set("DLSSNR.Color", color);
    p->Set("DLSSNR.Depth", depth);
    p->Set("DLSSNR.MVec", motion);
    p->Set("DLSSNR.Output", output);
    p->Set("DLSSNR.Enabled", 1u);
    p->Set("DLSSNR.Reset", static_cast<unsigned>(reset));
    p->Set("DLSSNR.DepthInverted", 0u);
    p->Set("DLSSNR.ColorSubrectBaseX", 0u);
    p->Set("DLSSNR.ColorSubrectBaseY", 0u);
    p->Set("DLSSNR.ColorSubrectWidth", width);
    p->Set("DLSSNR.ColorSubrectHeight", height);
    p->Set("DLSSNR.OutputSubrectBaseX", 0u);
    p->Set("DLSSNR.OutputSubrectBaseY", 0u);
    p->Set("DLSSNR.OutputSubrectWidth", width);
    p->Set("DLSSNR.OutputSubrectHeight", height);
    p->Set("DLSSNR.DepthSubrectBaseX", 0u);
    p->Set("DLSSNR.DepthSubrectBaseY", 0u);
    p->Set("DLSSNR.DepthSubrectWidth", width);
    p->Set("DLSSNR.DepthSubrectHeight", height);
    p->Set("DLSSNR.MVecSubrectBaseX", 0u);
    p->Set("DLSSNR.MVecSubrectBaseY", 0u);
    p->Set("DLSSNR.MVecSubrectWidth", width);
    p->Set("DLSSNR.MVecSubrectHeight", height);
    p->Set("DLSSNR.MVecScaleX", 1.0f);
    p->Set("DLSSNR.MVecScaleY", 1.0f);
}

} // namespace

extern "C" {

__declspec(dllexport) int DlssNrBridgeInit(const DlssNrBridgeInitInfo* info) {
    return Init(info);
}

__declspec(dllexport) void* DlssNrBridgeCreate(ID3D12GraphicsCommandList* cmd, unsigned width,
                                               unsigned height, const DlssNrModelParams* params) {
    if (!g.initialized || g.faulted || !cmd || !params) {
        return nullptr;
    }
    int result = SetCreateParams(width, height, params);
    if (result != kNgxSuccess) {
        Fail(result, "parameter block Reset");
        return nullptr;
    }
    void* handle = nullptr;
    result = GuardedModelCreate(g.model_create, cmd, g.params, &handle);
    if (result != kNgxSuccess || !handle) {
        Fail(result, "model CreateFeature");
        return nullptr;
    }
    return handle;
}

__declspec(dllexport) int DlssNrBridgeEvaluate(ID3D12GraphicsCommandList* cmd, void* feature,
                                               ID3D12Resource* color, ID3D12Resource* depth,
                                               ID3D12Resource* motion, ID3D12Resource* output,
                                               unsigned width, unsigned height, int reset) {
    if (g.faulted) {
        return DLSSNR_BRIDGE_FAULTED;
    }
    if (!g.initialized || !cmd || !feature) {
        return kNgxFail;
    }
    SetEvaluateParams(color, depth, motion, output, width, height, reset);
    const int result = GuardedModelEvaluate(g.model_evaluate, cmd, feature, g.params);
    if (result != kNgxSuccess) {
        return Fail(result, "model EvaluateFeature");
    }
    return result;
}

__declspec(dllexport) void DlssNrBridgeRelease(void* feature) {
    if (!feature || g.faulted || !g.model_release) {
        return;
    }
    const int result = GuardedModelRelease(g.model_release, feature);
    if (result != kNgxSuccess) {
        Fail(result, "model ReleaseFeature");
    }
}

__declspec(dllexport) void DlssNrBridgeShutdown() {
    // After a fault the NGX state is undefined and a second fault in teardown would only hide
    // the first report
    if (g.faulted) {
        return;
    }
    if (g.params && g.core_destroy) {
        GuardedCoreDestroy(g.core_destroy, g.params);
    }
    g.params = nullptr;
    if (g.core_initialized && g.core_shutdown && g.device) {
        GuardedCoreShutdown(g.core_shutdown, g.device);
    }
    g.core_initialized = false;
    g.initialized = false;
    g.device = nullptr;
}

__declspec(dllexport) unsigned DlssNrBridgeReadLog(char* buffer, unsigned size) {
    std::scoped_lock lock{g_log_lock};
    if (!buffer || size == 0) {
        return static_cast<unsigned>(g_log.size());
    }
    const unsigned count =
        static_cast<unsigned>(g_log.size() < size - 1 ? g_log.size() : size - 1);
    memcpy(buffer, g_log.data(), count);
    buffer[count] = '\0';
    g_log.erase(0, count);
    return count;
}

} // extern "C"

BOOL WINAPI DllMain(HINSTANCE, DWORD, LPVOID) {
    return TRUE;
}
