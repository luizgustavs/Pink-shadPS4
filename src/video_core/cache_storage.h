// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/path_util.h"
#include "common/singleton.h"
#include "common/types.h"

#include <functional>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

namespace Storage {

enum class BlobType : u32 {
    ShaderMeta,
    ShaderBinary,
    PipelineKey,
    ShaderProfile,
};

class DataBase {
public:
    static DataBase& Instance() {
        return *Common::Singleton<DataBase>::Instance();
    }

    void Open();
    void Close();
    [[nodiscard]] bool IsOpened() const {
        return opened;
    }
    void FinishPreload();

    bool Save(BlobType type, const std::string& name, std::vector<u8>&& data);
    bool Save(BlobType type, const std::string& name, std::vector<u32>&& data);

    void Load(BlobType type, const std::string& name, std::vector<u8>& data);
    void Load(BlobType type, const std::string& name, std::vector<u32>& data);

    void ForEachBlob(BlobType type, const std::function<void(std::vector<u8>&& data)>& func);

    /// Choose an index at least as large as min_index and higher than every permutation already
    /// stored for this program
    /// Some stored permutations are not preloaded because their pipeline key is unused or was
    /// skipped after a conflict
    /// Their indices still belong to them, so reusing one would replace its .spv and .meta files
    /// with a different specialization
    /// The next startup could then load mismatched shader information from the cache, giving
    /// invalid program data and the PLANO S3 failure
    /// Use both the in-memory and stored indices to keep newly compiled permutations separate from
    /// all of those existing files
    u32 AllocateShaderBinaryIndex(u64 pgm_hash, u32 min_index);

private:
    void ScanShaderBinaryIndices();

    std::mutex index_mutex{};
    std::unordered_map<u64, u32> next_binary_index{}; ///< Next available index for each program, always above the highest permutation already stored on
///< disk
    std::jthread io_worker{};
    std::filesystem::path cache_path{};
    bool opened{};
};

} // namespace Storage
