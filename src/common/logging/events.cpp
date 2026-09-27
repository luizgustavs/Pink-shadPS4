// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <fmt/format.h>

#include "common/logging/events.h"
#include "common/types.h"

namespace Common::Log {

namespace {

// Namespace-scope state, so it is still alive when Log::Shutdown runs from atexit

struct CallSite {
    const char* file;
    int line;
    bool operator==(const CallSite&) const = default;
};

struct CallSiteHash {
    size_t operator()(const CallSite& site) const noexcept {
        return std::hash<const void*>{}(site.file) ^ (static_cast<size_t>(site.line) << 1);
    }
};

struct MessageCount {
    Class log_class;
    Level level;
    u64 count;
    std::string first_message;
};

std::mutex g_counts_mutex;
std::unordered_map<CallSite, MessageCount, CallSiteHash> g_counts;

std::mutex g_first_time_mutex;
std::unordered_set<CallSite, CallSiteHash> g_seen_sites;
std::unordered_set<std::string> g_seen_keys;

std::atomic<bool> g_summary_printed{false};

thread_local ErrorScope* t_error_scope = nullptr;
thread_local bool t_in_error_scope = false;

constexpr size_t MaxStoredMessage = 160;
constexpr size_t SummaryRows = 12;

std::string_view KindName(EventKind kind) {
    switch (kind) {
    case EventKind::BrokenShader:
        return "BROKEN SHADER";
    case EventKind::Crash:
        return "CRASH";
    case EventKind::Render:
        return "RENDER";
    }
    return "EVENT";
}

// Console tags match the log levels so the launcher colors the lines like log messages
Level KindLevel(EventKind kind) {
    switch (kind) {
    case EventKind::BrokenShader:
        return Level::Error;
    case EventKind::Crash:
        return Level::Critical;
    case EventKind::Render:
        return Level::Warning;
    }
    return Level::Info;
}

std::string_view LevelName(Level level) {
    switch (level) {
    case Level::Trace:
        return "Trace";
    case Level::Debug:
        return "Debug";
    case Level::Info:
        return "Info";
    case Level::Warning:
        return "Warning";
    case Level::Error:
        return "Error";
    case Level::Critical:
        return "Critical";
    case Level::Off:
        break;
    }
    return "Off";
}

// First line only, without trailing spaces, capped
std::string OneLine(std::string_view text, size_t max_size) {
    const auto newline = text.find_first_of("\r\n");
    std::string line{text.substr(0, newline)};
    while (!line.empty() && (line.back() == ' ' || line.back() == ':')) {
        line.pop_back();
    }
    if (newline != std::string_view::npos && newline + 1 < text.size()) {
        line += " ...";
    }
    if (line.size() > max_size) {
        line.resize(max_size - 3);
        line += "...";
    }
    return line;
}

} // Anonymous namespace

void Event(EventKind kind, std::string_view headline,
           std::initializer_list<std::string_view> details) {
    if (!EventsEnabled()) {
        return;
    }
    const auto level = KindLevel(kind);
    std::string console = fmt::format("[{}] <{}> {}", KindName(kind), LevelName(level), headline);
    std::string file = fmt::format("[Event] {}: {}", KindName(kind), headline);
    for (const auto detail : details) {
        if (detail.empty()) {
            continue;
        }
        console += fmt::format("\n    {}", detail);
        file += fmt::format("\n    {}", detail);
    }
    Detail::WriteEvent(level, console, file);
}

ErrorScope::ErrorScope() : previous{t_error_scope} {
    t_error_scope = this;
}

ErrorScope::~ErrorScope() {
    t_error_scope = previous;
}

ErrorScope* CurrentErrorScope() {
    // A scope that logs an error while handling one must not be re-entered
    return t_in_error_scope ? nullptr : t_error_scope;
}

bool FirstTimeAt(const char* file, int line) {
    std::scoped_lock lock{g_first_time_mutex};
    return g_seen_sites.insert(CallSite{file, line}).second;
}

bool FirstTime(std::string_view key) {
    std::scoped_lock lock{g_first_time_mutex};
    return g_seen_keys.emplace(key).second;
}

void ReportRenderProblem(std::string_view message, std::string_view where) {
    Event(EventKind::Render, fmt::format("{}  ({})", OneLine(message, 200), where));
}

void PrintSessionSummary() {
    if (!EventsEnabled() || g_summary_printed.exchange(true)) {
        return;
    }
    std::vector<std::pair<CallSite, MessageCount>> rows;
    {
        std::scoped_lock lock{g_counts_mutex};
        rows.assign(g_counts.begin(), g_counts.end());
    }
    if (rows.empty()) {
        return;
    }
    u64 warnings = 0;
    u64 errors = 0;
    for (const auto& [site, entry] : rows) {
        (entry.level == Level::Warning ? warnings : errors) += entry.count;
    }
    std::ranges::sort(rows,
                      [](const auto& a, const auto& b) { return a.second.count > b.second.count; });

    std::string console = fmt::format(
        "[SUMMARY] <Info> {} warnings and {} errors/criticals logged from {} call sites", warnings,
        errors, rows.size());
    const auto log_path = Detail::LogFilePath();
    if (!log_path.empty()) {
        console += fmt::format(" (full text: {})", log_path);
    }
    const size_t shown = std::min(rows.size(), SummaryRows);
    for (size_t i = 0; i < shown; ++i) {
        const auto& [site, entry] = rows[i];
        console += fmt::format("\n    {:>8}x {:<8} {:<34} {}", entry.count, LevelName(entry.level),
                               fmt::format("{}:{}", Detail::BaseName(site.file), site.line),
                               entry.first_message);
    }
    if (rows.size() > shown) {
        console += fmt::format("\n    ... {} more call sites in the log file", rows.size() - shown);
    }
    Detail::WriteEvent(Level::Info, console, {});
}

namespace Detail {

std::string_view BaseName(std::string_view path) {
    const auto slash = path.find_last_of("/\\");
    return slash == std::string_view::npos ? path : path.substr(slash + 1);
}

void CountMessage(Class log_class, Level level, const char* file, int line,
                  std::string_view message) {
    std::scoped_lock lock{g_counts_mutex};
    auto [it, inserted] = g_counts.try_emplace(CallSite{file, line});
    auto& entry = it->second;
    if (inserted) {
        entry.log_class = log_class;
        entry.level = level;
        entry.first_message = OneLine(message, MaxStoredMessage);
    }
    ++entry.count;
}

ErrorScopeCallGuard::ErrorScopeCallGuard() : previous{t_in_error_scope} {
    t_in_error_scope = true;
}

ErrorScopeCallGuard::~ErrorScopeCallGuard() {
    t_in_error_scope = previous;
}

} // namespace Detail

} // namespace Common::Log
