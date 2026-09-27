// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstdint>
#include <initializer_list>
#include <string>
#include <string_view>
#include <fmt/format.h>

#include "common/logging/log.h"

// Console events: the few things worth reading while a game runs. Off by default; with
// Log.console_mode = "events" the console only shows these, Critical messages and a summary of the
// most repeated messages on exit. The log file keeps every message as before and also gets the
// events as "[Event] KIND: ..." lines

namespace Common::Log {

enum class EventKind : std::uint8_t {
    BrokenShader,
    Crash,
    Render,
};

// True with Log.console_mode = "events"; everything below is a no-op otherwise
bool EventsEnabled();

// Prints a headline and optional indented detail lines
void Event(EventKind kind, std::string_view headline,
           std::initializer_list<std::string_view> details = {});

// Receives the Error and Critical messages logged on the thread that installed it, e.g. by the
// shader compiler while it translates one shader. Scopes nest; only the innermost one is notified
class ErrorScope {
public:
    ErrorScope();
    virtual ~ErrorScope();

    ErrorScope(const ErrorScope&) = delete;
    ErrorScope& operator=(const ErrorScope&) = delete;

    virtual void OnError(Class log_class, Level level, std::string_view message) = 0;

    // The thread is about to die inside this scope (failed assertion or unhandled exception)
    virtual void OnFatal(std::string_view reason) = 0;

private:
    ErrorScope* previous;
};

ErrorScope* CurrentErrorScope();

// True the first time it is called for a call site / a key, so an event can be printed once
bool FirstTimeAt(const char* file, int line);
bool FirstTime(std::string_view key);

// Prints a Render event; "where" is the reporting call site or subsystem
void ReportRenderProblem(std::string_view message, std::string_view where);

// Last Critical message logged by this thread as "file:line: text", empty if none
std::string LastCriticalMessage();

// Prints the most repeated Warning/Error call sites of the session to the console, once
void PrintSessionSummary();

namespace Detail {
std::string_view BaseName(std::string_view path);
void CountMessage(Class log_class, Level level, const char* file, int line,
                  std::string_view message);
// Writes console_text to the console (skipped if empty) and file_text to the log file
void WriteEvent(Level level, std::string_view console_text, std::string_view file_text);

std::string LogFilePath();

// Marks this thread as running an ErrorScope callback, so errors it logs are not fed back to the
// scope
class ErrorScopeCallGuard {
public:
    ErrorScopeCallGuard();
    ~ErrorScopeCallGuard();

private:
    bool previous;
};

// While alive, Critical messages of this thread go to the log file only (the caller prints an event
// instead)
class FileOnlyCriticals {
public:
    FileOnlyCriticals();
    ~FileOnlyCriticals();

private:
    bool previous;
};
} // namespace Detail

} // namespace Common::Log

// Logs like LOG_GENERIC; the first message of the call site is also printed as a Render event
// (repeats are only counted in the session summary)
#define LOG_RENDER_PROBLEM(log_class, log_level, ...)                                              \
    do {                                                                                           \
        LOG_GENERIC(log_class, Common::Log::Level::log_level, __VA_ARGS__);                        \
        if (Common::Log::EventsEnabled() && Common::Log::FirstTimeAt(__FILE__, __LINE__)) {        \
            Common::Log::ReportRenderProblem(                                                      \
                fmt::format(__VA_ARGS__),                                                          \
                fmt::format("{}:{}", Common::Log::Detail::BaseName(__FILE__), __LINE__));          \
        }                                                                                          \
    } while (false)
