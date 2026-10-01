#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
# SPDX-License-Identifier: GPL-2.0-or-later
"""Generate src/video_core/renderer_vulkan/vk_recorder_cmds.inc from the Vulkan headers
Include every vkCmd member of DispatchLoaderDynamic so cp_record_thread can install its
command handlers without keeping a separate list by hand
Rerun this script after updating externals/vulkan-headers so the list matches the dispatcher
"""
import re
from pathlib import Path

root = Path(__file__).resolve().parent.parent
header = root / 'externals/vulkan-headers/include/vulkan/vulkan.hpp'
out = root / 'src/video_core/renderer_vulkan/vk_recorder_cmds.inc'
version = re.search(r'#define VK_HEADER_VERSION (\d+)',
                    (header.parent / 'vulkan_core.h').read_text(encoding='utf-8'))[1]
lines = header.read_text(encoding='utf-8').split('\n')
start = next(i for i, l in enumerate(lines) if 'class DispatchLoaderDynamic : public DispatchLoaderBase' in l)
guards, entries = [], []
for line in lines[start:]:
    s = line.strip()
    if s.startswith('void init('):
        break
    if s.startswith('#ifdef'):
        guards.append(f'defined({s.split()[1]})')
    elif s.startswith('#ifndef'):
        guards.append(f'!defined({s.split()[1]})')
    elif s.startswith('#if'):
        guards.append(s[3:].strip())
    elif s.startswith('#endif'):
        guards.pop()
    elif s.startswith('#else'):
        guards[-1] = f'!({guards[-1]})'
    elif s.startswith('#elif'):
        raise SystemExit(f'unexpected #elif in DispatchLoaderDynamic: {s}')
    m = re.match(r'PFN_(vkCmd\w+)\s+(\w+)\s*=\s*0;', s)
    if m:
        entries.append((m[2], list(guards)))
body = []
for name, entry_guards in entries:
    for g in entry_guards:
        body.append(f'#if {g}')
    body.append(f'VK_RECORDER_CMD({name})')
    for _ in entry_guards:
        body.append('#endif')
out.write_text('// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project\n'
               '// SPDX-License-Identifier: GPL-2.0-or-later\n\n'
               f'// Generate this command list with scripts/gen_recorder_cmds.py from vulkan.hpp, using\n'
               f'// VK_HEADER_VERSION {version}\n'
               '// Include every vkCmd member of DispatchLoaderDynamic so all fake handles reach the recorder\n'
               '// Rerun the generator after updating the Vulkan headers instead of editing this list by hand\n\n' +
               '\n'.join(body) + '\n', encoding='utf-8', newline='\n')
print(f'{len(entries)} commands -> {out}')
