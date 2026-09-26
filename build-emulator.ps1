param(
    [ValidateSet('Debug', 'Release', 'RelWithDebInfo')]
    [string]$Configuration = 'Release',

    [ValidateRange(1, 1024)]
    [int]$Jobs = [Environment]::ProcessorCount,

    [string]$ClangPath,

    [switch]$Fresh
)

$ErrorActionPreference = 'Stop'

$root = $PSScriptRoot
$binary = Join-Path $root 'out\emulator'
$output = Join-Path $root 'build'

function Import-VsEnvironment {
    $programRoots = @($env:ProgramFiles, [Environment]::GetEnvironmentVariable('ProgramFiles(x86)'))
    $vsDevCmd = $programRoots |
        Where-Object { $_ } |
        ForEach-Object { Get-Item -Path (Join-Path $_ 'Microsoft Visual Studio\*\*\Common7\Tools\VsDevCmd.bat') -ErrorAction SilentlyContinue } |
        Sort-Object FullName -Descending |
        Select-Object -First 1
    if (-not $vsDevCmd) {
        throw 'Visual Studio (or Build Tools) with C++ support was not found.'
    }
    $ErrorActionPreference = 'Continue'
    $vsEnvironment = & $env:ComSpec /d /s /c "call `"$($vsDevCmd.FullName)`" -arch=x64 -host_arch=x64 >nul && set"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to initialize the Visual Studio environment: $($vsDevCmd.FullName)"
    }
    foreach ($line in $vsEnvironment) {
        if ($line -match '^([^=]+)=(.*)$') {
            [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process')
        }
    }
}

function Find-ClangCl {
    param([string]$Requested)

    if ($Requested) {
        return (Resolve-Path -LiteralPath $Requested -ErrorAction Stop).Path
    }
    $command = Get-Command clang-cl -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }
    $candidates = @()
    if ($env:VCINSTALLDIR) {
        $candidates += Join-Path $env:VCINSTALLDIR 'Tools\Llvm\x64\bin\clang-cl.exe'
    }
    $candidates += Get-Item -Path (Join-Path $root '.tools\clang+llvm-*\bin\clang-cl.exe') -ErrorAction SilentlyContinue |
        Sort-Object FullName -Descending |
        ForEach-Object { $_.FullName }
    $found = $candidates | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
    if (-not $found) {
        throw 'clang-cl was not found. Pass -ClangPath, install Clang through the Visual Studio Installer or extract LLVM into .tools\.'
    }
    return $found
}

function Invoke-CMake {
    param([string[]]$CMakeArguments)

    $ErrorActionPreference = 'Continue'
    & cmake @CMakeArguments
    if ($LASTEXITCODE -ne 0) {
        throw "CMake failed with exit code $LASTEXITCODE."
    }
}

Import-VsEnvironment
$clang = Find-ClangCl -Requested $ClangPath
$env:PATH = "$(Split-Path $clang -Parent);$env:PATH"
foreach ($tool in 'cmake', 'ninja') {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        throw "$tool was not found in PATH."
    }
}

$cache = Join-Path $binary 'CMakeCache.txt'
if (Test-Path -LiteralPath $cache) {
    $cachedHome = Select-String -LiteralPath $cache -Pattern '^CMAKE_HOME_DIRECTORY:INTERNAL=(.*)$' |
        Select-Object -First 1 | ForEach-Object { $_.Matches[0].Groups[1].Value }
    $sameSource = $cachedHome -and
        ([IO.Path]::GetFullPath($cachedHome).TrimEnd('\', '/') -eq [IO.Path]::GetFullPath($root).TrimEnd('\', '/'))
    if (-not $sameSource -or -not (Test-Path -LiteralPath (Join-Path $binary 'build.ninja'))) {
        $Fresh = $true
    }
}

$configure = @(
    '-S', $root,
    '-B', $binary,
    '-G', 'Ninja',
    "-DCMAKE_BUILD_TYPE=$Configuration",
    "-DCMAKE_C_COMPILER=$clang",
    "-DCMAKE_CXX_COMPILER=$clang",
    "-DCMAKE_ASM_COMPILER=$clang",
    "-DFETCHCONTENT_SOURCE_DIR_FMT=$(Join-Path $root 'externals\fmt')"
)
if ($Fresh) {
    $configure = @('--fresh') + $configure
}

Write-Host "Configuring emulator ($Configuration, $clang)..."
Invoke-CMake -CMakeArguments $configure

Write-Host 'Building emulator...'
Invoke-CMake -CMakeArguments @('--build', $binary, '--target', 'shadps4', '--parallel', "$Jobs")

New-Item -ItemType Directory -Path $output -Force | Out-Null
$artifacts = @(Get-Item -Path (Join-Path $binary 'shadps4.exe'))
$artifacts += Get-Item -Path (Join-Path $binary '*.dll') -ErrorAction SilentlyContinue
if ($Configuration -ne 'Release') {
    $artifacts += Get-Item -Path (Join-Path $binary 'shadps4.pdb') -ErrorAction SilentlyContinue
}
foreach ($artifact in $artifacts) {
    Copy-Item -LiteralPath $artifact.FullName -Destination $output -Force
    Write-Host "  -> build\$($artifact.Name)"
}
Write-Host "Emulator ready in $output"
