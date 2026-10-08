[CmdletBinding(SupportsShouldProcess)]
param()

$ErrorActionPreference = 'Stop'

$root = $PSScriptRoot
$output = Join-Path $root 'build'

# Preserve these paths relative to build/
$keep = @('games', 'user\sys_modules', 'user\home', 'user\savedata', 'user\users.json')

function Test-Kept([string]$relative) {
    foreach ($path in $keep) {
        if ($relative -ieq $path) { return $true }
    }
    return $false
}

function Test-KeptAncestor([string]$relative) {
    foreach ($path in $keep) {
        if ($path.StartsWith("$relative\", [StringComparison]::OrdinalIgnoreCase)) { return $true }
    }
    return $false
}

function Clear-Directory([string]$directory, [string]$relative) {
    foreach ($item in Get-ChildItem -LiteralPath $directory -Force) {
        $itemRelative = if ($relative) { "$relative\$($item.Name)" } else { $item.Name }

        if (Test-Kept $itemRelative) { continue }

        if (-not $item.PSIsContainer) {
            if ($item.Extension -ine '.zip' -and $PSCmdlet.ShouldProcess($itemRelative, 'Remove file')) {
                Remove-Item -LiteralPath $item.FullName -Force
            }
            continue
        }

        $hasZip = Get-ChildItem -LiteralPath $item.FullName -Recurse -File -Filter '*.zip' -Force |
            Select-Object -First 1
        if ((Test-KeptAncestor $itemRelative) -or $hasZip) {
            Clear-Directory $item.FullName $itemRelative
            if (-not (Get-ChildItem -LiteralPath $item.FullName -Force) -and
                $PSCmdlet.ShouldProcess($itemRelative, 'Remove empty directory')) {
                Remove-Item -LiteralPath $item.FullName -Force
            }
        } elseif ($PSCmdlet.ShouldProcess($itemRelative, 'Remove directory')) {
            Remove-Item -LiteralPath $item.FullName -Recurse -Force
        }
    }
}

if (-not (Test-Path -LiteralPath $output)) {
    Write-Host "Nothing to clean: $output does not exist."
    return
}

Clear-Directory $output ''
Write-Host "Cleaned $output (kept *.zip, $($keep -join ', '))."
