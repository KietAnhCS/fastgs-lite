# Adds Defender exclusions for the LaTeX toolchain.
# Run once, from an ELEVATED PowerShell (Run as administrator):
#     powershell -ExecutionPolicy Bypass -File .\latex\setup-defender-exclusions.ps1

$ErrorActionPreference = 'Stop'

if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Error "Can chay bang PowerShell voi quyen Administrator."
}

# Every pdflatex run reads ~358 files from the MiKTeX tree; scanning them each
# time is the bulk of the per-compile antivirus cost.
$paths = @(
    "$env:LOCALAPPDATA\Programs\MiKTeX",   # the installation (binaries + packages)
    "$env:LOCALAPPDATA\MiKTeX",            # generated fonts and the package cache
    "$PSScriptRoot"                        # this latex/ folder, incl. out/
)

$procs = @(
    "$env:LOCALAPPDATA\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe",
    "$env:LOCALAPPDATA\Programs\MiKTeX\miktex\bin\x64\miktex-pdftex.exe"
)

foreach ($p in $paths) {
    if (Test-Path $p) {
        Add-MpPreference -ExclusionPath $p
        Write-Host "  + path    $p"
    } else {
        Write-Warning "  ! khong tim thay: $p"
    }
}

foreach ($p in $procs) {
    if (Test-Path $p) {
        Add-MpPreference -ExclusionProcess $p
        Write-Host "  + process $p"
    }
}

Write-Host "`n--- Exclusions sau khi them ---"
(Get-MpPreference).ExclusionPath
