param([string]$Reason = 'manual')

$ErrorActionPreference = 'Continue'
Set-StrictMode -Version 2.0
. (Join-Path $PSScriptRoot 'common.ps1')

Initialize-PackageDirectories
$root = Get-PackageRoot
$reportPath = New-PackageLogPath -Prefix 'diagnostics'
$lines = New-Object Collections.Generic.List[string]
$lines.Add("generated=$(Get-Date -Format o)")
$lines.Add("reason=$Reason")
$lines.Add("package_root=$root")
$lines.Add("os=$([Environment]::OSVersion.VersionString)")
$lines.Add("powershell=$($PSVersionTable.PSVersion)")

try {
    $manifest = Get-Content -LiteralPath (Join-Path $root 'runtime-manifest.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    $lines.Add("profile=$($manifest.profile)")
    foreach ($component in @($manifest.components)) {
        $lines.Add("component=$($component.id);version=$($component.version)")
    }
} catch {
    $lines.Add("manifest_error=$($_.Exception.Message)")
}

try {
    foreach ($gpu in @(Get-NvidiaGpuInfo)) {
        $lines.Add("gpu=$($gpu.Name);driver=$($gpu.DriverVersion);memory_mib=$($gpu.MemoryMiB)")
    }
} catch {
    $lines.Add("gpu_error=$($_.Exception.Message)")
}

foreach ($relative in @(Get-RequiredRuntimeFiles)) {
    $path = Join-Path $root $relative
    if (Test-Path -LiteralPath $path -PathType Leaf) {
        $item = Get-Item -LiteralPath $path
        $lines.Add("file=$relative;size=$($item.Length)")
    } else {
        $lines.Add("missing=$relative")
    }
}

$cache = Join-Path $root 'cache\ort-trt-sm61-fp32'
if (Test-Path -LiteralPath $cache) {
    foreach ($file in @(Get-ChildItem -LiteralPath $cache -File -Recurse)) {
        $lines.Add("cache_file=$($file.Name);size=$($file.Length)")
    }
}

try {
    $help = Invoke-WithRuntimeEnvironment {
        Invoke-PackageCommand -FilePath (Join-Path $root 'app\vision_analyzer.exe') -Arguments @('--help') -Quiet
    }
    $lines.Add("cli_help_exit=$($help.ExitCode)")
} catch {
    $lines.Add("cli_help_error=$($_.Exception.Message)")
}

[IO.File]::WriteAllLines($reportPath, $lines, (New-Object Text.UTF8Encoding($false)))
Write-Host "诊断报告：$reportPath"
$reportPath
