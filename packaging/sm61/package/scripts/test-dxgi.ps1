param(
    [ValidateRange(0, 32)][int]$Adapter = 0,
    [ValidateRange(0, 32)][int]$Output = 0,
    [ValidateRange(1, 1000)][int]$MaxFrames = 3
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2.0
. (Join-Path $PSScriptRoot 'common.ps1')

Initialize-PackageDirectories
$root = Get-PackageRoot
$exe = Join-Path $root 'app\vision_analyzer.exe'
$config = Join-Path $root 'config\runtime-sm61.cfg'
$model = Join-Path $root 'model\best.onnx'
$schema = Join-Path $root 'model\best.onnx.schema.json'
$cache = Join-Path $root 'cache\ort-trt-sm61-fp32'
$logPath = New-PackageLogPath -Prefix 'dxgi'

foreach ($probeArgs in @(@('--list-dxgi-outputs'), @('--probe-dxgi-outputs'))) {
    $probe = Invoke-WithRuntimeEnvironment {
        Invoke-PackageCommand -FilePath $exe -Arguments $probeArgs -LogPath $logPath
    }
    if ($probe.ExitCode -ne 0) {
        throw "DXGI 枚举/探测失败，日志=$logPath"
    }
}

$verifyArgs = @(
    '--config', $config,
    '--input', 'dxgi',
    '--dxgi-adapter', $Adapter.ToString(),
    '--dxgi-output', $Output.ToString(),
    '--verify-input',
    '--dxgi-debug'
)
$verify = Invoke-WithRuntimeEnvironment {
    Invoke-PackageCommand -FilePath $exe -Arguments $verifyArgs -LogPath $logPath
}
if ($verify.ExitCode -ne 0 -or $verify.Text -notmatch 'input_verify') {
    throw "DXGI 输入验证失败；请根据探测结果调整 Adapter/Output，日志=$logPath"
}

$runArgs = @(
    '--config', $config,
    '--backend', 'ort-tensorrt',
    '--tensorrt-cache-path', $cache,
    '--model', $model,
    '--schema', $schema,
    '--input', 'dxgi',
    '--dxgi-adapter', $Adapter.ToString(),
    '--dxgi-output', $Output.ToString(),
    '--player-side', 'unknown',
    '--dry-run',
    '--warmup-frames', '1',
    '--max-frames', $MaxFrames.ToString(),
    '--status-every', '1'
)
$run = Invoke-WithRuntimeEnvironment {
    Invoke-PackageCommand -FilePath $exe -Arguments $runArgs -LogPath $logPath
}
if ($run.ExitCode -ne 0 -or $run.Text -notmatch "processed_frames=$MaxFrames") {
    throw "DXGI TensorRT 测试失败，日志=$logPath"
}

Write-Host "PASS DXGI：adapter=$Adapter output=$Output frames=$MaxFrames"
[pscustomobject]@{ Passed = $true; Adapter = $Adapter; Output = $Output; LogPath = $logPath }
