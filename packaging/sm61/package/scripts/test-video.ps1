param(
    [ValidateRange(1, 1000)][int]$MaxFrames = 3,
    [ValidateRange(1, 5)][int]$PassCount = 2
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
$sample = Join-Path $root 'samples\smoke-test.mp4'
$cache = Join-Path $root 'cache\ort-trt-sm61-fp32'
$results = New-Object Collections.Generic.List[object]

Assert-RequiredRuntimeFiles
for ($pass = 1; $pass -le $PassCount; $pass++) {
    $logPath = New-PackageLogPath -Prefix ("video-pass-$pass")
    $actionPath = Join-Path (Join-Path $root 'logs') ("video-pass-$pass-actions.txt")
    $arguments = @(
        '--config', $config,
        '--backend', 'ort-tensorrt',
        '--tensorrt-cache-path', $cache,
        '--model', $model,
        '--schema', $schema,
        '--video', $sample,
        '--player-side', 'unknown',
        '--dry-run',
        '--warmup-frames', '1',
        '--max-frames', $MaxFrames.ToString(),
        '--status-every', '1',
        '--action-log', $actionPath
    )

    Write-PackageLog -LogPath $logPath -Message "开始 TensorRT 视频测试，第 $pass/$PassCount 次。首次运行可能需要数分钟构建引擎。"
    $stopwatch = [Diagnostics.Stopwatch]::StartNew()
    $result = Invoke-WithRuntimeEnvironment {
        Invoke-PackageCommand -FilePath $exe -Arguments $arguments -LogPath $logPath
    }
    $stopwatch.Stop()
    if ($result.ExitCode -ne 0) {
        throw "TensorRT 视频测试第 $pass 次失败，退出码=$($result.ExitCode)，日志=$logPath"
    }
    if ($result.Text -notmatch "processed_frames=$MaxFrames") {
        throw "TensorRT 视频测试没有处理预期的 $MaxFrames 帧，日志=$logPath"
    }

    $timing = [regex]::Matches($result.Text, 'fps=(?<fps>[0-9.]+).*?preprocess_ms=(?<pre>[0-9.]+).*?inference_ms=(?<infer>[0-9.]+).*?postprocess_ms=(?<post>[0-9.]+).*?total_ms=(?<total>[0-9.]+)')
    if ($timing.Count -eq 0) {
        throw "CLI 没有输出 FPS/阶段耗时，日志=$logPath"
    }
    $last = $timing[$timing.Count - 1]
    $results.Add([pscustomobject]@{
        Pass = $pass
        ElapsedSeconds = [Math]::Round($stopwatch.Elapsed.TotalSeconds, 3)
        Fps = [double]$last.Groups['fps'].Value
        PreprocessMs = [double]$last.Groups['pre'].Value
        InferenceMs = [double]$last.Groups['infer'].Value
        PostprocessMs = [double]$last.Groups['post'].Value
        TotalMs = [double]$last.Groups['total'].Value
        LogPath = $logPath
    })
}

$cacheFiles = @(Get-ChildItem -LiteralPath $cache -File -Recurse | Where-Object { $_.Length -gt 0 })
if ($cacheFiles.Count -eq 0) {
    throw "TensorRT 测试完成但未生成引擎缓存：$cache"
}

$lastResult = $results[$results.Count - 1]
Write-Host ("PASS TensorRT 视频测试：FPS={0:N2}，推理={1:N2} ms，缓存文件={2}" -f $lastResult.Fps, $lastResult.InferenceMs, $cacheFiles.Count)
[pscustomobject]@{
    Passed = $true
    Runs = @($results)
    CachePath = $cache
    CacheFileCount = $cacheFiles.Count
}
