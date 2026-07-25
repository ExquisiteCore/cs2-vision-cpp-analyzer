param(
    [switch]$AllowUnsupportedGpu,
    [switch]$StaticOnly
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2.0
. (Join-Path $PSScriptRoot 'common.ps1')

try { [Console]::OutputEncoding = New-Object Text.UTF8Encoding($false) } catch {}
Initialize-PackageDirectories
$root = Get-PackageRoot
$logPath = New-PackageLogPath -Prefix 'verify-runtime'

Write-PackageLog -LogPath $logPath -Message '开始验证 SM61 便携运行环境。'
$manifest = Test-PackageManifest
if (-not $manifest.Valid) {
    throw "文件清单校验失败；缺失=$($manifest.Missing -join ',')，变化=$($manifest.Changed -join ',')，额外=$($manifest.Unexpected -join ',')"
}
if ($manifest.Profile -ne 'sm61-ort1173-trt861-fp32') {
    throw "运行环境配置错误：$($manifest.Profile)"
}
Assert-Rp2350ProtocolV2Manifest -Manifest $manifest.Manifest
Write-PackageLog -LogPath $logPath -Message "清单校验通过：$($manifest.Profile)"

Assert-RequiredRuntimeFiles
Assert-PackageRuntimeGenerations
Write-PackageLog -LogPath $logPath -Message 'DLL、模型和测试输入完整。'

$cache = Join-Path $root 'cache\ort-trt-sm61-fp32'
New-Item -ItemType Directory -Path $cache -Force | Out-Null
$writeProbe = Join-Path $cache ('.write-probe-' + [guid]::NewGuid().ToString('N'))
[IO.File]::WriteAllText($writeProbe, 'ok')
Remove-Item -LiteralPath $writeProbe -Force
Write-PackageLog -LogPath $logPath -Message "TensorRT 缓存目录可写：$cache"

$gpus = @(Get-NvidiaGpuInfo)
if ($gpus.Count -eq 0) {
    throw 'nvidia-smi 没有返回可用 GPU。'
}
foreach ($gpu in $gpus) {
    Write-PackageLog -LogPath $logPath -Message ("GPU={0}; driver={1}; memory={2} MiB" -f $gpu.Name, $gpu.DriverVersion, $gpu.MemoryMiB)
}

$targetGpu = @($gpus | Where-Object { $_.Name -match 'GTX\s+1080\s+Ti' } | Select-Object -First 1)
if ($targetGpu.Count -eq 0 -and -not $AllowUnsupportedGpu) {
    throw "未检测到 GTX 1080 Ti；检测到：$($gpus.Name -join ', ')"
}
$driverSource = if ($targetGpu.Count -gt 0) { $targetGpu[0] } else { $gpus[0] }
try {
    $driverVersion = New-Object Version($driverSource.DriverVersion)
    if ($driverVersion -lt (New-Object Version('522.06'))) {
        throw "NVIDIA 驱动过旧：$driverVersion；CUDA 11.8 Windows 需要 522.06 或更高版本。"
    }
} catch [ArgumentException] {
    throw "无法解析 NVIDIA 驱动版本：$($driverSource.DriverVersion)"
}

$exe = Join-Path $root 'app\vision_analyzer.exe'
$helpLog = New-PackageLogPath -Prefix 'cli-help'
$helpResult = Invoke-WithRuntimeEnvironment {
    Invoke-PackageCommand -FilePath $exe -Arguments @('--help') -LogPath $helpLog -Quiet
}
if ($helpResult.ExitCode -ne 0 -or $helpResult.Text -notmatch 'vision_analyzer') {
    throw "CLI 无法在私有运行环境中启动；日志：$helpLog"
}

Write-PackageLog -LogPath $logPath -Message ($(if ($StaticOnly) { '静态验证通过；未加载 TensorRT 引擎。' } else { '运行环境验证通过；TensorRT 推理由视频测试执行。' }))
[pscustomobject]@{
    Valid = $true
    Profile = $manifest.Profile
    Gpus = $gpus
    TargetGpuFound = ($targetGpu.Count -gt 0)
    DriverVersion = $driverSource.DriverVersion
    StaticOnly = [bool]$StaticOnly
    LogPath = $logPath
}
