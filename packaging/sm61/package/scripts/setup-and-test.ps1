param(
    [ValidateRange(1, 1000)][int]$MaxFrames = 3
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2.0
. (Join-Path $PSScriptRoot 'common.ps1')

try { [Console]::OutputEncoding = New-Object Text.UTF8Encoding($false) } catch {}
Initialize-PackageDirectories
$root = Get-PackageRoot
Push-Location $root
try {
    Write-Host '============================================================'
    Write-Host ' GTX 1080 Ti / ORT TensorRT 8.6 一键检查与安全测试'
    Write-Host ' 默认禁止 RP2350 输出，不会移动鼠标或点击。'
    Write-Host '============================================================'

    $verify = & (Join-Path $PSScriptRoot 'verify-runtime.ps1')
    $video = & (Join-Path $PSScriptRoot 'test-video.ps1') -MaxFrames $MaxFrames -PassCount 2

    Write-Host ''
    Write-Host 'PASS：运行环境和 TensorRT 视频测试均已通过。'
    Write-Host ("配置：{0}" -f $verify.Profile)
    Write-Host ("缓存：{0}（{1} 个文件）" -f $video.CachePath, $video.CacheFileCount)
    Write-Host '下一步如需测试当前屏幕，请按 README_中文.md 单独运行 DXGI 脚本。'
    exit 0
} catch {
    Write-Host ''
    Write-Host ("FAIL：{0}" -f $_.Exception.Message) -ForegroundColor Red
    try {
        $report = & (Join-Path $PSScriptRoot 'collect-diagnostics.ps1') -Reason $_.Exception.Message
        Write-Host "请把诊断报告发回来：$report"
    } catch {
        Write-Host ("生成诊断报告也失败：{0}" -f $_.Exception.Message)
    }
    exit 1
} finally {
    Pop-Location
}
