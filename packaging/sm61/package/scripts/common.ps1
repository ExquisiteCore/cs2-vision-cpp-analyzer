Set-StrictMode -Version 2.0

$script:PackageRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..')).TrimEnd('\')

function Get-PackageRoot {
    $script:PackageRoot
}

function Initialize-PackageDirectories {
    $root = Get-PackageRoot
    foreach ($relative in @('logs', 'cache', 'cache\ort-trt-sm61-fp32')) {
        New-Item -ItemType Directory -Path (Join-Path $root $relative) -Force | Out-Null
    }
}

function New-PackageLogPath {
    param([Parameter(Mandatory)][string]$Prefix)
    Initialize-PackageDirectories
    $stamp = [DateTime]::Now.ToString('yyyyMMdd-HHmmss')
    Join-Path (Join-Path (Get-PackageRoot) 'logs') ("$Prefix-$stamp.log")
}

function Write-PackageLog {
    param(
        [Parameter(Mandatory)][string]$Message,
        [string]$LogPath,
        [ValidateSet('INFO', 'WARN', 'ERROR')][string]$Level = 'INFO'
    )
    $line = '[{0}] [{1}] {2}' -f [DateTime]::Now.ToString('yyyy-MM-dd HH:mm:ss'), $Level, $Message
    Write-Host $line
    if (-not [string]::IsNullOrWhiteSpace($LogPath)) {
        Add-Content -LiteralPath $LogPath -Value $line -Encoding UTF8
    }
}

function Get-PackageRelativePath {
    param([Parameter(Mandatory)][string]$LiteralPath)
    $root = (Get-PackageRoot).TrimEnd('\')
    $full = [IO.Path]::GetFullPath($LiteralPath)
    if (-not $full.StartsWith($root + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "Path is outside package root: $full"
    }
    $full.Substring($root.Length + 1).Replace('\', '/')
}

function Get-ImmutablePackageFiles {
    $root = Get-PackageRoot
    Get-ChildItem -LiteralPath $root -File -Recurse | Where-Object {
        $relative = Get-PackageRelativePath -LiteralPath $_.FullName
        $relative -ne 'runtime-manifest.json' -and
        $relative -notlike 'logs/*' -and
        $relative -notlike 'cache/*' -and
        $relative -notmatch '(?i)(^|/)__pycache__(/|$)' -and
        $relative -notmatch '(?i)\.py[co]$'
    } | Sort-Object FullName
}

function Test-PackageManifest {
    $root = Get-PackageRoot
    $manifestPath = Join-Path $root 'runtime-manifest.json'
    if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
        throw "运行环境清单不存在：$manifestPath"
    }

    $manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($manifest.schemaVersion -ne 1) {
        throw "不支持的运行环境清单版本：$($manifest.schemaVersion)"
    }

    $expected = @{}
    foreach ($entry in @($manifest.files)) {
        $relative = [string]$entry.path
        if ([string]::IsNullOrWhiteSpace($relative) -or $relative.Contains('..') -or $relative.StartsWith('/')) {
            throw "运行环境清单包含不安全路径：$relative"
        }
        if ($expected.ContainsKey($relative)) {
            throw "运行环境清单包含重复路径：$relative"
        }
        $expected[$relative] = $entry
    }

    $missing = New-Object Collections.Generic.List[string]
    $changed = New-Object Collections.Generic.List[string]
    foreach ($relative in @($expected.Keys | Sort-Object)) {
        $entry = $expected[$relative]
        $full = Join-Path $root $relative.Replace('/', '\')
        if (-not (Test-Path -LiteralPath $full -PathType Leaf)) {
            $missing.Add($relative)
            continue
        }
        $item = Get-Item -LiteralPath $full
        $hash = (Get-FileHash -LiteralPath $full -Algorithm SHA256).Hash.ToUpperInvariant()
        if ([int64]$item.Length -ne [int64]$entry.size -or $hash -ne [string]$entry.sha256) {
            $changed.Add($relative)
        }
    }

    $unexpected = New-Object Collections.Generic.List[string]
    foreach ($file in @(Get-ImmutablePackageFiles)) {
        $relative = Get-PackageRelativePath -LiteralPath $file.FullName
        if (-not $expected.ContainsKey($relative)) {
            $unexpected.Add($relative)
        }
    }

    [pscustomobject]@{
        Valid = ($missing.Count -eq 0 -and $changed.Count -eq 0 -and $unexpected.Count -eq 0)
        Profile = [string]$manifest.profile
        Manifest = $manifest
        Missing = @($missing)
        Changed = @($changed)
        Unexpected = @($unexpected)
    }
}

function Get-RuntimePathEntries {
    $root = Get-PackageRoot
    @(
        (Join-Path $root 'app')
        (Join-Path $root 'runtime\tensorrt-8.6.1.6')
        (Join-Path $root 'runtime\cudnn-8.9')
        (Join-Path $root 'runtime\cuda-11.8')
        (Join-Path $root 'runtime\msvc-x64')
    )
}

function Invoke-WithRuntimeEnvironment {
    param([Parameter(Mandatory)][scriptblock]$ScriptBlock)
    $oldPath = $env:PATH
    try {
        $privatePath = (Get-RuntimePathEntries) -join ';'
        $env:PATH = $privatePath + ';' + $oldPath
        & $ScriptBlock
    } finally {
        $env:PATH = $oldPath
    }
}

function Invoke-PackageCommand {
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [string[]]$Arguments = @(),
        [string]$LogPath,
        [switch]$Quiet
    )
    $oldErrorActionPreference = $ErrorActionPreference
    try {
        # Windows PowerShell 5.1 promotes native stderr to a terminating
        # RemoteException when the caller uses Stop, even if the process exits 0.
        $ErrorActionPreference = 'Continue'
        $captured = @(& $FilePath @Arguments 2>&1)
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $oldErrorActionPreference
    }
    $lines = @($captured | ForEach-Object { $_.ToString() })
    if (-not [string]::IsNullOrWhiteSpace($LogPath)) {
        Add-Content -LiteralPath $LogPath -Value ($lines -join [Environment]::NewLine) -Encoding UTF8
    }
    if (-not $Quiet) {
        foreach ($line in $lines) { Write-Host $line }
    }
    [pscustomobject]@{
        ExitCode = [int]$exitCode
        Lines = $lines
        Text = $lines -join [Environment]::NewLine
    }
}

function Get-RequiredRuntimeFiles {
    @(
        'app\vision_runtime.dll',
        'app\vision_runtime.lib',
        'app\vision_analyzer.exe',
        'app\vision_runtime_c_api.h',
        'app\onnxruntime.dll',
        'app\onnxruntime_providers_shared.dll',
        'app\onnxruntime_providers_cuda.dll',
        'app\onnxruntime_providers_tensorrt.dll',
        'model\best.onnx',
        'model\best.onnx.schema.json',
        'samples\smoke-test.mp4',
        'python\cs2_vision_runtime\__init__.py',
        'python\cs2_vision_runtime\runtime.py',
        'examples\runtime_live_move.py',
        'runtime\cuda-11.8\cudart64_110.dll',
        'runtime\cuda-11.8\cublas64_11.dll',
        'runtime\cuda-11.8\cublasLt64_11.dll',
        'runtime\cuda-11.8\cufft64_10.dll',
        'runtime\cudnn-8.9\cudnn64_8.dll',
        'runtime\tensorrt-8.6.1.6\nvinfer.dll',
        'runtime\tensorrt-8.6.1.6\nvinfer_plugin.dll',
        'runtime\tensorrt-8.6.1.6\nvonnxparser.dll',
        'runtime\tensorrt-8.6.1.6\nvinfer_builder_resource.dll',
        'runtime\msvc-x64\MSVCP140.dll',
        'runtime\msvc-x64\VCRUNTIME140.dll',
        'runtime\msvc-x64\VCRUNTIME140_1.dll',
        'runtime\msvc-x64\CONCRT140.dll'
    )
}

function Assert-RequiredRuntimeFiles {
    $root = Get-PackageRoot
    $missing = @(
        Get-RequiredRuntimeFiles | Where-Object {
            -not (Test-Path -LiteralPath (Join-Path $root $_) -PathType Leaf)
        }
    )
    if ($missing.Count -gt 0) {
        throw "运行环境缺少文件：$($missing -join ', ')"
    }
}

function Assert-PackageRuntimeGenerations {
    $root = Get-PackageRoot
    foreach ($file in @(Get-ChildItem -LiteralPath $root -File -Recurse -Filter '*.dll')) {
        $name = $file.Name.ToLowerInvariant()
        $relative = Get-PackageRelativePath -LiteralPath $file.FullName
        $wrongCuda =
            $name -match '^cublas(lt)?64_12\.dll$' -or
            $name -match '^cudart64_12\d*\.dll$' -or
            $name -match '^cufft64_11\.dll$' -or
            $name -match '^nvrtc64_12\d_0\.dll$'
        if ($wrongCuda -or $name -match '^cudnn.*64_9\.dll$' -or $name -match '^(nvinfer|nvonnxparser|nvparsers).*_(10|11)\.dll$') {
            throw "发现与 SM61 配置不兼容的 DLL：$relative"
        }
    }
}

function Get-NvidiaGpuInfo {
    $command = Get-Command 'nvidia-smi.exe' -ErrorAction SilentlyContinue
    if ($null -eq $command) {
        $command = Get-Command 'nvidia-smi' -ErrorAction SilentlyContinue
    }
    if ($null -eq $command) {
        throw '找不到 nvidia-smi；请先安装 NVIDIA 显卡驱动。'
    }

    $output = @(& $command.Source '--query-gpu=name,driver_version,memory.total' '--format=csv,noheader,nounits' 2>&1)
    if ($LASTEXITCODE -ne 0) {
        throw "nvidia-smi 执行失败：$($output -join ' ')"
    }
    foreach ($line in $output) {
        $parts = $line.ToString().Split(',')
        if ($parts.Count -ge 3) {
            [pscustomobject]@{
                Name = $parts[0].Trim()
                DriverVersion = $parts[1].Trim()
                MemoryMiB = [int]$parts[2].Trim()
            }
        }
    }
}
