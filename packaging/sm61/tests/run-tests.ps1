$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2.0

$modulePath = Join-Path (Join-Path $PSScriptRoot '..') 'PackageTools.psm1'
Import-Module $modulePath -Force

$script:Passed = 0
$script:Failed = 0

function Assert-True {
    param(
        [Parameter(Mandatory)][bool]$Condition,
        [Parameter(Mandatory)][string]$Message
    )
    if (-not $Condition) {
        throw "ASSERT: $Message"
    }
}

function Assert-Equal {
    param(
        [Parameter(Mandatory)]$Expected,
        [Parameter(Mandatory)]$Actual,
        [Parameter(Mandatory)][string]$Message
    )
    if ($Expected -ne $Actual) {
        throw "ASSERT: $Message`nExpected: $Expected`nActual:   $Actual"
    }
}

function Assert-Throws {
    param(
        [Parameter(Mandatory)][scriptblock]$Action,
        [Parameter(Mandatory)][string]$Pattern,
        [Parameter(Mandatory)][string]$Message
    )

    try {
        & $Action
    } catch {
        if ($_.Exception.Message -notmatch $Pattern) {
            throw "ASSERT: $Message`nExpected error matching: $Pattern`nActual error: $($_.Exception.Message)"
        }
        return
    }
    throw "ASSERT: $Message`nExpected an exception, but none was thrown."
}

function Invoke-Test {
    param(
        [Parameter(Mandatory)][string]$Name,
        [Parameter(Mandatory)][scriptblock]$Body
    )

    try {
        & $Body
        $script:Passed++
        Write-Host "PASS $Name"
    } catch {
        $script:Failed++
        Write-Host "FAIL $Name"
        Write-Host $_.Exception.Message
    }
}

function New-EmptyFile {
    param([Parameter(Mandatory)][string]$LiteralPath)
    $parent = Split-Path -Parent $LiteralPath
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
    [IO.File]::WriteAllBytes($LiteralPath, [byte[]]@())
}

$testRoot = Join-Path ([System.IO.Path]::GetTempPath()) ('sm61-package-tests-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $testRoot -Force | Out-Null

try {
    Invoke-Test 'SHA256 is uppercase and stable' {
        $path = Join-Path $testRoot 'known-content.txt'
        [IO.File]::WriteAllText($path, 'known-content')
        $actual = Get-FileSha256 -LiteralPath $path
        Assert-Equal 'C651CCB96B0C0E490DE4CC12B9B46D643E6DBA87840FAB27E2C8D4D5CC2037FA' $actual 'known content hash'
    }

    Invoke-Test 'manifest round-trip detects tampering and unexpected files' {
        $root = Join-Path $testRoot 'manifest'
        New-Item -ItemType Directory -Path (Join-Path $root 'app') -Force | Out-Null
        [IO.File]::WriteAllText((Join-Path $root 'app\probe.txt'), 'known-content')

        Write-PackageManifest -PackageRoot $root -Profile 'sm61-test' -Components @(
            [pscustomobject]@{ id = 'fixture'; version = '1.0' }
        )

        $fresh = Test-PackageManifest -PackageRoot $root
        Assert-True $fresh.Valid 'fresh manifest must validate'
        Assert-Equal 'sm61-test' $fresh.Profile 'profile must round-trip'

        [IO.File]::AppendAllText((Join-Path $root 'app\probe.txt'), 'tampered')
        $tampered = Test-PackageManifest -PackageRoot $root
        Assert-True (-not $tampered.Valid) 'tampering must fail validation'
        Assert-True (@($tampered.Changed) -contains 'app/probe.txt') 'changed path must be reported'

        [IO.File]::WriteAllText((Join-Path $root 'app\extra.txt'), 'unexpected')
        $unexpected = Test-PackageManifest -PackageRoot $root
        Assert-True (@($unexpected.Unexpected) -contains 'app/extra.txt') 'unexpected path must be reported'
    }

    Invoke-Test 'manifest reports missing immutable files' {
        $root = Join-Path $testRoot 'manifest-missing'
        New-Item -ItemType Directory -Path (Join-Path $root 'app') -Force | Out-Null
        $path = Join-Path $root 'app\required.txt'
        [IO.File]::WriteAllText($path, 'required')
        Write-PackageManifest -PackageRoot $root -Profile 'sm61-test' -Components @()
        Remove-Item -LiteralPath $path -Force

        $result = Test-PackageManifest -PackageRoot $root
        Assert-True (-not $result.Valid) 'missing file must fail validation'
        Assert-True (@($result.Missing) -contains 'app/required.txt') 'missing path must be reported'
    }

    Invoke-Test 'compatible SM61 runtime filenames are accepted' {
        $root = Join-Path $testRoot 'compatible-runtime'
        New-EmptyFile (Join-Path $root 'app\onnxruntime_providers_cuda.dll')
        New-EmptyFile (Join-Path $root 'runtime\cuda-11.8\cudart64_110.dll')
        New-EmptyFile (Join-Path $root 'runtime\cuda-11.8\cublas64_11.dll')
        New-EmptyFile (Join-Path $root 'runtime\cuda-11.8\nvrtc64_112_0.dll')
        New-EmptyFile (Join-Path $root 'runtime\cudnn-8.9\cudnn64_8.dll')
        New-EmptyFile (Join-Path $root 'runtime\tensorrt-8.6.1.6\nvinfer.dll')
        Assert-CompatibleRuntimeFiles -PackageRoot $root
    }

    Invoke-Test 'CUDA 12 runtime filenames are rejected' {
        $root = Join-Path $testRoot 'cuda12-runtime'
        New-EmptyFile (Join-Path $root 'runtime\cuda-11.8\cublasLt64_12.dll')
        Assert-Throws { Assert-CompatibleRuntimeFiles -PackageRoot $root } 'CUDA 12|cublasLt64_12' 'CUDA 12 must be rejected'
    }

    Invoke-Test 'cuDNN 9 and TensorRT 10 runtime filenames are rejected' {
        $cudnnRoot = Join-Path $testRoot 'cudnn9-runtime'
        New-EmptyFile (Join-Path $cudnnRoot 'runtime\cudnn-8.9\cudnn64_9.dll')
        Assert-Throws { Assert-CompatibleRuntimeFiles -PackageRoot $cudnnRoot } 'cuDNN 9|cudnn64_9' 'cuDNN 9 must be rejected'

        $trtRoot = Join-Path $testRoot 'trt10-runtime'
        New-EmptyFile (Join-Path $trtRoot 'runtime\tensorrt-8.6.1.6\nvinfer_10.dll')
        Assert-Throws { Assert-CompatibleRuntimeFiles -PackageRoot $trtRoot } 'TensorRT 10|nvinfer_10' 'TensorRT 10 must be rejected'
    }

    Invoke-Test 'ORT provider DLL outside app is rejected' {
        $root = Join-Path $testRoot 'misplaced-ort'
        New-EmptyFile (Join-Path $root 'runtime\cuda-11.8\onnxruntime_providers_cuda.dll')
        Assert-Throws { Assert-CompatibleRuntimeFiles -PackageRoot $root } 'ONNX Runtime provider|onnxruntime_providers_cuda' 'misplaced ORT provider must be rejected'
    }

    Invoke-Test 'TensorRT 8.6.1.6 layout is accepted' {
        $root = Join-Path $testRoot 'trt-layout'
        $include = Join-Path $root 'include'
        New-Item -ItemType Directory -Path $include -Force | Out-Null
        [IO.File]::WriteAllText((Join-Path $include 'NvInferVersion.h'), @'
#define NV_TENSORRT_MAJOR 8
#define NV_TENSORRT_MINOR 6
#define NV_TENSORRT_PATCH 1
#define NV_TENSORRT_BUILD 6
'@)
        foreach ($name in @('nvinfer.dll', 'nvinfer_plugin.dll', 'nvonnxparser.dll', 'nvinfer_builder_resource.dll')) {
            New-EmptyFile (Join-Path $root ('lib\' + $name))
        }

        $result = Test-TensorRtArchiveLayout -ExtractedRoot $root
        Assert-True $result.Valid 'matching TensorRT layout must validate'
        Assert-Equal '8.6.1.6' $result.Version 'TensorRT version must be parsed from the header'
    }

    Invoke-Test 'wrong TensorRT build and missing DLL are rejected' {
        $root = Join-Path $testRoot 'wrong-trt-layout'
        $include = Join-Path $root 'include'
        New-Item -ItemType Directory -Path $include -Force | Out-Null
        [IO.File]::WriteAllText((Join-Path $include 'NvInferVersion.h'), @'
#define NV_TENSORRT_MAJOR 8
#define NV_TENSORRT_MINOR 6
#define NV_TENSORRT_PATCH 1
#define NV_TENSORRT_BUILD 5
'@)
        foreach ($name in @('nvinfer.dll', 'nvinfer_plugin.dll', 'nvonnxparser.dll')) {
            New-EmptyFile (Join-Path $root ('lib\' + $name))
        }

        Assert-Throws { Test-TensorRtArchiveLayout -ExtractedRoot $root } '8\.6\.1\.6|nvinfer_builder_resource' 'wrong TensorRT archive must be rejected'
    }

    Invoke-Test 'runtime PATH order is deterministic' {
        $root = Join-Path $testRoot 'path-order'
        $actual = @(Get-RuntimePathEntries -PackageRoot $root)
        $expected = @(
            (Join-Path $root 'app'),
            (Join-Path $root 'runtime\tensorrt-8.6.1.6'),
            (Join-Path $root 'runtime\cudnn-8.9'),
            (Join-Path $root 'runtime\cuda-11.8'),
            (Join-Path $root 'runtime\msvc-x64')
        )
        Assert-Equal ($expected -join '|') ($actual -join '|') 'runtime path precedence'
    }

    Invoke-Test 'dependency lock is complete and uses approved sources' {
        $lockPath = Join-Path (Join-Path $PSScriptRoot '..') 'dependencies.lock.json'
        Assert-True (Test-Path -LiteralPath $lockPath -PathType Leaf) 'dependencies.lock.json must exist'
        $lock = Get-Content -LiteralPath $lockPath -Raw | ConvertFrom-Json

        Assert-Equal 1 $lock.schemaVersion 'dependency lock schema'
        Assert-Equal 'sm61-ort1173-trt861-fp32' $lock.profile 'dependency profile'

        $expectedIds = @(
            'cuda-cublas',
            'cuda-cudart',
            'cuda-cufft',
            'cuda-nvrtc',
            'cudnn',
            'msvc-crt',
            'onnxruntime-gpu',
            'tensorrt'
        )
        $actualIds = @($lock.components | ForEach-Object { [string]$_.id } | Sort-Object)
        Assert-Equal ($expectedIds -join '|') ($actualIds -join '|') 'locked component IDs'
        Assert-Equal $actualIds.Count @($actualIds | Select-Object -Unique).Count 'component IDs must be unique'

        $approvedHosts = @('github.com', 'developer.download.nvidia.com', 'developer.nvidia.com')
        foreach ($component in @($lock.components)) {
            if ($component.sourceMode -eq 'public') {
                $uri = New-Object Uri([string]$component.url)
                Assert-Equal 'https' $uri.Scheme "public source must use HTTPS: $($component.id)"
                Assert-True ($approvedHosts -contains $uri.Host) "unapproved source host for $($component.id): $($uri.Host)"
                Assert-True ([string]$component.sha256 -match '^[0-9A-F]{64}$') "public archive must have uppercase SHA256: $($component.id)"
                Assert-True (-not [string]::IsNullOrWhiteSpace([string]$component.archive)) "public archive name is required: $($component.id)"
            } elseif ($component.sourceMode -eq 'authenticated-manual') {
                Assert-Equal 'tensorrt' ([string]$component.id) 'only TensorRT may require authenticated manual download'
                Assert-Equal 'TensorRT-8.6.1.6.Windows10.x86_64.cuda-11.8.zip' ([string]$component.archive) 'TensorRT archive name'
                Assert-Equal 'version-header-and-required-dlls' ([string]$component.validation) 'TensorRT validation policy'
            } elseif ($component.sourceMode -eq 'local-visual-studio-redist') {
                Assert-Equal 'msvc-crt' ([string]$component.id) 'only MSVC CRT may use local redistributable files'
            } else {
                throw "ASSERT: unsupported sourceMode for $($component.id): $($component.sourceMode)"
            }
        }
    }
} finally {
    if (Test-Path -LiteralPath $testRoot) {
        Remove-Item -LiteralPath $testRoot -Recurse -Force
    }
}

if ($script:Failed -gt 0) {
    Write-Host "FAILED $($script:Failed) test(s); passed $($script:Passed)"
    exit 1
}

Write-Host "PASS package tool tests ($($script:Passed))"
exit 0
