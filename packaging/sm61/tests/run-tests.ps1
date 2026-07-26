param(
    [string]$PythonProjectRoot = ''
)

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

function New-PortablePackageFixture {
    param(
        [Parameter(Mandatory)][string]$LiteralPath,
        [switch]$SkipManifest
    )

    foreach ($relative in @(
        'app',
        'model',
        'runtime\cuda-11.8',
        'runtime\cudnn-8.9',
        'runtime\tensorrt-8.6.1.6',
        'runtime\msvc-x64',
        'config',
        'licenses\runtime',
        'python\cs2_vision_runtime',
        'python\rp2350_hid_bridge',
        'examples',
        'logs',
        'cache'
    )) {
        New-Item -ItemType Directory -Path (Join-Path $LiteralPath $relative) -Force | Out-Null
    }

    $files = [ordered]@{
        'app\vision_runtime.dll' = 'runtime-dll'
        'app\rp2350_hid_bridge.dll' = 'RP2350 protocol v2 capabilities are required rp2350_hid_session_mouse_move'
        'app\rp2350_hid_bridge.lib' = 'hid-import-library'
        'app\rp2350_hid_bridge_c_api.h' = 'hid-c-api'
        'app\vision_analyzer.exe' = 'diagnostic-exe'
        'app\onnxruntime.dll' = 'ort-core'
        'app\onnxruntime_providers_shared.dll' = 'ort-shared'
        'app\onnxruntime_providers_cuda.dll' = 'ort-cuda'
        'app\onnxruntime_providers_tensorrt.dll' = 'ort-tensorrt'
        'model\best.onnx' = 'model'
        'model\best.onnx.schema.json' = '{"classes":["ct_head","ct_body","t_head","t_body"]}'
        'runtime\cuda-11.8\cudart64_110.dll' = 'cuda'
        'runtime\cudnn-8.9\cudnn64_8.dll' = 'cudnn'
        'runtime\tensorrt-8.6.1.6\nvinfer.dll' = 'tensorrt'
        'runtime\msvc-x64\VCRUNTIME140.dll' = 'msvc'
        'config\runtime-sm61.cfg' = 'backend=ort-tensorrt'
        'licenses\runtime\NOTICE.txt' = 'license'
        'python\cs2_vision_runtime\runtime.py' = 'must-not-copy'
        'python\rp2350_hid_bridge\__init__.py' = 'hid-python-init'
        'python\rp2350_hid_bridge\_version.py' = '__version__ = "0.2.0"'
        'python\rp2350_hid_bridge\native.py' = 'hid-native-loader'
        'python\rp2350_hid_bridge\client.py' = 'hid-client'
        'examples\runtime_live_move.py' = 'must-not-copy'
        'logs\old.log' = 'must-not-copy'
        'cache\old.engine' = 'must-not-copy'
    }
    foreach ($entry in $files.GetEnumerator()) {
        [IO.File]::WriteAllText(
            (Join-Path $LiteralPath ([string]$entry.Key)),
            [string]$entry.Value
        )
    }

    if (-not $SkipManifest) {
        Write-PackageManifest `
            -PackageRoot $LiteralPath `
            -Profile 'sm61-ort1173-trt861-fp32' `
            -Components @(
                [pscustomobject]@{ id = 'onnxruntime-gpu'; version = '1.17.3' },
                [pscustomobject]@{ id = 'cuda-cudart'; version = '11.8' },
                [pscustomobject]@{ id = 'cudnn'; version = '8.9.7' },
                [pscustomobject]@{ id = 'tensorrt'; version = '8.6.1.6' },
                [pscustomobject]@{ id = 'msvc-crt'; version = '14.x-v14-compatible' }
            )
    }
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
        $unicodeName = 'README_' + [char]0x4E2D + [char]0x6587 + '.md'
        [IO.File]::WriteAllText((Join-Path $root $unicodeName), 'unicode-path')

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

    Invoke-Test 'package-side manifest validates UTF-8 paths under Windows PowerShell' {
        $root = Join-Path $testRoot 'package-manifest-unicode'
        $scripts = Join-Path $root 'scripts'
        New-Item -ItemType Directory -Path $scripts -Force | Out-Null
        Copy-Item -LiteralPath (Join-Path (Join-Path $PSScriptRoot '..') 'package\scripts\common.ps1') -Destination (Join-Path $scripts 'common.ps1')
        $unicodeName = 'README_' + [char]0x4E2D + [char]0x6587 + '.md'
        [IO.File]::WriteAllText((Join-Path $root $unicodeName), 'unicode-path')
        Write-PackageManifest -PackageRoot $root -Profile 'sm61-test' -Components @()

        & {
            . (Join-Path $scripts 'common.ps1')
            $result = Test-PackageManifest
            Assert-True $result.Valid 'package-side validator must preserve UTF-8 manifest paths'

            New-EmptyFile (Join-Path $root 'python\cs2_vision_runtime\__pycache__\runtime.cpython-312.pyc')
            $withPythonCache = Test-PackageManifest
            Assert-True $withPythonCache.Valid 'Python bytecode cache must be treated as mutable runtime data'

            New-EmptyFile (Join-Path $root 'python\unexpected.bin')
            $withUnexpectedFile = Test-PackageManifest
            Assert-True (-not $withUnexpectedFile.Valid) 'non-cache Python files must remain protected by the manifest'
        }
    }

    Invoke-Test 'package-side manifest requires one RP2350 protocol v2 component' {
        $common = Join-Path (Join-Path $PSScriptRoot '..') 'package\scripts\common.ps1'
        & {
            . $common

            $missing = [pscustomobject]@{ components = @() }
            Assert-Throws {
                Assert-Rp2350ProtocolV2Manifest -Manifest $missing
            } 'rp2350-hid-sdk|protocol-v2' 'missing RP2350 build metadata must be rejected'

            $legacy = [pscustomobject]@{
                components = @([pscustomobject]@{
                    id = 'rp2350-hid-sdk'
                    version = 'protocol-v1'
                    sourceMode = 'header-only-build'
                })
            }
            Assert-Throws {
                Assert-Rp2350ProtocolV2Manifest -Manifest $legacy
            } 'protocol-v2' 'legacy RP2350 build metadata must be rejected'

            $current = [pscustomobject]@{
                components = @([pscustomobject]@{
                    id = 'rp2350-hid-sdk'
                    version = 'abi-1.0-protocol-v2'
                    sourceMode = 'shared-library'
                })
            }
            Assert-Rp2350ProtocolV2Manifest -Manifest $current
        }
    }

    Invoke-Test 'package command preserves native stderr warnings under Windows PowerShell 5.1' {
        $common = Join-Path (Join-Path $PSScriptRoot '..') 'package\scripts\common.ps1'
        . $common

        $result = Invoke-PackageCommand `
            -FilePath $env:ComSpec `
            -Arguments @('/d', '/c', 'echo native-warning 1>&2 & echo processed_frames=3 & exit /b 0') `
            -Quiet

        Assert-Equal 0 $result.ExitCode 'a warning on native stderr must not abort a successful command'
        Assert-True ($result.Text -match 'native-warning') 'native stderr must remain available in captured output'
        Assert-True ($result.Text -match 'processed_frames=3') 'native stdout after the warning must remain available'
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

    Invoke-Test 'RP2350 protocol v2 binary gate rejects legacy releases' {
        $root = Join-Path $testRoot 'rp2350-binary-gate'
        New-Item -ItemType Directory -Path $root -Force | Out-Null
        $legacyDll = Join-Path $root 'legacy.dll'
        $v2Dll = Join-Path $root 'v2.dll'
        [IO.File]::WriteAllText($legacyDll, 'legacy RP2350 HID runtime')
        [IO.File]::WriteAllText(
            $v2Dll,
            'RP2350 protocol v2 capabilities are required rp2350_hid_session_mouse_move'
        )

        Assert-Throws {
            Assert-Rp2350SharedLibrary -LiteralPath $legacyDll
        } 'protocol v2' 'legacy HID release DLL must be rejected'
        Assert-Rp2350SharedLibrary -LiteralPath $v2Dll
    }

    Invoke-Test 'TensorRT 8.6.1.6 layout is accepted' {
        $root = Join-Path $testRoot 'trt-layout'
        $include = Join-Path $root 'include'
        New-Item -ItemType Directory -Path $include -Force | Out-Null
        $officialHeaderLines = @(
            '#define NV_TENSORRT_MAJOR 8 //!< TensorRT major version.',
            '#define NV_TENSORRT_MINOR 6 //!< TensorRT minor version.',
            '#define NV_TENSORRT_PATCH 1 //!< TensorRT patch version.',
            '#define NV_TENSORRT_BUILD 6 //!< TensorRT build number.'
        )
        [IO.File]::WriteAllText(
            (Join-Path $include 'NvInferVersion.h'),
            ($officialHeaderLines -join "`r`n") + "`r`n"
        )
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

    Invoke-Test 'project asset root resolves from a nested worktree' {
        $project = Join-Path $testRoot 'asset-project'
        New-Item -ItemType Directory -Path (Join-Path $project 'runs') -Force | Out-Null
        New-Item -ItemType Directory -Path (Join-Path $project 'videos') -Force | Out-Null
        New-Item -ItemType Directory -Path (Join-Path $project 'tools\cpp_analyzer\.worktrees\feature\packaging\sm61') -Force | Out-Null
        $start = Join-Path $project 'tools\cpp_analyzer\.worktrees\feature'
        $resolved = Find-ProjectAssetRoot -StartPath $start
        Assert-Equal $project $resolved 'nested worktree must resolve the parent project that owns runs and videos'
    }

    Invoke-Test 'latest compatible x64 MSVC v14 private runtime is selected' {
        $vsRoot = Join-Path $testRoot 'fake-visual-studio'
        $vc143 = Join-Path $vsRoot 'VC\Redist\MSVC\14.44.100\x64\Microsoft.VC143.CRT'
        $vc145 = Join-Path $vsRoot 'VC\Redist\MSVC\14.50.200\x64\Microsoft.VC145.CRT'
        $oneCore = Join-Path $vsRoot 'VC\Redist\MSVC\14.60.300\onecore\x64\Microsoft.VC146.CRT'
        foreach ($directory in @($vc143, $vc145, $oneCore)) {
            foreach ($name in @('MSVCP140.dll', 'VCRUNTIME140.dll', 'VCRUNTIME140_1.dll', 'CONCRT140.dll')) {
                New-EmptyFile (Join-Path $directory $name)
            }
        }
        $incomplete = Join-Path $vsRoot 'VC\Redist\MSVC\14.70.400\x64\Microsoft.VC147.CRT'
        foreach ($name in @('MSVCP140.dll', 'VCRUNTIME140.dll', 'CONCRT140.dll')) {
            New-EmptyFile (Join-Path $incomplete $name)
        }

        $resolved = Find-MsvcPrivateRuntimeRoot -SearchRoots @($vsRoot)
        Assert-Equal $vc145 $resolved 'latest complete desktop x64 v14 runtime should be selected; onecore and incomplete candidates must be excluded'
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
        $msvcComponent = @($lock.components | Where-Object { $_.id -eq 'msvc-crt' })[0]
        Assert-Equal '14.x-v14-compatible' ([string]$msvcComponent.version) 'MSVC lock must allow the binary-compatible v14 redistributable family'

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

    Invoke-Test 'package templates exist and safe defaults are fixed' {
        $templateRoot = Join-Path (Join-Path $PSScriptRoot '..') 'package'
        $required = @(
            'config/runtime-sm61.cfg',
            'scripts/common.ps1',
            'scripts/verify-runtime.ps1',
            'scripts/test-video.ps1',
            'scripts/test-dxgi.ps1',
            'scripts/collect-diagnostics.ps1',
            'scripts/setup-and-test.ps1',
            (([string][char]0x4E00) + [char]0x952E + [char]0x68C0 + [char]0x67E5 + [char]0x5E76 + [char]0x6D4B + [char]0x8BD5 + '.cmd'),
            ('README_' + [char]0x4E2D + [char]0x6587 + '.md')
        )
        foreach ($relative in $required) {
            $path = Join-Path $templateRoot $relative.Replace('/', '\')
            Assert-True (Test-Path -LiteralPath $path -PathType Leaf) "required package template is missing: $relative"
        }

        $config = Get-Content -LiteralPath (Join-Path $templateRoot 'config\runtime-sm61.cfg') -Raw
        Assert-True ($config -match '(?m)^backend=ort-tensorrt\s*$') 'TensorRT must be the config backend'
        Assert-True ($config -match '(?m)^tensorrt_cache_path=cache/ort-trt-sm61-fp32\s*$') 'cache path must be package-relative'
        Assert-True ($config -match '(?m)^output_enabled=false\s*$') 'output must default disabled'
        Assert-True ($config -match '(?m)^fire_enabled=false\s*$') 'automatic fire must default disabled'
        Assert-True ($config -match '(?m)^body_fire_enabled=true\s*$') 'body fallback policy must be explicit'
        Assert-True ($config -match '(?m)^head_fire_confidence=0\.35\s*$') 'head fire confidence must be fixed'
        Assert-True ($config -match '(?m)^body_fire_confidence=0\.45\s*$') 'body fire confidence must be fixed'
        Assert-True ($config -match '(?m)^hid_click_cooldown_frames=3\s*$') 'aggressive fire cooldown must be three frames'
        Assert-True ($config -match '(?m)^input=dxgi\s*$') 'DXGI must remain the live input default'

        $scriptFiles = @(Get-ChildItem -LiteralPath $templateRoot -File -Recurse | Where-Object { $_.Extension -in @('.ps1', '.cmd') })
        $scriptText = ($scriptFiles | ForEach-Object { Get-Content -LiteralPath $_.FullName -Raw }) -join "`n"
        Assert-True ($scriptText -notmatch '(?i)--output-enabled') 'package scripts must never arm HID output'
        Assert-True ($scriptText -notmatch '(?im)\bsetx(?:\.exe)?\b') 'package scripts must not persist PATH or variables'
        Assert-True ($scriptText -notmatch '(?i)HKLM:|HKCU:|New-ItemProperty|Set-ItemProperty') 'package scripts must not write the registry'

        $video = Get-Content -LiteralPath (Join-Path $templateRoot 'scripts\test-video.ps1') -Raw
        foreach ($token in @('--dry-run', '--max-frames', '--model', '--schema', '--tensorrt-cache-path', '--backend')) {
            Assert-True ($video.Contains($token)) "video test must contain $token"
        }
        Assert-True ($video -notmatch '(?i)--hid-port') 'video smoke test must not select an HID port'
        Assert-True ($video -notmatch 'Collections\.Generic\.List\[object\]') 'video results must use Windows PowerShell 5.1-compatible arrays'

        $setup = Get-Content -LiteralPath (Join-Path $templateRoot 'scripts\setup-and-test.ps1') -Raw
        Assert-True ($setup -notmatch '(?i)test-dxgi\.ps1') 'one-click setup must not run DXGI automatically'

        $readmeName = 'README_' + [char]0x4E2D + [char]0x6587 + '.md'
        $readme = Get-Content -LiteralPath (Join-Path $templateRoot $readmeName) -Raw -Encoding UTF8
        $protocolV2 = ([string][char]0x534F) + [char]0x8BAE + ' v2'
        $twoSeconds = ([string][char]0x4E24) + [char]0x79D2
        foreach ($token in @($protocolV2, 'ping/info/caps', '500 ms', $twoSeconds)) {
            Assert-True ($readme.Contains($token)) "package README must document $token"
        }
        foreach ($token in @(
            'set_hid_calibration_path',
            'get_hid_calibration',
            '--calibration-path',
            '--recalibrate',
            'max_step=120',
            '120 counts',
            'PYTHON_RUNTIME_SDK_INTEGRATION.md'
        )) {
            Assert-True ($readme.Contains($token)) "package README must document $token"
        }
        Assert-True ($readme -notmatch '(?i)2048\s+counts') 'package README must not describe the retired 2048-count probe'

        $repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\..'))
        $cApiHeader = Get-Content -LiteralPath `
            (Join-Path $repoRoot 'include\vision_analyzer\vision_runtime_c_api.h') `
            -Raw
        foreach ($token in @('va_set_hid_calibration_path', 'va_get_hid_calibration')) {
            Assert-True ($cApiHeader.Contains($token)) "C API header must declare $token"
        }

        $common = Get-Content -LiteralPath (Join-Path $templateRoot 'scripts\common.ps1') -Raw
        foreach ($relative in @(
            'app\rp2350_hid_bridge.dll',
            'app\rp2350_hid_bridge.lib',
            'app\rp2350_hid_bridge_c_api.h',
            'python\cs2_vision_runtime\__init__.py',
            'python\cs2_vision_runtime\runtime.py',
            'python\cs2_vision_runtime\_version.py',
            'python\cs2_vision_runtime\errors.py',
            'python\cs2_vision_runtime\package.py',
            'python\rp2350_hid_bridge\__init__.py',
            'python\rp2350_hid_bridge\_version.py',
            'python\rp2350_hid_bridge\native.py',
            'python\rp2350_hid_bridge\client.py',
            'examples\runtime_live_move.py',
            'examples\runtime_dxgi_dryrun.py',
            'docs\PYTHON_RUNTIME_SDK_INTEGRATION.md'
        )) {
            Assert-True ($common.Contains($relative)) "static package verification must require $relative"
        }

        $cmdName = ([string][char]0x4E00) + [char]0x952E + [char]0x68C0 + [char]0x67E5 + [char]0x5E76 + [char]0x6D4B + [char]0x8BD5 + '.cmd'
        $cmd = Get-Content -LiteralPath (Join-Path $templateRoot $cmdName) -Raw
        Assert-True ($cmd.Contains('%~dp0')) 'CMD launcher must resolve its own directory'
        Assert-True ($cmd -match '(?i)-ExecutionPolicy\s+Bypass') 'CMD launcher must use process-scoped policy bypass'
        Assert-True ($cmd -match '(?i)setup-and-test\.ps1') 'CMD launcher must invoke the orchestrator'
    }

    Invoke-Test 'PowerShell package templates parse under Windows PowerShell' {
        $templateRoot = Join-Path (Join-Path $PSScriptRoot '..') 'package'
        foreach ($file in @(Get-ChildItem -LiteralPath (Join-Path $templateRoot 'scripts') -File -Filter '*.ps1')) {
            $tokens = $null
            $errors = $null
            [void][Management.Automation.Language.Parser]::ParseFile($file.FullName, [ref]$tokens, [ref]$errors)
            Assert-Equal 0 @($errors).Count "PowerShell syntax errors in $($file.Name): $($errors -join '; ')"
        }
    }

    Invoke-Test 'build definitions pin RP2350 protocol v2 and thread support' {
        $repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\..'))
        $cmake = Get-Content -LiteralPath (Join-Path $repoRoot 'CMakeLists.txt') -Raw
        $hidOutput = Get-Content -LiteralPath (Join-Path $repoRoot 'src\hid_output.cpp') -Raw

        Assert-True `
            ($cmake -match 'cmake_minimum_required\(VERSION 3\.20\)') `
            'CMake must match the SDK minimum version'
        Assert-True `
            ($cmake.Contains('add_subdirectory(')) `
            'CMake must build the shared HID SDK'
        Assert-True `
            ($cmake.Contains('target_link_libraries(vision_analyzer_core PUBLIC rp2350_hid_bridge)')) `
            'vision core must link the shared HID library target'
        Assert-True `
            ($hidOutput.Contains('PROTOCOL_VERSION == kRp2350ProtocolV2')) `
            'the compiled runtime must reject an old SDK'
    }

    Invoke-Test 'verified dependency cache accepts matching hash and quarantines corruption' {
        $cache = Join-Path $testRoot 'archive-cache'
        New-Item -ItemType Directory -Path $cache -Force | Out-Null
        $archive = Join-Path $cache 'fixture.zip'
        [IO.File]::WriteAllText($archive, 'verified-archive')
        $component = [pscustomobject]@{
            id = 'fixture'
            sourceMode = 'public'
            archive = 'fixture.zip'
            url = 'https://developer.download.nvidia.com/fixture.zip'
            sha256 = Get-FileSha256 -LiteralPath $archive
        }

        $resolved = Get-VerifiedArchive -Component $component -CacheRoot $cache
        Assert-Equal $archive $resolved 'matching cached archive must be reused'

        [IO.File]::WriteAllText($archive, 'corrupted')
        Assert-Throws { Get-VerifiedArchive -Component $component -CacheRoot $cache } 'SHA256|download' 'corrupt cache entry must fail without download permission'
        Assert-True (-not (Test-Path -LiteralPath $archive)) 'corrupt cache entry must be moved away'
        Assert-Equal 1 @(Get-ChildItem -LiteralPath $cache -File -Filter 'fixture.zip.bad-*').Count 'one quarantined archive must remain for diagnosis'
    }

    Invoke-Test 'archive extraction copies runtime DLLs and component licenses only' {
        $source = Join-Path $testRoot 'archive-source'
        $archive = Join-Path $testRoot 'runtime-fixture.zip'
        New-EmptyFile (Join-Path $source 'cuda-package\bin\cudart64_110.dll')
        New-EmptyFile (Join-Path $source 'cuda-package\include\must-not-copy.dll')
        [IO.File]::WriteAllText((Join-Path $source 'cuda-package\LICENSE.txt'), 'license text')
        [IO.File]::WriteAllText((Join-Path $source 'cuda-package\ThirdPartyNotices.txt'), 'third-party notice')
        New-Item -ItemType Directory -Path (Join-Path $source 'cuda-package\doc') -Force | Out-Null
        [IO.File]::WriteAllText((Join-Path $source 'cuda-package\doc\Acknowledgements.txt'), 'acknowledgements')
        Compress-Archive -Path (Join-Path $source '*') -DestinationPath $archive -Force

        $expanded = Join-Path $testRoot 'archive-expanded'
        Expand-DependencyArchive -ArchivePath $archive -DestinationPath $expanded
        $runtimeDestination = Join-Path $testRoot 'cuda-runtime'
        Copy-ComponentRuntimeFiles -ExtractedRoot $expanded -DestinationPath $runtimeDestination -Layout 'bin'
        Assert-True (Test-Path -LiteralPath (Join-Path $runtimeDestination 'cudart64_110.dll')) 'bin DLL must be flattened into runtime directory'
        Assert-True (-not (Test-Path -LiteralPath (Join-Path $runtimeDestination 'must-not-copy.dll'))) 'include DLL must not be copied'

        $licenseDestination = Join-Path $testRoot 'cuda-license'
        Copy-ComponentLicenses -ExtractedRoot $expanded -DestinationPath $licenseDestination
        Assert-True (Test-Path -LiteralPath (Join-Path $licenseDestination 'LICENSE.txt')) 'component license must be preserved'
        Assert-True (Test-Path -LiteralPath (Join-Path $licenseDestination 'ThirdPartyNotices.txt')) 'third-party notice must be preserved'
        Assert-True (Test-Path -LiteralPath (Join-Path $licenseDestination 'Acknowledgements.txt')) 'component acknowledgements must be preserved'
    }

    Invoke-Test 'TensorRT runtime copy uses lib DLLs and ignores import libraries' {
        $source = Join-Path $testRoot 'trt-copy-source'
        New-EmptyFile (Join-Path $source 'TensorRT-8.6.1.6\lib\nvinfer.dll')
        New-EmptyFile (Join-Path $source 'TensorRT-8.6.1.6\lib\nvinfer.lib')
        New-EmptyFile (Join-Path $source 'TensorRT-8.6.1.6\include\must-not-copy.dll')
        $destination = Join-Path $testRoot 'trt-copy-destination'
        Copy-ComponentRuntimeFiles -ExtractedRoot $source -DestinationPath $destination -Layout 'lib'

        Assert-True (Test-Path -LiteralPath (Join-Path $destination 'nvinfer.dll')) 'TensorRT lib DLL must be copied'
        Assert-True (-not (Test-Path -LiteralPath (Join-Path $destination 'nvinfer.lib'))) 'TensorRT import library must not be copied'
        Assert-True (-not (Test-Path -LiteralPath (Join-Path $destination 'must-not-copy.dll'))) 'TensorRT include DLL must not be copied'
    }

    Invoke-Test 'missing authenticated TensorRT archive is a hard gate' {
        $installers = Join-Path $testRoot 'empty-installers'
        New-Item -ItemType Directory -Path $installers -Force | Out-Null
        Assert-Throws {
            Resolve-TensorRtArchive -ExplicitPath '' -InstallersRoot $installers
        } 'TensorRT-8\.6\.1\.6\.Windows10\.x86_64\.cuda-11\.8\.zip|NVIDIA' 'missing TensorRT archive must stop package creation'
    }

    Invoke-Test 'package builder entry point parses and validates source inputs before mutation' {
        $builder = Join-Path (Join-Path $PSScriptRoot '..') 'build-portable-package.ps1'
        Assert-True (Test-Path -LiteralPath $builder -PathType Leaf) 'build-portable-package.ps1 must exist'

        $tokens = $null
        $errors = $null
        [void][Management.Automation.Language.Parser]::ParseFile($builder, [ref]$tokens, [ref]$errors)
        Assert-Equal 0 @($errors).Count "builder syntax errors: $($errors -join '; ')"

        $missingRelease = Join-Path $testRoot 'missing-release-output'
        $outputRoot = Join-Path $testRoot 'builder-output'
        $cacheRoot = Join-Path $testRoot 'builder-cache'
        $oldErrorAction = $ErrorActionPreference
        try {
            $ErrorActionPreference = 'Continue'
            $captured = @(& powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File $builder `
                -ReleaseRoot $missingRelease -OutputRoot $outputRoot -DependencyCache $cacheRoot 2>&1)
            $exitCode = $LASTEXITCODE
        } finally {
            $ErrorActionPreference = $oldErrorAction
        }
        Assert-True ($exitCode -ne 0) 'builder must reject a missing release root'
        Assert-True (($captured -join ' ') -match 'Release output|release.*exist|构建产物') 'builder must explain the missing release output'

        $content = Get-Content -LiteralPath $builder -Raw
        foreach ($token in @(
            'Resolve-TensorRtArchive',
            'Write-PackageManifest',
            'Assert-CompatibleRuntimeFiles',
            'tar.exe',
            'PythonProjectRoot',
            'src\cs2_vision_runtime',
            'examples\runtime_live_move.py',
            'examples\runtime_dxgi_dryrun.py',
            'docs\PYTHON_RUNTIME_SDK_INTEGRATION.md',
            'python\cs2_vision_runtime',
            'python\rp2350_hid_bridge',
            'rp2350_hid_bridge.dll',
            'rp2350_hid_bridge.lib',
            'rp2350_hid_bridge_c_api.h',
            'Assert-Rp2350SharedLibrary',
            'rp2350-hid-sdk',
            'abi-1.0-protocol-v2',
            'shared-library'
        )) {
            Assert-True ($content.Contains($token)) "builder must use $token"
        }
        Assert-True ($content -notmatch '(?im)\bsetx(?:\.exe)?\b') 'builder must not persist environment changes'
    }

    Invoke-Test 'app-local builder requires an intact verified portable package' {
        $builder = Join-Path (Join-Path $PSScriptRoot '..') 'build-app-local-package.ps1'
        Assert-True (Test-Path -LiteralPath $builder -PathType Leaf) 'build-app-local-package.ps1 must exist'

        $withoutManifest = Join-Path $testRoot 'app-local-source-without-manifest'
        New-PortablePackageFixture -LiteralPath $withoutManifest -SkipManifest
        Assert-Throws {
            & $builder `
                -PortablePackageRoot $withoutManifest `
                -OutputRoot (Join-Path $testRoot 'app-local-missing-manifest-output')
        } 'manifest' 'app-local builder must reject a source without its verified manifest'

        $tampered = Join-Path $testRoot 'app-local-tampered-source'
        New-PortablePackageFixture -LiteralPath $tampered
        [IO.File]::AppendAllText((Join-Path $tampered 'model\best.onnx'), '-tampered')
        Assert-Throws {
            & $builder `
                -PortablePackageRoot $tampered `
                -OutputRoot (Join-Path $testRoot 'app-local-tampered-output')
        } 'validation|manifest|integrity|changed' 'app-local builder must reject changed source files'
    }

    Invoke-Test 'app-local builder creates the frozen-client layout and manifest' {
        $builder = Join-Path (Join-Path $PSScriptRoot '..') 'build-app-local-package.ps1'
        $source = Join-Path $testRoot 'app-local-source'
        $output = Join-Path $testRoot 'MyClient'
        New-PortablePackageFixture -LiteralPath $source

        & $builder `
            -PortablePackageRoot $source `
            -OutputRoot $output `
            -PythonSdkVersion '0.3.0'

        $resources = Join-Path $output 'resources\vision-runtime'
        foreach ($relative in @(
            'vision_runtime.dll',
            'rp2350_hid_bridge.dll',
            'resources\vision-runtime\runtime-manifest.json',
            'resources\vision-runtime\model\best.onnx',
            'resources\vision-runtime\model\best.onnx.schema.json',
            'resources\vision-runtime\native\onnxruntime\onnxruntime.dll',
            'resources\vision-runtime\native\onnxruntime\onnxruntime_providers_tensorrt.dll',
            'resources\vision-runtime\native\cuda-11.8\cudart64_110.dll',
            'resources\vision-runtime\native\cudnn-8.9\cudnn64_8.dll',
            'resources\vision-runtime\native\tensorrt-8.6.1.6\nvinfer.dll',
            'resources\vision-runtime\native\msvc-x64\VCRUNTIME140.dll',
            'resources\vision-runtime\config\runtime-sm61.cfg',
            'resources\vision-runtime\licenses\runtime\NOTICE.txt'
        )) {
            Assert-True (Test-Path -LiteralPath (Join-Path $output $relative)) "app-local output is missing $relative"
        }
        foreach ($relative in @(
            'vision_analyzer.exe',
            'resources\vision-runtime\python',
            'resources\vision-runtime\examples',
            'resources\vision-runtime\logs',
            'resources\vision-runtime\cache'
        )) {
            Assert-True (-not (Test-Path -LiteralPath (Join-Path $output $relative))) "app-local output must omit $relative"
        }

        $manifest = Get-Content -LiteralPath (Join-Path $resources 'runtime-manifest.json') -Raw | ConvertFrom-Json
        Assert-Equal 2 $manifest.manifest_version 'app-local manifest version'
        Assert-Equal '0.3.0' $manifest.package_version 'app-local package version'
        Assert-Equal '0.3.0' $manifest.python_sdk.minimum 'minimum Python SDK version'
        Assert-Equal '0.3.0' $manifest.python_sdk.recommended 'recommended Python SDK version'
        Assert-Equal 2 $manifest.dll.abi_major 'runtime ABI major'
        Assert-Equal 1 $manifest.dll.abi_minor 'runtime ABI minor'
        Assert-Equal 31 $manifest.dll.required_features 'runtime required feature flags'
        Assert-Equal (Get-FileSha256 -LiteralPath (Join-Path $output 'vision_runtime.dll')) ([string]$manifest.dll.sha256) 'runtime DLL hash'
        Assert-Equal 'rp2350_hid_bridge.dll' $manifest.hid_bridge.dll.file_name 'HID DLL filename'
        Assert-Equal 1 $manifest.hid_bridge.dll.abi_major 'HID ABI major'
        Assert-Equal 0 $manifest.hid_bridge.dll.abi_minor 'HID ABI minor'
        Assert-Equal '0.2.0' $manifest.hid_bridge.python_sdk.minimum 'minimum HID Python SDK version'
        Assert-Equal '0.2.0' $manifest.hid_bridge.python_sdk.recommended 'recommended HID Python SDK version'
        Assert-Equal (Get-FileSha256 -LiteralPath (Join-Path $output 'rp2350_hid_bridge.dll')) ([string]$manifest.hid_bridge.dll.sha256) 'HID DLL hash'
        Assert-Equal (Get-FileSha256 -LiteralPath (Join-Path $resources 'model\best.onnx')) ([string]$manifest.model.sha256) 'model hash'
        Assert-Equal (Get-FileSha256 -LiteralPath (Join-Path $resources 'model\best.onnx.schema.json')) ([string]$manifest.model.schema_sha256) 'schema hash'
        Assert-Equal '1.17.3' $manifest.components.onnxruntime 'ONNX Runtime component version'
        Assert-Equal '11.8' $manifest.components.cuda 'CUDA component version'
        Assert-Equal '8.9.7' $manifest.components.cudnn 'cuDNN component version'
        Assert-Equal '8.6.1.6' $manifest.components.tensorrt 'TensorRT component version'
        Assert-True ([string]$manifest.runtime_id -match '^sm61-ort1173-trt861-fp32-[0-9A-F]{36}$') 'runtime ID must include stable DLL, HID DLL, and model hashes'
        Assert-Equal 2 @(Get-ChildItem -LiteralPath $output -File -Filter '*.dll').Count 'app-local root must contain exactly two runtime DLLs'

        foreach ($path in @($manifest.model.path, $manifest.model.schema_path) + @($manifest.native_directories)) {
            Assert-True (-not [IO.Path]::IsPathRooted([string]$path)) "manifest path must be relative: $path"
            Assert-True (-not ([string]$path).Contains('..')) "manifest path must not escape resources: $path"
            Assert-True (-not ([string]$path).Contains('\')) "manifest path must use canonical separators: $path"
        }
    }

    Invoke-Test 'app-local builder refuses to replace an unmarked output directory' {
        $builder = Join-Path (Join-Path $PSScriptRoot '..') 'build-app-local-package.ps1'
        $source = Join-Path $testRoot 'app-local-overwrite-source'
        $output = Join-Path $testRoot 'app-local-unmarked-output'
        New-PortablePackageFixture -LiteralPath $source
        New-Item -ItemType Directory -Path $output -Force | Out-Null
        [IO.File]::WriteAllText((Join-Path $output 'owned-by-client.txt'), 'keep')

        Assert-Throws {
            & $builder -PortablePackageRoot $source -OutputRoot $output
        } 'Refusing|unrecognized|marker' 'unmarked client output must never be replaced'
        Assert-True (Test-Path -LiteralPath (Join-Path $output 'owned-by-client.txt')) 'rejected output must remain untouched'
    }

    Invoke-Test 'Python live example has explicit arming and cleanup boundaries' {
        if ([string]::IsNullOrWhiteSpace($PythonProjectRoot)) {
            return
        }
        $resolvedPythonRoot = [IO.Path]::GetFullPath($PythonProjectRoot)
        $example = Join-Path $resolvedPythonRoot 'examples\runtime_live_move.py'
        Assert-True (Test-Path -LiteralPath $example -PathType Leaf) 'Python live example must exist'
        $content = Get-Content -LiteralPath $example -Raw
        foreach ($token in @(
            'calibrate_hid',
            'HidSession',
            'VisionRuntime.from_app_dir',
            'hid_session=hid',
            'with runtime.armed_output',
            'hid.stop_all()',
            'finally',
            '--enable-live-output',
            '--app-dir',
            '--calibration-path',
            '--recalibrate',
            'set_hid_calibration_path',
            'get_hid_calibration'
        )) {
            Assert-True ($content.Contains($token)) "Python live example must contain $token"
        }
        Assert-True (-not $content.Contains('runtime.stop_all()')) 'vision cleanup must not globally release caller input'
        Assert-True (-not $content.Contains('set_hid_port')) 'shared-session example must not open a private HID port'

        $wrapper = Join-Path $resolvedPythonRoot 'src\cs2_vision_runtime\runtime.py'
        Assert-True (Test-Path -LiteralPath $wrapper -PathType Leaf) 'Python runtime wrapper must exist'
        $wrapperContent = Get-Content -LiteralPath $wrapper -Raw
        foreach ($token in @('va_set_hid_calibration_path', 'va_get_hid_calibration')) {
            Assert-True ($wrapperContent.Contains($token)) "Python wrapper must bind $token"
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
