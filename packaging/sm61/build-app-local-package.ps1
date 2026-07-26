[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string] $PortablePackageRoot,

    [Parameter(Mandatory = $true)]
    [string] $OutputRoot,

    [string] $PythonSdkVersion = '0.3.0'
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2.0

$profileName = 'sm61-ort1173-trt861-fp32'
$scriptRoot = [IO.Path]::GetFullPath($PSScriptRoot)
$modulePath = Join-Path $scriptRoot 'PackageTools.psm1'
Import-Module $modulePath -Force

function Assert-LeafFile {
    param(
        [Parameter(Mandatory)][string] $LiteralPath,
        [Parameter(Mandatory)][string] $Description
    )
    if (-not (Test-Path -LiteralPath $LiteralPath -PathType Leaf)) {
        throw "$Description is missing: $LiteralPath"
    }
}

function Assert-Directory {
    param(
        [Parameter(Mandatory)][string] $LiteralPath,
        [Parameter(Mandatory)][string] $Description
    )
    if (-not (Test-Path -LiteralPath $LiteralPath -PathType Container)) {
        throw "$Description is missing: $LiteralPath"
    }
}

function Remove-VerifiedChildDirectory {
    param(
        [Parameter(Mandatory)][string] $AllowedParent,
        [Parameter(Mandatory)][string] $LiteralPath
    )
    $parent = [IO.Path]::GetFullPath($AllowedParent).TrimEnd('\')
    $target = [IO.Path]::GetFullPath($LiteralPath).TrimEnd('\')
    if (
        $target -eq $parent -or
        -not $target.StartsWith(
            $parent + '\',
            [StringComparison]::OrdinalIgnoreCase
        )
    ) {
        throw "Refusing to remove directory outside intended parent '$parent': $target"
    }
    if (Test-Path -LiteralPath $target -PathType Container) {
        Remove-Item -LiteralPath $target -Recurse -Force
    }
}

function Copy-DirectoryContents {
    param(
        [Parameter(Mandatory)][string] $Source,
        [Parameter(Mandatory)][string] $Destination,
        [Parameter(Mandatory)][string] $Description
    )
    Assert-Directory -LiteralPath $Source -Description $Description
    New-Item -ItemType Directory -Path $Destination -Force | Out-Null
    foreach ($item in @(Get-ChildItem -LiteralPath $Source -Force)) {
        Copy-Item -LiteralPath $item.FullName -Destination $Destination -Recurse -Force
    }
}

function Get-ComponentVersion {
    param(
        [Parameter(Mandatory)] $Manifest,
        [Parameter(Mandatory)][string] $Id
    )
    $matches = @($Manifest.components | Where-Object { [string]$_.id -eq $Id })
    if ($matches.Count -ne 1) {
        throw "Verified portable manifest must contain exactly one '$Id' component."
    }
    [string]$matches[0].version
}

if ($PythonSdkVersion -notmatch '^\d+\.\d+\.\d+$') {
    throw "PythonSdkVersion must use numeric major.minor.patch format: $PythonSdkVersion"
}

$sourceRoot = [IO.Path]::GetFullPath($PortablePackageRoot).TrimEnd('\')
$outputFullPath = [IO.Path]::GetFullPath($OutputRoot).TrimEnd('\')
Assert-Directory -LiteralPath $sourceRoot -Description 'Portable package root'

$sourceManifestPath = Join-Path $sourceRoot 'runtime-manifest.json'
Assert-LeafFile -LiteralPath $sourceManifestPath -Description 'Portable package manifest'
$sourceValidation = Test-PackageManifest -PackageRoot $sourceRoot
if (-not $sourceValidation.Valid) {
    throw (
        'Portable package manifest validation failed; ' +
        "missing=$(@($sourceValidation.Missing) -join ','), " +
        "changed=$(@($sourceValidation.Changed) -join ','), " +
        "unexpected=$(@($sourceValidation.Unexpected) -join ',')"
    )
}
if ($sourceValidation.Profile -ne $profileName) {
    throw "Portable package profile must be $profileName; found $($sourceValidation.Profile)"
}

$sourceManifest = $sourceValidation.Manifest
$onnxRuntimeVersion = Get-ComponentVersion -Manifest $sourceManifest -Id 'onnxruntime-gpu'
$cudnnVersion = Get-ComponentVersion -Manifest $sourceManifest -Id 'cudnn'
$tensorRtVersion = Get-ComponentVersion -Manifest $sourceManifest -Id 'tensorrt'
$msvcVersion = Get-ComponentVersion -Manifest $sourceManifest -Id 'msvc-crt'
$cudaVersion = Get-ComponentVersion -Manifest $sourceManifest -Id 'cuda-cudart'
if ($onnxRuntimeVersion -ne '1.17.3') {
    throw "Portable package ONNX Runtime must be 1.17.3; found $onnxRuntimeVersion"
}
if ($cudaVersion -notmatch '^11\.8(?:\.|$)') {
    throw "Portable package CUDA runtime must be 11.8.x; found $cudaVersion"
}
if ($cudnnVersion -notmatch '^8\.9(?:\.|$)') {
    throw "Portable package cuDNN must be 8.9.x; found $cudnnVersion"
}
if ($tensorRtVersion -ne '8.6.1.6') {
    throw "Portable package TensorRT must be 8.6.1.6; found $tensorRtVersion"
}

$sourcePrefix = $sourceRoot + '\'
$outputPrefix = $outputFullPath + '\'
if (
    $sourceRoot -eq $outputFullPath -or
    $outputFullPath.StartsWith($sourcePrefix, [StringComparison]::OrdinalIgnoreCase) -or
    $sourceRoot.StartsWith($outputPrefix, [StringComparison]::OrdinalIgnoreCase)
) {
    throw 'PortablePackageRoot and OutputRoot must be separate, non-nested directories.'
}

$outputParent = Split-Path -Parent $outputFullPath
if ([string]::IsNullOrWhiteSpace($outputParent)) {
    throw "OutputRoot must have a parent directory: $outputFullPath"
}
New-Item -ItemType Directory -Path $outputParent -Force | Out-Null

if (Test-Path -LiteralPath $outputFullPath) {
    if (-not (Test-Path -LiteralPath $outputFullPath -PathType Container)) {
        throw "Refusing to replace a non-directory output path: $outputFullPath"
    }
    $marker = Join-Path $outputFullPath '.app-local-runtime-root'
    if (-not (Test-Path -LiteralPath $marker -PathType Leaf)) {
        throw "Refusing to replace unrecognized output directory without app-local marker: $outputFullPath"
    }
}

$sourceDll = Join-Path $sourceRoot 'app\vision_runtime.dll'
$sourceHidDll = Join-Path $sourceRoot 'app\rp2350_hid_bridge.dll'
$sourceModel = Join-Path $sourceRoot 'model\best.onnx'
$sourceSchema = Join-Path $sourceRoot 'model\best.onnx.schema.json'
Assert-LeafFile -LiteralPath $sourceDll -Description 'Portable vision runtime DLL'
Assert-LeafFile -LiteralPath $sourceHidDll -Description 'Portable RP2350 HID bridge DLL'
Assert-LeafFile -LiteralPath $sourceModel -Description 'Portable model'
Assert-LeafFile -LiteralPath $sourceSchema -Description 'Portable model schema'

$sourceOrt = Join-Path $sourceRoot 'app'
$ortDlls = @(Get-ChildItem -LiteralPath $sourceOrt -File -Filter 'onnxruntime*.dll')
foreach ($name in @(
    'onnxruntime.dll',
    'onnxruntime_providers_shared.dll',
    'onnxruntime_providers_cuda.dll',
    'onnxruntime_providers_tensorrt.dll'
)) {
    if (@($ortDlls | Where-Object { $_.Name -ieq $name }).Count -ne 1) {
        throw "Portable package is missing ONNX Runtime DLL: $name"
    }
}

$stageRoot = Join-Path $outputParent (
    '.' + (Split-Path -Leaf $outputFullPath) +
    '.app-local-staging-' + [guid]::NewGuid().ToString('N')
)
New-Item -ItemType Directory -Path $stageRoot -Force | Out-Null

try {
    $resourcesRoot = Join-Path $stageRoot 'resources\vision-runtime'
    $modelRoot = Join-Path $resourcesRoot 'model'
    $nativeRoot = Join-Path $resourcesRoot 'native'
    New-Item -ItemType Directory -Path $modelRoot -Force | Out-Null
    New-Item -ItemType Directory -Path $nativeRoot -Force | Out-Null

    Copy-Item -LiteralPath $sourceDll -Destination (Join-Path $stageRoot 'vision_runtime.dll')
    Copy-Item -LiteralPath $sourceHidDll -Destination (Join-Path $stageRoot 'rp2350_hid_bridge.dll')
    Copy-Item -LiteralPath $sourceModel -Destination (Join-Path $modelRoot 'best.onnx')
    Copy-Item -LiteralPath $sourceSchema -Destination (Join-Path $modelRoot 'best.onnx.schema.json')

    $ortDestination = Join-Path $nativeRoot 'onnxruntime'
    New-Item -ItemType Directory -Path $ortDestination -Force | Out-Null
    foreach ($file in $ortDlls) {
        Copy-Item -LiteralPath $file.FullName -Destination (Join-Path $ortDestination $file.Name)
    }

    foreach ($mapping in @(
        [pscustomobject]@{ Source = 'runtime\cuda-11.8'; Destination = 'native\cuda-11.8'; Description = 'CUDA 11.8 runtime' },
        [pscustomobject]@{ Source = 'runtime\cudnn-8.9'; Destination = 'native\cudnn-8.9'; Description = 'cuDNN 8.9 runtime' },
        [pscustomobject]@{ Source = 'runtime\tensorrt-8.6.1.6'; Destination = 'native\tensorrt-8.6.1.6'; Description = 'TensorRT 8.6.1.6 runtime' },
        [pscustomobject]@{ Source = 'runtime\msvc-x64'; Destination = 'native\msvc-x64'; Description = 'MSVC x64 runtime' },
        [pscustomobject]@{ Source = 'config'; Destination = 'config'; Description = 'runtime configuration' },
        [pscustomobject]@{ Source = 'licenses'; Destination = 'licenses'; Description = 'runtime licenses' }
    )) {
        Copy-DirectoryContents `
            -Source (Join-Path $sourceRoot $mapping.Source) `
            -Destination (Join-Path $resourcesRoot $mapping.Destination) `
            -Description $mapping.Description
    }

    $dllHash = Get-FileSha256 -LiteralPath (Join-Path $stageRoot 'vision_runtime.dll')
    $hidDllHash = Get-FileSha256 -LiteralPath (Join-Path $stageRoot 'rp2350_hid_bridge.dll')
    $modelHash = Get-FileSha256 -LiteralPath (Join-Path $modelRoot 'best.onnx')
    $schemaHash = Get-FileSha256 -LiteralPath (Join-Path $modelRoot 'best.onnx.schema.json')
    $runtimeId = "$profileName-$($dllHash.Substring(0, 12))$($hidDllHash.Substring(0, 12))$($modelHash.Substring(0, 12))"

    $manifest = [pscustomobject][ordered]@{
        manifest_version = 2
        package_version = $PythonSdkVersion
        runtime_id = $runtimeId
        profile = [pscustomobject][ordered]@{
            os = 'windows'
            arch = 'x86_64'
            gpu_sm = 61
            precision = 'fp32'
        }
        python_sdk = [pscustomobject][ordered]@{
            minimum = $PythonSdkVersion
            recommended = $PythonSdkVersion
        }
        backend = 'ort-tensorrt'
        dll = [pscustomobject][ordered]@{
            file_name = 'vision_runtime.dll'
            sha256 = $dllHash
            abi_major = 2
            abi_minor = 1
            required_features = 31
        }
        hid_bridge = [pscustomobject][ordered]@{
            dll = [pscustomobject][ordered]@{
                file_name = 'rp2350_hid_bridge.dll'
                sha256 = $hidDllHash
                abi_major = 1
                abi_minor = 0
            }
            python_sdk = [pscustomobject][ordered]@{
                minimum = '0.2.0'
                recommended = '0.2.0'
            }
        }
        model = [pscustomobject][ordered]@{
            path = 'model/best.onnx'
            sha256 = $modelHash
            schema_path = 'model/best.onnx.schema.json'
            schema_sha256 = $schemaHash
            class_order_source = 'model/best.onnx.schema.json'
        }
        native_directories = @(
            'native/onnxruntime',
            'native/tensorrt-8.6.1.6',
            'native/cudnn-8.9',
            'native/cuda-11.8',
            'native/msvc-x64'
        )
        components = [pscustomobject][ordered]@{
            onnxruntime = $onnxRuntimeVersion
            cuda = '11.8'
            cudnn = $cudnnVersion
            tensorrt = $tensorRtVersion
            msvc = $msvcVersion
        }
        licenses_path = 'licenses'
        source = [pscustomobject][ordered]@{
            portable_profile = $sourceValidation.Profile
            portable_manifest_sha256 = Get-FileSha256 -LiteralPath $sourceManifestPath
        }
    }
    $manifestJson = $manifest | ConvertTo-Json -Depth 10
    [IO.File]::WriteAllText(
        (Join-Path $resourcesRoot 'runtime-manifest.json'),
        $manifestJson + [Environment]::NewLine,
        (New-Object Text.UTF8Encoding($false))
    )
    [IO.File]::WriteAllText(
        (Join-Path $stageRoot '.app-local-runtime-root'),
        $runtimeId + [Environment]::NewLine,
        (New-Object Text.UTF8Encoding($false))
    )

    if (Test-Path -LiteralPath $outputFullPath -PathType Container) {
        Remove-VerifiedChildDirectory -AllowedParent $outputParent -LiteralPath $outputFullPath
    }
    Move-Item -LiteralPath $stageRoot -Destination $outputFullPath
    $stageRoot = ''

    Write-Host "app_local_runtime runtime_id=$runtimeId output=$outputFullPath"
    [pscustomobject]@{
        OutputRoot = $outputFullPath
        RuntimeId = $runtimeId
        ManifestPath = Join-Path $outputFullPath 'resources\vision-runtime\runtime-manifest.json'
    }
}
finally {
    if (
        -not [string]::IsNullOrWhiteSpace($stageRoot) -and
        (Test-Path -LiteralPath $stageRoot -PathType Container)
    ) {
        Remove-VerifiedChildDirectory -AllowedParent $outputParent -LiteralPath $stageRoot
    }
}
