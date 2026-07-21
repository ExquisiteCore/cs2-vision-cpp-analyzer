[CmdletBinding()]
param(
    [string]$ReleaseRoot = '',
    [string]$OrtRoot = '',
    [string]$ModelPath = '',
    [string]$SchemaPath = '',
    [string]$SampleVideoPath = '',
    [string]$DependencyCache = '',
    [string]$OutputRoot = '',
    [string]$OutputZip = '',
    [string]$MsvcRedistRoot = '',
    [string]$TensorRtArchive = '',
    [switch]$DownloadPublicDependencies
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2.0

$profile = 'sm61-ort1173-trt861-fp32'
$scriptRoot = [IO.Path]::GetFullPath($PSScriptRoot)
$repoRoot = [IO.Path]::GetFullPath((Join-Path $scriptRoot '..\..'))
$templateRoot = Join-Path $scriptRoot 'package'
$lockPath = Join-Path $scriptRoot 'dependencies.lock.json'
$modulePath = Join-Path $scriptRoot 'PackageTools.psm1'
$installersRoot = Join-Path $scriptRoot 'installers'
Import-Module $modulePath -Force

function Get-LockComponent {
    param(
        [Parameter(Mandatory)]$Lock,
        [Parameter(Mandatory)][string]$Id
    )
    $matches = @($Lock.components | Where-Object { $_.id -eq $Id })
    if ($matches.Count -ne 1) {
        throw "Dependency lock must contain exactly one '$Id' component."
    }
    $matches[0]
}

function Assert-LeafFile {
    param(
        [Parameter(Mandatory)][string]$LiteralPath,
        [Parameter(Mandatory)][string]$Description
    )
    if (-not (Test-Path -LiteralPath $LiteralPath -PathType Leaf)) {
        throw "$Description does not exist: $LiteralPath"
    }
}

function Remove-VerifiedChildDirectory {
    param(
        [Parameter(Mandatory)][string]$AllowedParent,
        [Parameter(Mandatory)][string]$LiteralPath
    )
    $parent = [IO.Path]::GetFullPath($AllowedParent).TrimEnd('\')
    $target = [IO.Path]::GetFullPath($LiteralPath).TrimEnd('\')
    if ($target -eq $parent -or -not $target.StartsWith($parent + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove directory outside intended parent '$parent': $target"
    }
    if (Test-Path -LiteralPath $target -PathType Container) {
        Remove-Item -LiteralPath $target -Recurse -Force
    }
}

function Get-DefaultOrtRoot {
    $candidates = New-Object Collections.Generic.List[string]
    $known = 'D:\Tool\onnxruntime-win-x64-gpu-1.17.3'
    if (Test-Path -LiteralPath $known -PathType Container) { $candidates.Add($known) }
    if (-not [string]::IsNullOrWhiteSpace($env:ONNXRUNTIME_ROOT) -and (Test-Path -LiteralPath $env:ONNXRUNTIME_ROOT -PathType Container)) {
        $candidates.Add($env:ONNXRUNTIME_ROOT)
    }
    foreach ($candidate in $candidates) {
        $versionPath = Join-Path $candidate 'VERSION_NUMBER'
        if (Test-Path -LiteralPath $versionPath -PathType Leaf) {
            $version = (Get-Content -LiteralPath $versionPath -Raw).Trim()
            if ($version -eq '1.17.3') { return [IO.Path]::GetFullPath($candidate) }
        }
    }
    ''
}

function Copy-FlatFiles {
    param(
        [Parameter(Mandatory)][IO.FileInfo[]]$Files,
        [Parameter(Mandatory)][string]$Destination
    )
    New-Item -ItemType Directory -Path $Destination -Force | Out-Null
    foreach ($file in $Files) {
        Copy-Item -LiteralPath $file.FullName -Destination (Join-Path $Destination $file.Name) -Force
    }
}

function Copy-OrtRuntime {
    param(
        [Parameter(Mandatory)][string]$SourceRoot,
        [Parameter(Mandatory)][string]$AppDestination,
        [Parameter(Mandatory)][string]$LicenseDestination
    )
    $versionPath = Join-Path $SourceRoot 'VERSION_NUMBER'
    Assert-LeafFile -LiteralPath $versionPath -Description 'ONNX Runtime VERSION_NUMBER'
    if ((Get-Content -LiteralPath $versionPath -Raw).Trim() -ne '1.17.3') {
        throw "ONNX Runtime root must be version 1.17.3: $SourceRoot"
    }
    $dlls = @(Get-ChildItem -LiteralPath (Join-Path $SourceRoot 'lib') -File -Filter 'onnxruntime*.dll')
    $required = @('onnxruntime.dll', 'onnxruntime_providers_shared.dll', 'onnxruntime_providers_cuda.dll', 'onnxruntime_providers_tensorrt.dll')
    foreach ($name in $required) {
        if (@($dlls | Where-Object { $_.Name -ieq $name }).Count -ne 1) {
            throw "ONNX Runtime 1.17.3 root is missing $name"
        }
    }
    Copy-FlatFiles -Files $dlls -Destination $AppDestination
    Copy-ComponentLicenses -ExtractedRoot $SourceRoot -DestinationPath $LicenseDestination
}

function Find-ExtractedOrtRoot {
    param([Parameter(Mandatory)][string]$ExtractedRoot)
    $matches = @(
        Get-ChildItem -LiteralPath $ExtractedRoot -Directory -Recurse | Where-Object {
            Test-Path -LiteralPath (Join-Path $_.FullName 'lib\onnxruntime.dll') -PathType Leaf
        }
    )
    if ($matches.Count -ne 1) {
        throw "Expected one ONNX Runtime SDK root in archive, found $($matches.Count)."
    }
    $matches[0].FullName
}

$projectRoot = Find-ProjectAssetRoot -StartPath $repoRoot
if ([string]::IsNullOrWhiteSpace($ReleaseRoot)) { $ReleaseRoot = Join-Path $repoRoot 'build\windows\x64\release' }
if ([string]::IsNullOrWhiteSpace($ModelPath)) { $ModelPath = Join-Path $projectRoot 'runs\detect\train-2\weights\best.onnx' }
if ([string]::IsNullOrWhiteSpace($SchemaPath)) { $SchemaPath = $ModelPath + '.schema.json' }
if ([string]::IsNullOrWhiteSpace($SampleVideoPath)) { $SampleVideoPath = Join-Path $projectRoot 'videos\02.mp4' }
if ([string]::IsNullOrWhiteSpace($DependencyCache)) { $DependencyCache = Join-Path $projectRoot 'dist\.sm61-cache' }
if ([string]::IsNullOrWhiteSpace($OutputRoot)) { $OutputRoot = Join-Path $projectRoot 'dist\cs2-vision-runtime-sm61' }
if ([string]::IsNullOrWhiteSpace($OutputZip)) { $OutputZip = $OutputRoot.TrimEnd('\') + '.zip' }
if ([string]::IsNullOrWhiteSpace($OrtRoot)) { $OrtRoot = Get-DefaultOrtRoot }

$ReleaseRoot = [IO.Path]::GetFullPath($ReleaseRoot)
$ModelPath = [IO.Path]::GetFullPath($ModelPath)
$SchemaPath = [IO.Path]::GetFullPath($SchemaPath)
$SampleVideoPath = [IO.Path]::GetFullPath($SampleVideoPath)
$DependencyCache = [IO.Path]::GetFullPath($DependencyCache)
$OutputRoot = [IO.Path]::GetFullPath($OutputRoot)
$OutputZip = [IO.Path]::GetFullPath($OutputZip)
if (-not [string]::IsNullOrWhiteSpace($OrtRoot)) { $OrtRoot = [IO.Path]::GetFullPath($OrtRoot) }

if (-not (Test-Path -LiteralPath $ReleaseRoot -PathType Container)) {
    throw "Release output directory does not exist: $ReleaseRoot"
}
foreach ($name in @('vision_runtime.dll', 'vision_runtime.lib', 'vision_analyzer.exe')) {
    Assert-LeafFile -LiteralPath (Join-Path $ReleaseRoot $name) -Description "Release output '$name'"
}
Assert-LeafFile -LiteralPath (Join-Path $repoRoot 'include\vision_analyzer\vision_runtime_c_api.h') -Description 'C API header'
Assert-LeafFile -LiteralPath $ModelPath -Description 'Model'
Assert-LeafFile -LiteralPath $SchemaPath -Description 'Model schema'
Assert-LeafFile -LiteralPath $SampleVideoPath -Description 'Sample video'
Assert-LeafFile -LiteralPath $lockPath -Description 'Dependency lock'
if (-not (Test-Path -LiteralPath $templateRoot -PathType Container)) { throw "Package template directory does not exist: $templateRoot" }

$lock = Get-Content -LiteralPath $lockPath -Raw | ConvertFrom-Json
if ($lock.profile -ne $profile) { throw "Dependency lock profile mismatch: $($lock.profile)" }

$workRoot = $null
try {
    New-Item -ItemType Directory -Path $DependencyCache -Force | Out-Null
    $archives = @{}
    foreach ($component in @($lock.components | Where-Object { $_.sourceMode -eq 'public' })) {
        if ($component.id -eq 'onnxruntime-gpu' -and -not [string]::IsNullOrWhiteSpace($OrtRoot)) {
            continue
        }
        Write-Host "Resolving $($component.id) $($component.version)..."
        $archives[$component.id] = Get-VerifiedArchive -Component $component -CacheRoot $DependencyCache -DownloadPublicDependencies:$DownloadPublicDependencies
    }

    $resolvedTensorRtArchive = Resolve-TensorRtArchive -ExplicitPath $TensorRtArchive -InstallersRoot $installersRoot
    $tensorRtArchiveSha256 = Get-FileSha256 -LiteralPath $resolvedTensorRtArchive

    $workRoot = Join-Path $DependencyCache ('work-' + [guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $workRoot -Force | Out-Null

    $expanded = @{}
    foreach ($id in @($archives.Keys | Sort-Object)) {
        $destination = Join-Path $workRoot $id
        Expand-DependencyArchive -ArchivePath $archives[$id] -DestinationPath $destination
        $expanded[$id] = $destination
    }
    $tensorRtExpanded = Join-Path $workRoot 'tensorrt'
    Expand-DependencyArchive -ArchivePath $resolvedTensorRtArchive -DestinationPath $tensorRtExpanded
    [void](Test-TensorRtArchiveLayout -ExtractedRoot $tensorRtExpanded)

    $outputParent = Split-Path -Parent $OutputRoot
    New-Item -ItemType Directory -Path $outputParent -Force | Out-Null
    if (Test-Path -LiteralPath $OutputRoot -PathType Container) {
        $marker = Join-Path $OutputRoot '.portable-package-root'
        $oldManifest = Join-Path $OutputRoot 'runtime-manifest.json'
        $knownOutput = Test-Path -LiteralPath $marker -PathType Leaf
        if (-not $knownOutput -and (Test-Path -LiteralPath $oldManifest -PathType Leaf)) {
            try {
                $oldProfile = (Get-Content -LiteralPath $oldManifest -Raw | ConvertFrom-Json).profile
                $knownOutput = ($oldProfile -eq $profile)
            } catch { $knownOutput = $false }
        }
        if (-not $knownOutput) {
            throw "Refusing to replace unrecognized output directory: $OutputRoot"
        }
        Remove-VerifiedChildDirectory -AllowedParent $outputParent -LiteralPath $OutputRoot
    }
    New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null
    [IO.File]::WriteAllText((Join-Path $OutputRoot '.portable-package-root'), $profile, (New-Object Text.UTF8Encoding($false)))
    Copy-Item -Path (Join-Path $templateRoot '*') -Destination $OutputRoot -Recurse -Force

    foreach ($relative in @('app', 'model', 'samples', 'licenses', 'runtime\cuda-11.8', 'runtime\cudnn-8.9', 'runtime\tensorrt-8.6.1.6', 'runtime\msvc-x64', 'logs', 'cache\ort-trt-sm61-fp32')) {
        New-Item -ItemType Directory -Path (Join-Path $OutputRoot $relative) -Force | Out-Null
    }

    $app = Join-Path $OutputRoot 'app'
    foreach ($name in @('vision_runtime.dll', 'vision_runtime.lib', 'vision_analyzer.exe')) {
        Copy-Item -LiteralPath (Join-Path $ReleaseRoot $name) -Destination (Join-Path $app $name) -Force
    }
    Copy-Item -LiteralPath (Join-Path $repoRoot 'include\vision_analyzer\vision_runtime_c_api.h') -Destination (Join-Path $app 'vision_runtime_c_api.h') -Force

    if ([string]::IsNullOrWhiteSpace($OrtRoot)) {
        $OrtRoot = Find-ExtractedOrtRoot -ExtractedRoot $expanded['onnxruntime-gpu']
    }
    Copy-OrtRuntime -SourceRoot $OrtRoot -AppDestination $app -LicenseDestination (Join-Path $OutputRoot 'licenses\onnxruntime')

    $cudaDestination = Join-Path $OutputRoot 'runtime\cuda-11.8'
    foreach ($id in @('cuda-cudart', 'cuda-cublas', 'cuda-cufft', 'cuda-nvrtc')) {
        Copy-ComponentRuntimeFiles -ExtractedRoot $expanded[$id] -DestinationPath $cudaDestination -Layout 'bin'
        Copy-ComponentLicenses -ExtractedRoot $expanded[$id] -DestinationPath (Join-Path $OutputRoot ("licenses\cuda\$id"))
    }

    Copy-ComponentRuntimeFiles -ExtractedRoot $expanded['cudnn'] -DestinationPath (Join-Path $OutputRoot 'runtime\cudnn-8.9') -Layout 'bin'
    Copy-ComponentLicenses -ExtractedRoot $expanded['cudnn'] -DestinationPath (Join-Path $OutputRoot 'licenses\cudnn')

    Copy-ComponentRuntimeFiles -ExtractedRoot $tensorRtExpanded -DestinationPath (Join-Path $OutputRoot 'runtime\tensorrt-8.6.1.6') -Layout 'lib'
    Copy-ComponentLicenses -ExtractedRoot $tensorRtExpanded -DestinationPath (Join-Path $OutputRoot 'licenses\tensorrt')

    if ([string]::IsNullOrWhiteSpace($MsvcRedistRoot)) {
        $msvcSearchRoots = @()
        foreach ($programFilesRoot in @($env:ProgramFiles, ${env:ProgramFiles(x86)})) {
            if (-not [string]::IsNullOrWhiteSpace($programFilesRoot)) {
                $msvcSearchRoots += Join-Path $programFilesRoot 'Microsoft Visual Studio'
            }
        }
        $MsvcRedistRoot = Find-MsvcPrivateRuntimeRoot -SearchRoots @($msvcSearchRoots | Select-Object -Unique)
    }
    $MsvcRedistRoot = [IO.Path]::GetFullPath($MsvcRedistRoot)
    $msvcDlls = @(Get-ChildItem -LiteralPath $MsvcRedistRoot -File -Filter '*.dll')
    $msvcDestination = Join-Path $OutputRoot 'runtime\msvc-x64'
    Copy-FlatFiles -Files $msvcDlls -Destination $msvcDestination
    foreach ($name in @('MSVCP140.dll', 'VCRUNTIME140.dll', 'VCRUNTIME140_1.dll', 'CONCRT140.dll')) {
        Assert-LeafFile -LiteralPath (Join-Path $msvcDestination $name) -Description "MSVC runtime '$name'"
    }
    $msvcLicense = Join-Path $OutputRoot 'licenses\msvc'
    New-Item -ItemType Directory -Path $msvcLicense -Force | Out-Null
    foreach ($manifestFile in @(Get-ChildItem -LiteralPath $MsvcRedistRoot -File -Filter '*.manifest')) {
        Copy-Item -LiteralPath $manifestFile.FullName -Destination (Join-Path $msvcLicense $manifestFile.Name) -Force
    }
    $sourceText = "Microsoft MSVC v14-compatible CRT private deployment files copied from:`r`n$MsvcRedistRoot`r`nGoverned by the installed Visual Studio license and redistribution terms.`r`n"
    [IO.File]::WriteAllText((Join-Path $msvcLicense 'SOURCE.txt'), $sourceText, (New-Object Text.UTF8Encoding($false)))

    Copy-Item -LiteralPath $ModelPath -Destination (Join-Path $OutputRoot 'model\best.onnx') -Force
    Copy-Item -LiteralPath $SchemaPath -Destination (Join-Path $OutputRoot 'model\best.onnx.schema.json') -Force

    $ffmpeg = Get-Command 'ffmpeg.exe' -ErrorAction SilentlyContinue
    if ($null -eq $ffmpeg) { $ffmpeg = Get-Command 'ffmpeg' -ErrorAction SilentlyContinue }
    if ($null -eq $ffmpeg) { throw 'ffmpeg is required to create the five-second smoke-test video.' }
    $smokeVideo = Join-Path $OutputRoot 'samples\smoke-test.mp4'
    & $ffmpeg.Source '-hide_banner' '-loglevel' 'error' '-y' '-ss' '0' '-i' $SampleVideoPath '-t' '5' '-an' '-c:v' 'copy' $smokeVideo
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $smokeVideo -PathType Leaf) -or (Get-Item -LiteralPath $smokeVideo).Length -eq 0) {
        throw "ffmpeg failed to create smoke-test video from $SampleVideoPath"
    }

    Assert-CompatibleRuntimeFiles -PackageRoot $OutputRoot
    foreach ($relative in @(
        'app\onnxruntime.dll',
        'app\onnxruntime_providers_shared.dll',
        'app\onnxruntime_providers_cuda.dll',
        'app\onnxruntime_providers_tensorrt.dll',
        'runtime\cuda-11.8\cudart64_110.dll',
        'runtime\cuda-11.8\cublas64_11.dll',
        'runtime\cuda-11.8\cublasLt64_11.dll',
        'runtime\cuda-11.8\cufft64_10.dll',
        'runtime\cudnn-8.9\cudnn64_8.dll',
        'runtime\tensorrt-8.6.1.6\nvinfer.dll',
        'runtime\tensorrt-8.6.1.6\nvinfer_plugin.dll',
        'runtime\tensorrt-8.6.1.6\nvonnxparser.dll',
        'runtime\tensorrt-8.6.1.6\nvinfer_builder_resource.dll'
    )) {
        Assert-LeafFile -LiteralPath (Join-Path $OutputRoot $relative) -Description "Packaged runtime '$relative'"
    }

    $manifestComponents = @()
    foreach ($component in @($lock.components)) {
        $copy = $component | ConvertTo-Json -Depth 8 | ConvertFrom-Json
        if ($copy.id -eq 'tensorrt') {
            $copy | Add-Member -NotePropertyName archiveSha256 -NotePropertyValue $tensorRtArchiveSha256
        }
        if ($copy.id -eq 'msvc-crt') {
            $msvcp = Get-Item -LiteralPath (Join-Path $msvcDestination 'MSVCP140.dll')
            $copy | Add-Member -NotePropertyName actualFileVersion -NotePropertyValue $msvcp.VersionInfo.FileVersion
        }
        $manifestComponents += $copy
    }
    Write-PackageManifest -PackageRoot $OutputRoot -Profile $profile -Components @($manifestComponents)
    $manifestResult = Test-PackageManifest -PackageRoot $OutputRoot
    if (-not $manifestResult.Valid) {
        throw "Generated package manifest failed validation."
    }

    $testScript = Join-Path $scriptRoot 'tests\run-tests.ps1'
    & powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File $testScript
    if ($LASTEXITCODE -ne 0) { throw 'Packaging unit and safety tests failed.' }

    $tar = Get-Command 'tar.exe' -ErrorAction SilentlyContinue
    if ($null -eq $tar) { throw 'tar.exe is required to create a Zip64-compatible package archive.' }
    $zipParent = Split-Path -Parent $OutputZip
    New-Item -ItemType Directory -Path $zipParent -Force | Out-Null
    if (Test-Path -LiteralPath $OutputZip -PathType Leaf) { Remove-Item -LiteralPath $OutputZip -Force }
    & $tar.Source '-a' '-c' '-f' $OutputZip '-C' $outputParent (Split-Path -Leaf $OutputRoot)
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $OutputZip -PathType Leaf)) {
        throw "Failed to create package ZIP: $OutputZip"
    }

    $zipItem = Get-Item -LiteralPath $OutputZip
    $zipSha256 = Get-FileSha256 -LiteralPath $OutputZip
    Write-Host "Package complete: $OutputZip"
    Write-Host "ZIP bytes: $($zipItem.Length)"
    Write-Host "ZIP SHA256: $zipSha256"
    [pscustomobject]@{
        Profile = $profile
        StagingRoot = $OutputRoot
        ZipPath = $OutputZip
        ZipBytes = $zipItem.Length
        ZipSha256 = $zipSha256
        TensorRtArchiveSha256 = $tensorRtArchiveSha256
    }
} catch {
    Write-Error $_.Exception.Message
    exit 1
} finally {
    if (-not [string]::IsNullOrWhiteSpace($workRoot) -and (Test-Path -LiteralPath $workRoot -PathType Container)) {
        Remove-VerifiedChildDirectory -AllowedParent $DependencyCache -LiteralPath $workRoot
    }
}
