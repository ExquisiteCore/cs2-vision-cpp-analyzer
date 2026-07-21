Set-StrictMode -Version 2.0

function Get-FullPath {
    param([Parameter(Mandatory)][string]$LiteralPath)
    [IO.Path]::GetFullPath($LiteralPath).TrimEnd('\')
}

function Get-RelativePackagePath {
    param(
        [Parameter(Mandatory)][string]$PackageRoot,
        [Parameter(Mandatory)][string]$LiteralPath
    )

    $root = Get-FullPath -LiteralPath $PackageRoot
    $full = Get-FullPath -LiteralPath $LiteralPath
    if (-not $full.StartsWith($root + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "Path is outside package root: $full"
    }
    $full.Substring($root.Length + 1).Replace('\', '/')
}

function Get-FileSha256 {
    param([Parameter(Mandatory)][string]$LiteralPath)
    (Get-FileHash -LiteralPath $LiteralPath -Algorithm SHA256).Hash.ToUpperInvariant()
}

function Get-ImmutablePackageFiles {
    param([Parameter(Mandatory)][string]$PackageRoot)

    $root = Get-FullPath -LiteralPath $PackageRoot
    if (-not (Test-Path -LiteralPath $root -PathType Container)) {
        throw "Package root does not exist: $root"
    }

    Get-ChildItem -LiteralPath $root -File -Recurse | Where-Object {
        $relative = Get-RelativePackagePath -PackageRoot $root -LiteralPath $_.FullName
        $relative -ne 'runtime-manifest.json' -and
        $relative -notlike 'logs/*' -and
        $relative -notlike 'cache/*'
    } | Sort-Object FullName
}

function Write-PackageManifest {
    param(
        [Parameter(Mandatory)][string]$PackageRoot,
        [Parameter(Mandatory)][string]$Profile,
        [Parameter(Mandatory)][AllowEmptyCollection()][object[]]$Components
    )

    $root = Get-FullPath -LiteralPath $PackageRoot
    $files = @(
        Get-ImmutablePackageFiles -PackageRoot $root | ForEach-Object {
            [pscustomobject][ordered]@{
                path = Get-RelativePackagePath -PackageRoot $root -LiteralPath $_.FullName
                size = [int64]$_.Length
                sha256 = Get-FileSha256 -LiteralPath $_.FullName
            }
        }
    )

    $manifest = [pscustomobject][ordered]@{
        schemaVersion = 1
        profile = $Profile
        createdUtc = [DateTime]::UtcNow.ToString('o')
        components = @($Components)
        files = $files
    }
    $json = $manifest | ConvertTo-Json -Depth 10
    $manifestPath = Join-Path $root 'runtime-manifest.json'
    [IO.File]::WriteAllText($manifestPath, $json + [Environment]::NewLine, (New-Object Text.UTF8Encoding($false)))
}

function Test-PackageManifest {
    param([Parameter(Mandatory)][string]$PackageRoot)

    $root = Get-FullPath -LiteralPath $PackageRoot
    $manifestPath = Join-Path $root 'runtime-manifest.json'
    if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
        throw "Package manifest is missing: $manifestPath"
    }

    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    if ($manifest.schemaVersion -ne 1) {
        throw "Unsupported package manifest schema: $($manifest.schemaVersion)"
    }

    $expected = @{}
    foreach ($entry in @($manifest.files)) {
        $path = [string]$entry.path
        if ([string]::IsNullOrWhiteSpace($path) -or $path.Contains('..') -or $path.StartsWith('/')) {
            throw "Unsafe manifest path: $path"
        }
        if ($expected.ContainsKey($path)) {
            throw "Duplicate manifest path: $path"
        }
        $expected[$path] = $entry
    }

    $missing = New-Object Collections.Generic.List[string]
    $changed = New-Object Collections.Generic.List[string]
    foreach ($path in @($expected.Keys | Sort-Object)) {
        $entry = $expected[$path]
        $full = Join-Path $root $path.Replace('/', '\')
        if (-not (Test-Path -LiteralPath $full -PathType Leaf)) {
            $missing.Add($path)
            continue
        }
        $item = Get-Item -LiteralPath $full
        $actualHash = Get-FileSha256 -LiteralPath $full
        if ([int64]$item.Length -ne [int64]$entry.size -or $actualHash -ne [string]$entry.sha256) {
            $changed.Add($path)
        }
    }

    $unexpected = New-Object Collections.Generic.List[string]
    foreach ($file in @(Get-ImmutablePackageFiles -PackageRoot $root)) {
        $path = Get-RelativePackagePath -PackageRoot $root -LiteralPath $file.FullName
        if (-not $expected.ContainsKey($path)) {
            $unexpected.Add($path)
        }
    }

    [pscustomobject]@{
        Valid = ($missing.Count -eq 0 -and $changed.Count -eq 0 -and $unexpected.Count -eq 0)
        Profile = [string]$manifest.profile
        Missing = @($missing)
        Changed = @($changed)
        Unexpected = @($unexpected)
        Manifest = $manifest
    }
}

function Assert-CompatibleRuntimeFiles {
    param([Parameter(Mandatory)][string]$PackageRoot)

    $root = Get-FullPath -LiteralPath $PackageRoot
    foreach ($file in @(Get-ChildItem -LiteralPath $root -File -Recurse -Filter '*.dll')) {
        $name = $file.Name.ToLowerInvariant()
        $relative = Get-RelativePackagePath -PackageRoot $root -LiteralPath $file.FullName

        $isCuda12 =
            $name -match '^cublas(lt)?64_12\.dll$' -or
            $name -match '^cudart64_12\d*\.dll$' -or
            $name -match '^cusparse64_12\.dll$' -or
            $name -match '^cufft64_11\.dll$' -or
            $name -match '^nvrtc64_12\d_0\.dll$' -or
            $name -match '^nvrtc-builtins64_12\d\.dll$'
        if ($isCuda12) {
            throw "CUDA 12 DLL is incompatible with the SM61 profile: $relative"
        }
        if ($name -match '^cudnn.*64_9\.dll$') {
            throw "cuDNN 9 DLL is incompatible with the SM61 profile: $relative"
        }
        if ($name -match '^(nvinfer|nvonnxparser|nvparsers).*_(10|11)\.dll$') {
            throw "TensorRT 10/11 DLL is incompatible with the SM61 profile: $relative"
        }
        if ($name -match '^onnxruntime_providers_.*\.dll$' -and $relative -notlike 'app/*') {
            throw "ONNX Runtime provider DLL must be under app/: $relative"
        }
    }
}

function Get-TensorRtMacroValue {
    param(
        [Parameter(Mandatory)][string]$Header,
        [Parameter(Mandatory)][string]$Name
    )
    $escapedName = [regex]::Escape($Name)
    $match = [regex]::Match($Header, "(?m)^\s*#define\s+$escapedName\s+(\d+)\b[^\r\n]*\r?$")
    if (-not $match.Success) {
        throw "TensorRT version macro is missing: $Name"
    }
    [int]$match.Groups[1].Value
}

function Test-TensorRtArchiveLayout {
    param([Parameter(Mandatory)][string]$ExtractedRoot)

    $root = Get-FullPath -LiteralPath $ExtractedRoot
    $headers = @(Get-ChildItem -LiteralPath $root -File -Recurse -Filter 'NvInferVersion.h')
    if ($headers.Count -ne 1) {
        throw "Expected one NvInferVersion.h in TensorRT archive, found $($headers.Count)."
    }

    $content = Get-Content -LiteralPath $headers[0].FullName -Raw
    $major = Get-TensorRtMacroValue -Header $content -Name 'NV_TENSORRT_MAJOR'
    $minor = Get-TensorRtMacroValue -Header $content -Name 'NV_TENSORRT_MINOR'
    $patch = Get-TensorRtMacroValue -Header $content -Name 'NV_TENSORRT_PATCH'
    $build = Get-TensorRtMacroValue -Header $content -Name 'NV_TENSORRT_BUILD'
    $version = "$major.$minor.$patch.$build"
    if ($version -ne '8.6.1.6') {
        throw "TensorRT archive must be version 8.6.1.6; found $version."
    }

    $required = @('nvinfer.dll', 'nvinfer_plugin.dll', 'nvonnxparser.dll', 'nvinfer_builder_resource.dll')
    $dlls = @{}
    foreach ($file in @(Get-ChildItem -LiteralPath $root -File -Recurse -Filter '*.dll')) {
        $dlls[$file.Name.ToLowerInvariant()] = $file.FullName
    }
    $missing = @($required | Where-Object { -not $dlls.ContainsKey($_) })
    if ($missing.Count -gt 0) {
        throw "TensorRT archive is missing required DLL(s): $($missing -join ', ')"
    }

    [pscustomobject]@{
        Valid = $true
        Version = $version
        HeaderPath = $headers[0].FullName
        RequiredDlls = @($required | ForEach-Object { $dlls[$_] })
    }
}

function Get-RuntimePathEntries {
    param([Parameter(Mandatory)][string]$PackageRoot)

    $root = Get-FullPath -LiteralPath $PackageRoot
    @(
        (Join-Path $root 'app')
        (Join-Path $root 'runtime\tensorrt-8.6.1.6')
        (Join-Path $root 'runtime\cudnn-8.9')
        (Join-Path $root 'runtime\cuda-11.8')
        (Join-Path $root 'runtime\msvc-x64')
    )
}

function Find-ProjectAssetRoot {
    param([Parameter(Mandatory)][string]$StartPath)

    $current = [IO.Path]::GetFullPath($StartPath).TrimEnd('\')
    if (Test-Path -LiteralPath $current -PathType Leaf) {
        $current = Split-Path -Parent $current
    }
    while (-not [string]::IsNullOrWhiteSpace($current)) {
        if ((Test-Path -LiteralPath (Join-Path $current 'runs') -PathType Container) -and
            (Test-Path -LiteralPath (Join-Path $current 'videos') -PathType Container)) {
            return $current
        }
        $parent = [IO.Directory]::GetParent($current)
        if ($null -eq $parent -or $parent.FullName -eq $current) { break }
        $current = $parent.FullName.TrimEnd('\')
    }
    throw "Could not find a project asset root containing runs/ and videos/ above: $StartPath"
}

function Get-VerifiedArchive {
    param(
        [Parameter(Mandatory)]$Component,
        [Parameter(Mandatory)][string]$CacheRoot,
        [switch]$DownloadPublicDependencies
    )

    $cache = [IO.Path]::GetFullPath($CacheRoot)
    New-Item -ItemType Directory -Path $cache -Force | Out-Null
    $archivePath = Join-Path $cache ([string]$Component.archive)
    $expectedHash = [string]$Component.sha256

    if (Test-Path -LiteralPath $archivePath -PathType Leaf) {
        $actualHash = Get-FileSha256 -LiteralPath $archivePath
        if ($actualHash -eq $expectedHash) {
            return $archivePath
        }

        $quarantine = $archivePath + '.bad-' + [DateTime]::Now.ToString('yyyyMMddHHmmssfff')
        Move-Item -LiteralPath $archivePath -Destination $quarantine
        if (-not $DownloadPublicDependencies) {
            throw "Cached archive SHA256 mismatch for $($Component.id); quarantined as $quarantine. Enable download to reacquire it."
        }
    }

    if (-not $DownloadPublicDependencies) {
        throw "Verified archive is missing for $($Component.id): $archivePath. Enable download to acquire it."
    }
    if ([string]$Component.sourceMode -ne 'public') {
        throw "Automatic download is allowed only for public locked dependencies: $($Component.id)"
    }

    $curl = Get-Command 'curl.exe' -ErrorAction SilentlyContinue
    if ($null -eq $curl) {
        throw 'curl.exe is required for resumable dependency downloads.'
    }
    $partial = $archivePath + '.partial'
    $arguments = @(
        '--location',
        '--fail',
        '--silent',
        '--show-error',
        '--retry', '3',
        '--continue-at', '-',
        '--output', $partial,
        [string]$Component.url
    )
    & $curl.Source @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Download failed for $($Component.id): $($Component.url)"
    }

    $actualHash = Get-FileSha256 -LiteralPath $partial
    if ($actualHash -ne $expectedHash) {
        $quarantine = $partial + '.bad-' + [DateTime]::Now.ToString('yyyyMMddHHmmssfff')
        Move-Item -LiteralPath $partial -Destination $quarantine
        throw "Downloaded archive SHA256 mismatch for $($Component.id); quarantined as $quarantine."
    }
    Move-Item -LiteralPath $partial -Destination $archivePath
    $archivePath
}

function Expand-DependencyArchive {
    param(
        [Parameter(Mandatory)][string]$ArchivePath,
        [Parameter(Mandatory)][string]$DestinationPath
    )

    if (-not (Test-Path -LiteralPath $ArchivePath -PathType Leaf)) {
        throw "Archive does not exist: $ArchivePath"
    }
    if (Test-Path -LiteralPath $DestinationPath) {
        throw "Archive destination must not already exist: $DestinationPath"
    }
    New-Item -ItemType Directory -Path $DestinationPath -Force | Out-Null
    Expand-Archive -LiteralPath $ArchivePath -DestinationPath $DestinationPath -Force
}

function Copy-ComponentRuntimeFiles {
    param(
        [Parameter(Mandatory)][string]$ExtractedRoot,
        [Parameter(Mandatory)][string]$DestinationPath,
        [Parameter(Mandatory)][ValidateSet('bin', 'lib')][string]$Layout
    )

    $root = [IO.Path]::GetFullPath($ExtractedRoot)
    $layoutDirectories = New-Object Collections.Generic.List[IO.DirectoryInfo]
    $rootItem = Get-Item -LiteralPath $root
    if ($rootItem.Name -ieq $Layout) {
        $layoutDirectories.Add($rootItem)
    }
    foreach ($directory in @(Get-ChildItem -LiteralPath $root -Directory -Recurse | Where-Object { $_.Name -ieq $Layout })) {
        $layoutDirectories.Add($directory)
    }

    $runtimeFiles = @(
        $layoutDirectories | ForEach-Object {
            Get-ChildItem -LiteralPath $_.FullName -File -Filter '*.dll'
        } | Sort-Object FullName
    )
    if ($runtimeFiles.Count -eq 0) {
        throw "No runtime DLLs found under '$Layout' directories in $root"
    }

    New-Item -ItemType Directory -Path $DestinationPath -Force | Out-Null
    foreach ($file in $runtimeFiles) {
        $destination = Join-Path $DestinationPath $file.Name
        if (Test-Path -LiteralPath $destination -PathType Leaf) {
            $sourceHash = Get-FileSha256 -LiteralPath $file.FullName
            $destinationHash = Get-FileSha256 -LiteralPath $destination
            if ($sourceHash -ne $destinationHash) {
                throw "Runtime DLL name collision with different contents: $($file.Name)"
            }
            continue
        }
        Copy-Item -LiteralPath $file.FullName -Destination $destination
    }
}

function Copy-ComponentLicenses {
    param(
        [Parameter(Mandatory)][string]$ExtractedRoot,
        [Parameter(Mandatory)][string]$DestinationPath
    )

    $licenses = @(
        Get-ChildItem -LiteralPath $ExtractedRoot -File -Recurse | Where-Object {
            $_.Name -match '^(LICENSE|NOTICE|EULA)(\..*)?$' -or
            $_.Name -match '^ThirdPartyNotices?(\..*)?$' -or
            $_.Name -match '^Acknowledgements?(\..*)?$'
        } | Sort-Object FullName
    )
    if ($licenses.Count -eq 0) {
        throw "No license or notice file found in component archive: $ExtractedRoot"
    }

    New-Item -ItemType Directory -Path $DestinationPath -Force | Out-Null
    foreach ($license in $licenses) {
        $destination = Join-Path $DestinationPath $license.Name
        if (Test-Path -LiteralPath $destination -PathType Leaf) {
            if ((Get-FileSha256 -LiteralPath $license.FullName) -eq (Get-FileSha256 -LiteralPath $destination)) {
                continue
            }
            $destination = Join-Path $DestinationPath (([IO.Path]::GetFileNameWithoutExtension($license.Name)) + '-' + [guid]::NewGuid().ToString('N') + $license.Extension)
        }
        Copy-Item -LiteralPath $license.FullName -Destination $destination
    }
}

function Resolve-TensorRtArchive {
    param(
        [string]$ExplicitPath,
        [Parameter(Mandatory)][string]$InstallersRoot
    )

    $requiredName = 'TensorRT-8.6.1.6.Windows10.x86_64.cuda-11.8.zip'
    $candidate = if ([string]::IsNullOrWhiteSpace($ExplicitPath)) {
        Join-Path $InstallersRoot $requiredName
    } else {
        [IO.Path]::GetFullPath($ExplicitPath)
    }
    if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
        throw "Missing NVIDIA TensorRT archive '$requiredName'. Download it from https://developer.nvidia.com/nvidia-tensorrt-8x-download after login/EULA acceptance."
    }
    if ([IO.Path]::GetFileName($candidate) -cne $requiredName) {
        throw "TensorRT archive must use the exact official filename '$requiredName'; found '$([IO.Path]::GetFileName($candidate))'."
    }
    [IO.Path]::GetFullPath($candidate)
}

Export-ModuleMember -Function @(
    'Get-FileSha256',
    'Get-ImmutablePackageFiles',
    'Write-PackageManifest',
    'Test-PackageManifest',
    'Assert-CompatibleRuntimeFiles',
    'Test-TensorRtArchiveLayout',
    'Get-RuntimePathEntries',
    'Find-ProjectAssetRoot',
    'Get-VerifiedArchive',
    'Expand-DependencyArchive',
    'Copy-ComponentRuntimeFiles',
    'Copy-ComponentLicenses',
    'Resolve-TensorRtArchive'
)
