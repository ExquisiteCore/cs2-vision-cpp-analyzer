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
    $match = [regex]::Match($Header, "(?m)^\s*#define\s+$Name\s+(\d+)\s*$")
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

Export-ModuleMember -Function @(
    'Get-FileSha256',
    'Get-ImmutablePackageFiles',
    'Write-PackageManifest',
    'Test-PackageManifest',
    'Assert-CompatibleRuntimeFiles',
    'Test-TensorRtArchiveLayout',
    'Get-RuntimePathEntries'
)
