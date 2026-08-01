param(
    [string]$CondaEnvironment = "manga-llm",
    [string]$IcuVersion = "78.3",
    [string]$PyIcuVersion = "2.16.2",
    [string]$ArtifactRoot = "output\icu4c_pyicu_runtime_deployment_successor_20260730"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Resolve-CondaEnvironmentPrefix {
    param([string]$Name)

    $json = & conda env list --json
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to query Conda environments."
    }

    $payload = $json | ConvertFrom-Json
    $suffix = [IO.Path]::Combine("envs", $Name)
    $match = @($payload.envs | Where-Object {
        $_ -eq $Name -or $_.TrimEnd("\") -like "*\$suffix"
    })
    if ($match.Count -ne 1) {
        throw "Expected exactly one Conda environment named '$Name'."
    }
    return [IO.Path]::GetFullPath([string]$match[0])
}

function Resolve-VcVars64 {
    $vswhereCandidates = @(
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe",
        "$env:ProgramFiles\Microsoft Visual Studio\Installer\vswhere.exe"
    )
    foreach ($candidate in $vswhereCandidates) {
        if (-not (Test-Path -LiteralPath $candidate)) {
            continue
        }
        $installation = & $candidate -latest -products * `
            -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
            -property installationPath
        if ($LASTEXITCODE -eq 0 -and $installation) {
            $vcvars = Join-Path $installation "VC\Auxiliary\Build\vcvars64.bat"
            if (Test-Path -LiteralPath $vcvars) {
                return [IO.Path]::GetFullPath($vcvars)
            }
        }
    }

    $fixedCandidate = "C:\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    if (Test-Path -LiteralPath $fixedCandidate) {
        return $fixedCandidate
    }
    throw "Visual Studio 2022 Build Tools with the x64 C++ workload is required."
}

$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$artifactPath = [IO.Path]::GetFullPath((Join-Path $repoRoot $ArtifactRoot))
$wheelhouse = Join-Path $artifactPath "wheelhouse"
$linkLibraryDir = Join-Path $artifactPath "pyicu-link-libraries"
New-Item -ItemType Directory -Force -Path $wheelhouse, $linkLibraryDir |
    Out-Null

& conda install -n $CondaEnvironment -y "icu=$IcuVersion"
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install ICU4C $IcuVersion."
}

$prefix = Resolve-CondaEnvironmentPrefix -Name $CondaEnvironment
$python = Join-Path $prefix "python.exe"
$includeDir = Join-Path $prefix "Library\include"
$libraryDir = Join-Path $prefix "Library\lib"
$runtimeDir = Join-Path $prefix "Library\bin"
$vcvars64 = Resolve-VcVars64

foreach ($requiredPath in @(
    $python,
    (Join-Path $includeDir "unicode\ubrk.h"),
    (Join-Path $libraryDir "icuin.lib"),
    (Join-Path $libraryDir "icuuc.lib"),
    (Join-Path $libraryDir "icudt.lib")
)) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        throw "Missing ICU/Python build prerequisite: $requiredPath"
    }
}

$libraryAliases = [ordered]@{
    "pyicu_icuin.lib" = (Join-Path $libraryDir "icuin.lib")
    "pyicu_icuuc.lib" = (Join-Path $libraryDir "icuuc.lib")
    "pyicu_icudt.lib" = (Join-Path $libraryDir "icudt.lib")
}
$aliasInventory = @()
foreach ($entry in $libraryAliases.GetEnumerator()) {
    $aliasPath = Join-Path $linkLibraryDir $entry.Key
    Copy-Item -LiteralPath $entry.Value -Destination $aliasPath -Force
    $sourceHash = (
        Get-FileHash -Algorithm SHA256 -LiteralPath $entry.Value
    ).Hash.ToLowerInvariant()
    $aliasHash = (
        Get-FileHash -Algorithm SHA256 -LiteralPath $aliasPath
    ).Hash.ToLowerInvariant()
    if ($aliasHash -ne $sourceHash) {
        throw "ICU import-library alias hash mismatch: $aliasPath"
    }
    $aliasInventory += [ordered]@{
        alias = $aliasPath
        source = $entry.Value
        sha256 = $aliasHash
    }
}

Get-ChildItem -LiteralPath $wheelhouse -Filter "pyicu-$PyIcuVersion-*.whl" `
    -ErrorAction SilentlyContinue | Remove-Item -Force

$vcCommand = "call `"$vcvars64`" >nul && set"
$vcEnvironment = & $env:COMSPEC /d /c $vcCommand
if ($LASTEXITCODE -ne 0) {
    throw "Unable to initialize the Visual Studio x64 build environment."
}
foreach ($line in $vcEnvironment) {
    if ($line -match "^([^=]+)=(.*)$") {
        [Environment]::SetEnvironmentVariable(
            $Matches[1],
            $Matches[2],
            [EnvironmentVariableTarget]::Process
        )
    }
}

$env:ICU_VERSION = $IcuVersion
$env:PYICU_INCLUDES = $includeDir
$env:PYICU_CFLAGS = "/std:c++17;/Zc:wchar_t;/EHsc"
$env:PYICU_LFLAGS = "/LIBPATH:$linkLibraryDir"
$env:PYICU_LIBRARIES = "pyicu_icuin;pyicu_icuuc;pyicu_icudt"
$env:PATH = "$runtimeDir;$prefix;$env:PATH"

$requirement = "PyICU==$PyIcuVersion"
& $python -m pip wheel --no-deps --no-build-isolation --no-cache-dir `
    --wheel-dir $wheelhouse $requirement
if ($LASTEXITCODE -ne 0) {
    throw "PyICU wheel build failed."
}

$wheels = @(Get-ChildItem -LiteralPath $wheelhouse `
    -Filter "pyicu-$PyIcuVersion-*.whl")
if ($wheels.Count -ne 1) {
    throw "Expected exactly one PyICU $PyIcuVersion wheel."
}
$wheel = $wheels[0]

& $python -m pip install --force-reinstall --no-deps $wheel.FullName
if ($LASTEXITCODE -ne 0) {
    throw "PyICU wheel installation failed."
}

$versionProbe = @"
import icu
assert icu.VERSION == "$PyIcuVersion", icu.VERSION
assert icu.ICU_VERSION == "$IcuVersion", icu.ICU_VERSION
print(icu.__file__)
print(icu.VERSION)
print(icu.ICU_VERSION)
"@
& $python -c $versionProbe
if ($LASTEXITCODE -ne 0) {
    throw "Installed PyICU runtime verification failed."
}

$wheelHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $wheel.FullName).Hash
$inventory = [ordered]@{
    schema_version = 1
    kind = "icu4c-pyicu-wheel-inventory"
    generated_at = [DateTime]::UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ")
    conda_environment = $CondaEnvironment
    environment_prefix = $prefix
    python = $python
    icu_version = $IcuVersion
    pyicu_version = $PyIcuVersion
    wheel = $wheel.FullName
    wheel_sha256 = $wheelHash.ToLowerInvariant()
    compiler_bootstrap = $vcvars64
    include_dir = $includeDir
    library_dir = $libraryDir
    link_library_dir = $linkLibraryDir
    import_library_aliases = $aliasInventory
    runtime_dir = $runtimeDir
}
$inventoryPath = Join-Path $artifactPath "wheel_inventory.json"
$inventory | ConvertTo-Json -Depth 4 |
    Set-Content -LiteralPath $inventoryPath -Encoding utf8

Write-Host "Installed ICU4C $IcuVersion and PyICU $PyIcuVersion."
Write-Host "Wheel: $($wheel.FullName)"
Write-Host "SHA256: $($wheelHash.ToLowerInvariant())"
Write-Host "Inventory: $inventoryPath"
