param(
    [Parameter(Mandatory = $true)]
    [string]$UvExe,

    [Parameter(Mandatory = $true)]
    [string]$ProjectRoot
)

$ErrorActionPreference = "Stop"
$UvExe = $UvExe.Trim('"')
$ProjectRoot = $ProjectRoot.Trim('"').TrimEnd([char]92, [char]47)
$repoApi = "https://api.github.com/repos/JamePeng/llama-cpp-python/releases?per_page=100"
$wheelsDir = Join-Path $ProjectRoot "wheels"
$localUvCache = Join-Path $ProjectRoot ".uv_cache"
$cudaConfigPath = Join-Path $ProjectRoot ".llama_cpp_cuda.json"
if (-not $env:UV_CACHE_DIR) {
    $env:UV_CACHE_DIR = $localUvCache
}
New-Item -ItemType Directory -Force -Path $wheelsDir | Out-Null
New-Item -ItemType Directory -Force -Path $env:UV_CACHE_DIR | Out-Null

function Invoke-UvPipInstall {
    param([string[]]$PipArgs)

    & $UvExe @("pip", "install") @PipArgs
    if ($LASTEXITCODE -ne 0) {
        throw "uv pip install failed with exit code $LASTEXITCODE"
    }
}

function Test-WheelFile {
    param([string]$WheelPath)

    if (-not (Test-Path $WheelPath)) {
        return $false
    }

    try {
        Add-Type -AssemblyName System.IO.Compression.FileSystem -ErrorAction SilentlyContinue
        $zip = [System.IO.Compression.ZipFile]::OpenRead($WheelPath)
        $zip.Dispose()
        return $true
    } catch {
        return $false
    }
}

function Get-PythonTag {
    $tag = & $UvExe run python -c "import sys,platform; print(f'cp{sys.version_info.major}{sys.version_info.minor}|{platform.machine().lower()}')"
    if ($LASTEXITCODE -ne 0 -or -not $tag) {
        throw "Could not detect Python version through uv."
    }

    $parts = $tag.Trim() -split "\|"
    return [pscustomobject]@{
        Tag = $parts[0]
        Machine = $parts[1]
    }
}

function Get-CudaCode {
    $candidates = @()

    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($nvidiaSmi) {
        try {
            $query = & $nvidiaSmi.Source --query-gpu=cuda_version --format=csv,noheader 2>$null | Select-Object -First 1
            if ($query -match "(\d+)\.(\d+)") {
                $candidates += [int]("$($matches[1])$($matches[2])")
            }
        } catch {
        }

        try {
            $plain = & $nvidiaSmi.Source 2>$null | Out-String
            if ($plain -match "CUDA Version:\s*(\d+)\.(\d+)") {
                $candidates += [int]("$($matches[1])$($matches[2])")
            }
        } catch {
        }
    }

    $nvcc = Get-Command nvcc -ErrorAction SilentlyContinue
    if ($nvcc) {
        try {
            $nvccText = & $nvcc.Source --version 2>$null | Out-String
            if ($nvccText -match "release\s+(\d+)\.(\d+)") {
                $candidates += [int]("$($matches[1])$($matches[2])")
            }
        } catch {
        }
    }

    foreach ($envName in @("CUDA_PATH", "CUDA_HOME")) {
        $value = [Environment]::GetEnvironmentVariable($envName)
        if ($value -and $value -match "v?(\d+)\.(\d+)") {
            $candidates += [int]("$($matches[1])$($matches[2])")
        }
    }

    if (-not $candidates) {
        return $null
    }
    return ($candidates | Sort-Object -Descending | Select-Object -First 1)
}

function Get-LocalWheel {
    param([string]$PythonTag)

    $wheels = Get-ChildItem -Path $wheelsDir -Filter "llama_cpp_python*.whl" -File -ErrorAction SilentlyContinue
    foreach ($wheel in $wheels) {
        if (-not (Test-WheelFile -WheelPath $wheel.FullName)) {
            Write-Host "Removing invalid or incomplete wheel: $($wheel.FullName)"
            Remove-Item -Force -Path $wheel.FullName
        }
    }

    $local = Get-ChildItem -Path $wheelsDir -Filter "llama_cpp_python*.whl" -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match $PythonTag -and $_.Name -match "win_amd64" } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1

    if ($local) {
        return $local.FullName
    }

    $fallback = Get-ChildItem -Path $wheelsDir -Filter "llama_cpp_python*.whl" -File -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1

    if ($fallback) {
        return $fallback.FullName
    }

    return $null
}

function Save-CudaConfig {
    param(
        [Nullable[int]]$WheelCudaCode,
        [Nullable[int]]$DetectedCudaCode,
        [string]$WheelSource
    )

    $payload = [ordered]@{
        wheel_cuda = if ($WheelCudaCode.HasValue) { "cu$($WheelCudaCode.Value)" } else { $null }
        detected_cuda = if ($DetectedCudaCode.HasValue) { "cu$($DetectedCudaCode.Value)" } else { $null }
        wheel_source = $WheelSource
        written_at = (Get-Date).ToString("o")
    }
    $payload | ConvertTo-Json | Set-Content -Path $cudaConfigPath -Encoding UTF8
}

function Get-CudaCodeFromWheelName {
    param([string]$Name)

    if ($Name -match "cu(\d+)") {
        return [int]$matches[1]
    }
    return $null
}

function Get-ReleaseWheel {
    param(
        [string]$PythonTag,
        [Nullable[int]]$CudaCode
    )

    Write-Host "Querying JamePeng llama-cpp-python releases..."
    $headers = @{
        "Accept" = "application/vnd.github+json"
        "User-Agent" = "llamacpp-native-batch-captioner-installer"
    }
    $releases = Invoke-RestMethod -Uri $repoApi -Headers $headers

    $candidates = foreach ($release in $releases) {
        if ($release.draft -or $release.prerelease) {
            continue
        }
        if ($release.tag_name -notmatch "win") {
            continue
        }
        if ($release.tag_name -notmatch "cu(\d+)") {
            continue
        }

        $releaseCuda = [int]$matches[1]
        $asset = $release.assets |
            Where-Object {
                $_.name -match "\.whl$" -and
                $_.name -match "llama_cpp_python" -and
                $_.name -match $PythonTag -and
                $_.name -match "win_amd64"
            } |
            Sort-Object name |
            Select-Object -First 1

        if (-not $asset) {
            continue
        }

        $cudaRank = if ($CudaCode.HasValue) {
            if ($releaseCuda -le $CudaCode.Value) {
                10000 - ($CudaCode.Value - $releaseCuda)
            } else {
                0 - ($releaseCuda - $CudaCode.Value)
            }
        } else {
            0
        }

        [pscustomobject]@{
            Release = $release
            Asset = $asset
            Cuda = $releaseCuda
            CudaRank = $cudaRank
            PublishedAt = [datetime]$release.published_at
        }
    }

    return $candidates |
        Sort-Object @{ Expression = "CudaRank"; Descending = $true }, @{ Expression = "PublishedAt"; Descending = $true } |
        Select-Object -First 1
}

try {
    $python = Get-PythonTag
    if ($python.Machine -notin @("amd64", "x86_64")) {
        Write-Host "Python architecture is '$($python.Machine)'. JamePeng Windows wheels are expected for win_amd64."
    }
    Write-Host "Detected Python wheel tag: $($python.Tag)"

    $cudaCode = Get-CudaCode
    if ($null -eq $cudaCode) {
        Write-Host "CUDA was not detected. Will try a local wheel first, then PyPI."
    } else {
        Write-Host "Detected CUDA capability target: cu$cudaCode"
    }

    if ($env:LLAMA_CPP_PYTHON_WHEEL) {
        Write-Host "Installing wheel from LLAMA_CPP_PYTHON_WHEEL: $env:LLAMA_CPP_PYTHON_WHEEL"
        Invoke-UvPipInstall -PipArgs @($env:LLAMA_CPP_PYTHON_WHEEL, "--force-reinstall")
        Save-CudaConfig -WheelCudaCode (Get-CudaCodeFromWheelName -Name $env:LLAMA_CPP_PYTHON_WHEEL) -DetectedCudaCode $cudaCode -WheelSource $env:LLAMA_CPP_PYTHON_WHEEL
        exit 0
    }

    $localWheel = Get-LocalWheel -PythonTag $python.Tag
    if ($localWheel) {
        Write-Host "Installing local wheel: $localWheel"
        Invoke-UvPipInstall -PipArgs @($localWheel, "--force-reinstall")
        Save-CudaConfig -WheelCudaCode (Get-CudaCodeFromWheelName -Name $localWheel) -DetectedCudaCode $cudaCode -WheelSource $localWheel
        exit 0
    }

    if ($null -eq $cudaCode) {
        Write-Host "No CUDA runtime/driver was detected. Falling back to PyPI."
        Invoke-UvPipInstall -PipArgs @("llama-cpp-python")
        exit 0
    }

    $candidate = Get-ReleaseWheel -PythonTag $python.Tag -CudaCode $cudaCode
    if ($candidate) {
        Write-Host "Selected release: $($candidate.Release.tag_name)"
        Write-Host "Selected CUDA wheel family: cu$($candidate.Cuda)"
        Write-Host "Installing with uv directly from: $($candidate.Asset.browser_download_url)"
        Invoke-UvPipInstall -PipArgs @($candidate.Asset.browser_download_url, "--force-reinstall")
        Save-CudaConfig -WheelCudaCode $candidate.Cuda -DetectedCudaCode $cudaCode -WheelSource $candidate.Asset.browser_download_url
        exit 0
    }

    Write-Host "No matching JamePeng Windows wheel found for $($python.Tag). Falling back to PyPI."
    Invoke-UvPipInstall -PipArgs @("llama-cpp-python")
    Save-CudaConfig -WheelCudaCode $null -DetectedCudaCode $cudaCode -WheelSource "pypi"
    exit 0
} catch {
    Write-Host ""
    Write-Host "Automatic JamePeng wheel install failed:"
    Write-Host $_.Exception.Message
    Write-Host ""
    Write-Host "Trying PyPI fallback..."
    Invoke-UvPipInstall -PipArgs @("llama-cpp-python")
    Save-CudaConfig -WheelCudaCode $null -DetectedCudaCode $cudaCode -WheelSource "pypi-fallback"
    exit 0
}
