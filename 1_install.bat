@echo off
setlocal
cd /d "%~dp0"
set "UV_CACHE_DIR=%CD%\.uv_cache"
call :add_cuda_paths

echo Checking uv...
set "UV_EXE="
for /f "delims=" %%P in ('where uv 2^>nul') do (
    set "UV_EXE=%%P"
    goto :uv_found
)
for %%P in (
    "%USERPROFILE%\.local\bin\uv.exe"
    "%USERPROFILE%\.cargo\bin\uv.exe"
    "%LocalAppData%\Programs\uv\uv.exe"
    "%ProgramFiles%\uv\uv.exe"
) do (
    if exist "%%~P" (
        set "UV_EXE=%%~P"
        goto :uv_found
    )
)

:uv_found

if not defined UV_EXE (
    echo uv was not found. Installing uv with winget...
    winget install --id astral-sh.uv -e --accept-package-agreements --accept-source-agreements
    if errorlevel 1 (
        echo.
        echo Failed to install uv with winget.
        pause
        exit /b 1
    )
    for /f "delims=" %%P in ('where uv 2^>nul') do (
        set "UV_EXE=%%P"
        goto :uv_installed
    )
    for %%P in (
        "%USERPROFILE%\.local\bin\uv.exe"
        "%USERPROFILE%\.cargo\bin\uv.exe"
        "%LocalAppData%\Programs\uv\uv.exe"
        "%ProgramFiles%\uv\uv.exe"
    ) do (
        if exist "%%~P" (
            set "UV_EXE=%%~P"
            goto :uv_installed
        )
    )
)

:uv_installed

if not defined UV_EXE (
    echo.
    echo uv is still not available after install.
    echo Close this terminal and run this installer again, or install uv manually.
    pause
    exit /b 1
)

echo Found uv: %UV_EXE%
echo Using uv cache: %UV_CACHE_DIR%
echo.
echo Creating/updating virtual environment with uv sync...
"%UV_EXE%" sync
if errorlevel 1 (
    echo.
    echo uv sync failed.
    pause
    exit /b 1
)

echo.
echo Installing llama-cpp-python...
powershell -NoProfile -ExecutionPolicy Bypass -File "%CD%\scripts\install_llamacpp_python.ps1" -UvExe "%UV_EXE%" -ProjectRoot "%CD%"
if errorlevel 1 (
    echo.
    echo Failed to install llama-cpp-python.
    pause
    exit /b 1
)

echo.
echo Installation finished.
pause
exit /b 0

:add_cuda_paths
if defined CUDA_PATH (
    if exist "%CUDA_PATH%\bin\x64" set "PATH=%CUDA_PATH%\bin\x64;%PATH%"
    if exist "%CUDA_PATH%\bin" set "PATH=%CUDA_PATH%\bin;%PATH%"
)
if defined CUDA_HOME (
    if exist "%CUDA_HOME%\bin\x64" set "PATH=%CUDA_HOME%\bin\x64;%PATH%"
    if exist "%CUDA_HOME%\bin" set "PATH=%CUDA_HOME%\bin;%PATH%"
)
for /d %%D in ("C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*") do (
    if exist "%%~fD\bin\x64" set "PATH=%%~fD\bin\x64;%PATH%"
    if exist "%%~fD\bin" set "PATH=%%~fD\bin;%PATH%"
)
exit /b 0
