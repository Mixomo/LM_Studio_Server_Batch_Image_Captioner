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
    echo uv was not found. Run 1_install.bat first.
    pause
    exit /b 1
)

echo Found uv: %UV_EXE%
echo Using uv cache: %UV_CACHE_DIR%
"%UV_EXE%" run --no-sync python app.py
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
