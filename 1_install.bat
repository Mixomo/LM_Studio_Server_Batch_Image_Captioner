@echo off
setlocal
cd /d "%~dp0"

echo Checking uv...
call :find_uv

if not defined UV_EXE (
    echo uv was not found. Installing uv with winget...
    winget install --id astral-sh.uv -e --accept-package-agreements --accept-source-agreements
    if errorlevel 1 (
        echo.
        echo Failed to install uv with winget.
        pause
        exit /b 1
    )
    call :find_uv
)

if not defined UV_EXE (
    echo.
    echo uv is still not available after install.
    echo Close this terminal and run this installer again, or install uv manually.
    pause
    exit /b 1
)

echo Found uv: %UV_EXE%
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
echo Installation finished.
pause
exit /b 0

:find_uv
set "UV_EXE="
for %%P in (
    "%USERPROFILE%\.local\bin\uv.exe"
    "%USERPROFILE%\.cargo\bin\uv.exe"
    "%LocalAppData%\Programs\uv\uv.exe"
    "%ProgramFiles%\uv\uv.exe"
) do (
    if exist "%%~P" (
        set "UV_EXE=%%~P"
        exit /b 0
    )
)
for /f "delims=" %%P in ('where uv 2^>nul') do (
    set "UV_EXE=%%P"
    exit /b 0
)
exit /b 0
