@echo off
setlocal
cd /d "%~dp0"

call :find_uv
if not defined UV_EXE (
    echo uv was not found. Run 1_install.bat first.
    pause
    exit /b 1
)

"%UV_EXE%" run python app.py
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
