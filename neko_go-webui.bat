@echo off
setlocal enabledelayedexpansion

:: 当前 .bat 目录
set "CURRENT_DIR=%~dp0"
set "CURRENT_DIR=!CURRENT_DIR:~0,-1!"

:: 从当前目录开始，最多向上回溯      5      层
set "SEARCH_DIR=!CURRENT_DIR!"
set MAX_DEPTH=5
set DEPTH=0

:search_loop
if !DEPTH! gtr !MAX_DEPTH! (
    echo Error: Could not find project root containing 'runtime\python.exe' and 'api_neko\entry.py'.
    pause
    exit /b 1
)

:: 当前 SEARCH_DIR 包含 runtime\python.exe 和 api_neko\entry.py ?
if exist "!SEARCH_DIR!\runtime\python.exe" if exist "!SEARCH_DIR!\api_neko\entry.py" (
    set "PROJECT_ROOT=!SEARCH_DIR!"
    goto found_root
)

:: 向上一级目录
for %%F in ("!SEARCH_DIR!") do set "PARENT=%%~dpF"
if "!PARENT!"=="!SEARCH_DIR!" (
    echo Error: Reached drive root without finding project root.
    pause
    exit /b 1
)
set "SEARCH_DIR=!PARENT:~0,-1!"
set /a DEPTH+=1
goto search_loop

:found_root
cd /d "!PROJECT_ROOT!"
echo Found project root: !PROJECT_ROOT!
echo Running entry.py...
"!PROJECT_ROOT!\runtime\python.exe" -I "!PROJECT_ROOT!\api_neko\entry.py"

pause