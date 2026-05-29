set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
cd /d "%SCRIPT_DIR%"
set "PATH=%SCRIPT_DIR%\runtime;%PATH%"

:: 在新窗口中启动 api_neko 服务 (端口 9881)
start "GPT-SoVITS API v3" runtime\python.exe -m api_neko.server -p 9881

runtime\python.exe -I webui.py zh_CN
pause
