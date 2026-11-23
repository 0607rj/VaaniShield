@echo off
REM VaaniShield demo: starts the backend and opens the web page. Keep this window open during the demo.
cd /d "%~dp0"
echo Starting VaaniShield server on http://127.0.0.1:8000 ...
start "" /b cmd /c "timeout /t 12 >nul & start http://127.0.0.1:8000"
".venv\Scripts\python.exe" -m uvicorn server.app:app --host 127.0.0.1 --port 8000
pause
