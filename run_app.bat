@echo off
title Life Link Nexus - Local Web Server
echo ===================================================
echo   Starting Life Link Nexus Local Web Server...
echo ===================================================
echo.
echo [1/2] Launching Python HTTP Server on port 8000...
start "" py -m http.server 8000
echo [2/2] Opening Life Link Nexus in your default browser...
timeout /t 2 /nobreak > nul
start http://localhost:8000/index.html
echo.
echo ===================================================
echo   Server is running at http://localhost:8000
echo   To stop the server, close this command window.
echo ===================================================
echo.
pause
