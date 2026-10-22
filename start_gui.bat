@echo off
setlocal EnableDelayedExpansion

rem ============================================================
rem  Start the Cat vs Dog web GUI
rem
rem  Just double-click this file. Optional settings (edit below
rem  or set them in the shell before running):
rem     AGU_PORT=8000       AGU_HOST=127.0.0.1
rem     AGU_MODEL_FORMAT=keras|saved_model
rem     AGU_NO_BROWSER=1    do not open a browser window
rem ============================================================

title Cat vs Dog - web GUI

rem Internal helper: the batch file relaunches itself with this flag in the
rem background to open a browser after a delay. Kept at the top so it exits
rem before any of the checks below run.
if /i "%~1"=="--open-in" goto :open_later

rem Always run from the repository root, no matter where the
rem batch file was launched from (e.g. a desktop shortcut).
cd /d "%~dp0"

goto :main

:open_later
rem %~2 is the delay in seconds. AGU_PORT/AGU_HOST are inherited.
set "OPEN_PORT=%AGU_PORT%"
if not defined OPEN_PORT set "OPEN_PORT=8000"
set "OPEN_HOST=%AGU_HOST%"
if not defined OPEN_HOST set "OPEN_HOST=127.0.0.1"
rem `ping` with a non-resolving address is a portable sleep that, unlike
rem `timeout`, also works when stdin is redirected or there is no console.
ping -n %~2 127.0.0.1 >nul 2>&1
if "%OPEN_HOST%"=="0.0.0.0" set "OPEN_HOST=127.0.0.1"
start "" "http://%OPEN_HOST%:%OPEN_PORT%"
exit /b 0

:main

if "%AGU_PORT%"=="" set "AGU_HOST=127.0.0.1" & set "AGU_PORT=8000"
if "%AGU_HOST%"=="" set "AGU_HOST=127.0.0.1"

rem Keep TensorFlow's startup chatter out of the way.
if "%TF_CPP_MIN_LOG_LEVEL%"=="" set "TF_CPP_MIN_LOG_LEVEL=3"

rem ---- find a Python interpreter -----------------------------
rem Prefer a project venv, fall back to whatever is on PATH.
set "PY="
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if not defined PY if exist "venv\Scripts\python.exe" set "PY=venv\Scripts\python.exe"
if not defined PY (
    where python >nul 2>&1
    if not errorlevel 1 set "PY=python"
)

if not defined PY (
    echo.
    echo   [ERROR] Python was not found.
    echo.
    echo   Install Python 3.10+ from https://www.python.org/downloads/
    echo   and make sure "Add python.exe to PATH" is checked.
    echo.
    pause
    exit /b 1
)

rem ---- check the model exists ---------------------------------
if not exist "model\exports\model_dogs_vs_cats.keras" (
    if not exist "model\exports\saved_model_dogs_vs_cats\saved_model.pb" (
        echo.
        echo   [WARNING] No exported model found in model\exports\.
        echo   Run the export cells in model\training_model.ipynb first.
        echo   The web page will load, but it will show an error panel
        echo   instead of predicting until the model is exported.
        echo.
    )
)

rem ---- check dependencies -------------------------------------
echo Checking dependencies...
"%PY%" -c "import fastapi, uvicorn, jinja2, PIL, numpy, multipart" >nul 2>&1
if errorlevel 1 (
    echo.
    echo   Some Python packages are missing. Install them with:
    echo.
    echo       "%PY%" -m pip install -r gui\requirements.txt
    echo.
    choice /c YN /n /m "  Install them now? [Y/N] "
    set "ANSWER=!errorlevel!"
    if "!ANSWER!"=="1" (
        "%PY%" -m pip install -r gui\requirements.txt
        if errorlevel 1 (
            echo.
            echo   [ERROR] Installation failed. See the messages above.
            pause
            exit /b 1
        )
    ) else (
        echo.
        echo   Aborted. Nothing was installed.
        pause
        exit /b 1
    )
) else (
    echo   OK
)

rem TensorFlow is the slow one to import, so check it separately
rem to give a clearer message if only it is missing.
"%PY%" -c "import tensorflow" >nul 2>&1
if errorlevel 1 (
    echo.
    echo   [WARNING] TensorFlow is not installed, so the model cannot load.
    echo   Install it with:  "%PY%" -m pip install tensorflow
    echo.
)

rem ---- warn if the port is already taken ----------------------
set "URL=http://%AGU_HOST%:%AGU_PORT%"
if "%AGU_HOST%"=="0.0.0.0" set "URL=http://127.0.0.1:%AGU_PORT%"

netstat -an | findstr /r /c:"LISTENING" | findstr /c:":%AGU_PORT% " >nul 2>&1
if not errorlevel 1 (
    echo.
    echo   [WARNING] Something is already listening on port %AGU_PORT%.
    echo   If it is an old copy of this app, close it first, or
    echo   start this one on a different port:
    echo.
    echo       set AGU_PORT=8001 ^&^& start_gui.bat
    echo.
    choice /c YN /n /m "  Try to start anyway? [Y/N] "
    rem Any answer other than an explicit Y aborts, so a failed or missing
    rem prompt (non-interactive stdin) cannot silently start a second server.
    set "ANSWER=!errorlevel!"
    if not "!ANSWER!"=="1" (
        echo   Aborted.
        pause
        exit /b 1
    )
)

rem ---- launch ------------------------------------------------
echo.
echo Starting the Cat vs Dog web GUI...
echo.
echo   URL:    %URL%
echo   Python: %PY%
echo.
echo   The model takes a few seconds to load. Press Ctrl+C to stop.
echo.
echo ============================================================

rem Open the browser after a short delay so it does not race the
rem server. Runs in a separate window-less process, so it does not
rem block the server, and is skipped with AGU_NO_BROWSER=1.
if not "%AGU_NO_BROWSER%"=="1" start /b "" "%~f0" --open-in 6

"%PY%" -m gui.app

rem Reached only when the server stops.
echo.
echo The server has stopped.
pause
endlocal
