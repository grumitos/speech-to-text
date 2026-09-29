@echo off
rem Lanza SpeechToText en Windows. La primera vez prepara el entorno (venv y dependencias).
rem Uso: run.bat [opciones de main.py]   p. ej.: run.bat --model gemini-3.8-flash --prompt "..."
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "VENV_PY=venv\Scripts\python.exe"
set "EXIT_CODE=0"

rem Con doble clic (cmd /c "...\run.bat") se hace una pausa final para poder leer la salida.
set "DOUBLE_CLICK="
echo %cmdcmdline% | "%SystemRoot%\System32\find.exe" /i "%~nx0" >nul && set "DOUBLE_CLICK=1"

if not exist "%VENV_PY%" (
    call :create_venv || goto :failed
)

"%VENV_PY%" -c "import google.genai, dotenv, rich" >nul 2>&1 || (
    call :install_dependencies || goto :failed
)

if not exist ".env" (
    copy /y ".env.example" ".env" >nul
    echo Se creo el archivo .env: reemplaza tu_clave_aqui por tu GOOGLE_API_KEY.
)

"%VENV_PY%" main.py %*
set "EXIT_CODE=%ERRORLEVEL%"
goto :finish

:create_venv
echo Creando el entorno virtual...
set "PY_CMD=python"
where python >nul 2>&1 || set "PY_CMD=py -3"
%PY_CMD% -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if errorlevel 1 (
    echo ERROR: se necesita Python 3.10 o superior en el PATH.
    exit /b 1
)
%PY_CMD% -m venv venv
exit /b %ERRORLEVEL%

:install_dependencies
echo Instalando dependencias...
"%VENV_PY%" -m pip install -r requirements.in
exit /b %ERRORLEVEL%

:failed
set "EXIT_CODE=1"

:finish
if defined DOUBLE_CLICK pause
exit /b %EXIT_CODE%
