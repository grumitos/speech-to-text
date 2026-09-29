@echo off
rem SpeechToText: transcribe los audios de entrada\ y guarda los textos en salida\.
rem La primera vez prepara el entorno (venv y dependencias), crea .env y se detiene para completarlo.
rem Uso: run.bat [opciones]   p. ej.: run.bat --model gemini-3.8-flash --prompt "..."
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "MIN_PYTHON=3.10"
set "VENV_PY=venv\Scripts\python.exe"
set "EXIT_CODE=0"

rem Con doble clic (cmd /c "...\run.bat") se hace una pausa final para poder leer la salida.
set "DOUBLE_CLICK="
echo %cmdcmdline% | "%SystemRoot%\System32\find.exe" /i "%~nx0" >nul && set "DOUBLE_CLICK=1"

if not exist "%VENV_PY%" (
    call :find_python || goto :failed
    call :create_venv || goto :failed
)
"%VENV_PY%" -c "import google.genai, dotenv, rich" >nul 2>&1 || (
    call :install_dependencies || goto :failed
)
if not exist ".env" (
    call :create_env
    goto :failed
)

"%VENV_PY%" -m speech_to_text %*
set "EXIT_CODE=%ERRORLEVEL%"
goto :finish

:find_python
rem Deja en PY_CMD un Python %MIN_PYTHON% o superior: primero python y, si no, el lanzador py.
for %%P in ("python" "py -3") do (
    %%~P -c "import sys; sys.exit(sys.version_info < (%MIN_PYTHON:.=, %))" >nul 2>&1 && (
        set "PY_CMD=%%~P"
        exit /b 0
    )
)
echo ERROR: se necesita Python %MIN_PYTHON% o superior en el PATH.
exit /b 1

:create_venv
echo Creando el entorno virtual...
%PY_CMD% -m venv venv
exit /b %ERRORLEVEL%

:install_dependencies
echo Instalando dependencias...
"%VENV_PY%" -m pip install -r requirements.txt
exit /b %ERRORLEVEL%

:create_env
rem Se detiene siempre: el .env de ejemplo no trae la clave de la API.
if not exist "entrada" mkdir "entrada"
copy /y ".env.example" ".env" >nul || (
    echo ERROR: no se pudo crear .env a partir de .env.example.
    exit /b 1
)
echo Se creo .env: reemplaza tu_clave_aqui por tu GOOGLE_API_KEY, copia los audios en entrada\
echo y vuelve a ejecutar run.bat.
exit /b 1

:failed
set "EXIT_CODE=1"

:finish
if defined DOUBLE_CLICK pause
exit /b %EXIT_CODE%
