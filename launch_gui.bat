@echo off
REM ML Studio launcher for Windows: double-click, or run from a terminal.
REM Runs from this folder so Streamlit finds .streamlit\config.toml (the theme
REM and the upload-size limit). Uses whichever "python" is first on PATH - run
REM it from the same environment (conda env / venv) you use for the notebooks,
REM so a trained-models file saved there loads here.
cd /d "%~dp0"
python -m streamlit run gui\app.py
if errorlevel 1 (
  echo.
  echo Could not start the app. Install the requirements first:
  echo     python -m pip install -r requirements-gui.txt
  pause
)
