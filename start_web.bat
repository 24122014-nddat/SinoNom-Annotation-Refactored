@echo off
echo Starting SinoNom Annotation Web Tool...
cd /d "%~dp0"
.venv\Scripts\python.exe gradio/app.py --server-name 127.0.0.1 --port 7860
pause

