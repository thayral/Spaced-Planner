@echo off
setlocal
if not exist .venv python -m venv .venv
call .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
streamlit run calendar_ui.py