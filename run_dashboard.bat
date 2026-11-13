@echo off
cd /d "%~dp0"
echo Starting Sentiment Analysis Dashboard...
py -3 -m streamlit run dashboard.py
