@echo off
echo ============================================================
echo  KodaQuant - Streamlit UI with GPU
echo ============================================================
echo.
echo Starting Streamlit server in WSL2...
echo.
echo  - Direct URL:   http://172.26.83.89:8501
echo  - Localhost:    http://localhost:8501
echo.
echo  If the page looks stuck / blank in your browser:
echo    1. Try the DIRECT URL above (172.26.83.89:8501)
echo    2. Hard-refresh with Ctrl+Shift+R
echo    3. Try Incognito mode
echo ============================================================
echo.

wsl -d Ubuntu-22.04 -- bash /mnt/c/Users/rafa/ML_Trading/thesisproj/run_wsl.sh python -m streamlit run app.py --server.headless true --server.address 0.0.0.0 --server.port 8501 --server.enableWebsocketCompression false --browser.gatherUsageStats false