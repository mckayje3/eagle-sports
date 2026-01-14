@echo off
REM Sports Prediction Dashboard Launcher
echo.
echo ====================================
echo Eagle Eye Sports Dashboard
echo ====================================
echo.
echo Starting Streamlit server...
echo.
echo The dashboard will open in your browser automatically.
echo URL: http://localhost:8501
echo.
echo Press Ctrl+C to stop the server.
echo.

py -m streamlit run streamlit_app.py

pause
