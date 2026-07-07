@echo off
echo ========================================================
echo Setting up Manim Animation Explainer Environment
echo ========================================================
echo.
echo Checking for system dependencies (ffmpeg, miktex, cairo)...
echo Manim requires external dependencies to render videos.
echo If you haven't installed them, it is recommended to use Chocolatey:
echo    choco install manim-ce
echo Or install manually: ffmpeg, miktex (for LaTeX), and pango/cairo.
echo.
pause

echo Setting up Python virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

echo Installing required Python packages...
pip install manim pyyaml

echo Running Manim video generation...
python animtovideo.py

echo Done!
pause
