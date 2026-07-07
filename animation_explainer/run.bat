@echo off
echo Setting up Python virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

echo Installing required packages...
pip install manim pyyaml

echo Running Manim video generation...
python animtovideo.py

echo Done!
