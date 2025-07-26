rmdir /s /q .\dist
rmdir /s /q .\sim_env

python3 -m venv sim_env
call .\sim_env\Scripts\activate.bat

pip install matplotlib
pip install numpy
pip install pyinstaller

pyinstaller main.py --onefile --windowed
REM python3 main.py

.\dist\main.exe
