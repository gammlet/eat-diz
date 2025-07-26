python3 -m venv sim_env
sim_env\Scripts\activate.bat

pip install matplotlib
pip install numpy
pip install pyinstaller

pyinstaller main.py --onefile --windowed
python3 main.py
