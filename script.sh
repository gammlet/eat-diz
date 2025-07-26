python3 -m venv sim_env
source sim_env/bin/activate

pip install matplotlib
pip install numpy
pip install pyinstaller

pyinstaller main.py --onefile --windowed
#python3 main.py

#chmod +x script.sh for prommision 
#./script.sh run this script
