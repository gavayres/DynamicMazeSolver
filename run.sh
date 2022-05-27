#! /bin/bash
python3 -m venv project_venv
source ./project_venv/bin/activate
./project_venv/bin/pip3 install --upgrade pip
cat requirements.txt | xargs -n 1 ./project_venv/bin/pip3 install
./project_venv/bin/pip3 install torch 
cd src/dynamic-maze-solver/
../.././project_venv/bin/python3 train_dqn.py --load_path='./checkpoints' --evaluate=True > output.txt