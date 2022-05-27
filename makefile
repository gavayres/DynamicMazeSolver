run: venv/bin/activate
	cd src/dynamic-maze-solver/
	python src/dynamic-maze-solver/train_dqn.py --load_path='./checkpoints' --evaluate=True

venv/bin/activate: requirements.txt
	python3 -m venv project_venv
	cat requirements.txt | xargs -n 1 ./project_venv/bin/pip3 install

clean:
	rm -rf __pycache__
	rm -rf venv