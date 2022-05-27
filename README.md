# DynamicMazeSolver
We use a Double Deep Q Network to solve a dynamically chaning maze. 
At each position, the agent can only observe a 3 by 3 grid of cells. In each of these cells fire may appear at random. The goal is to make it from the top corner of the maze, position (1,1), to the bottom corner of the maze, position (199, 199).
## Running
To run the project simply type ``make run`` into the command line. This will create a new virtual environment into which the specifications of requirements.txt will be installed and the file ``train_dqn.py`` will be run with the ``evaluate=True`` configuration.