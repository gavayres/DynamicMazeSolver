from collections import namedtuple
import numpy as np
from read_maze import get_local_maze_information

PENALTIES = namedtuple('Penalties', 
                       ['fire', 'wall', 'revisit'], 
                       defaults=[-0.02, -0.04, -0.01])

class Env:
    def __init__(self, time_limit):
        self.x = 1
        self.y = 1
        self.time = 0
        self.state = self._get_state()
        self.actions = ['up', 'down', 'left', 'right', 'stay']
        self.timed_out = False
        self.time_limit = time_limit
        self.goal = (199, 199)
        self.penalties = PENALTIES()

    def update(self, action):
        """
        action is an integer index
        """
        action, penalty = self._check_valid_action(action)
        # move agent according to chosen action
        self._move_agent(action)
        # increment time steps
        self.time += 1
        # update state information
        self.state = self._get_state()
        # check if we are revisiting a state
        # note, we move agent to the new x,y 
        # check if it has been here before 
        # and THEN add the current x,y to the path
        penalty += self._check_path(self.x, self.y)
        # add new position to path
        self.path.append((self.x, self.y))
        return self.state, self._is_terminal(), penalty

    def reset(self):
        self.x=1
        self.y=1
        self.time=0
        self.state = self._get_state()
        self.timed_out=False
        self.path = [(self.x, self.y)]

    def _get_state(self):
        return get_local_maze_information(self.x, self.y)

    def _move_agent(self, action):
        if action == "up":
            self.y -= 1 
        if action == "down":
            self.y += 1
        if action == "right":
            self.x += 1
        if action == "left":
            self.x -= 1
        if action == "stay":
            pass

    def _check_path(self, x, y):
        if (x,y) in self.path:
            return self.penalties.revisit
        return 0

    def _check_valid_action(self, action):
        rel_x, rel_y = 1, 1 # relative position in state array
        valid = True # account for action validity
        penalty=0

        if action == "up":
            valid, penalty = self._check_state(rel_x, rel_y-1)
        elif action == "down":
            valid, penalty = self._check_state(rel_x, rel_y+1)
        elif action == "left":
            valid, penalty = self._check_state(rel_x-1, rel_y)
        elif action == "right":
            valid, penalty = self._check_state(rel_x+1, rel_y)
        return (action, 0) if valid else ("stay", penalty) 

    def _is_terminal(self):
        if (self.x == 199) & (self.y == 199):
            return 1
        elif self.time >= self.time_limit:
            self.timed_out = True
            return 1
        return 0

    def _check_state(self, rel_x, rel_y):
        if self.state[rel_x][rel_y][0] == 0: # wall
            return False, self.penalties.wall
        elif self.state[rel_x][rel_y][0] == 1:
            if self.state[rel_x][rel_y][1] > 0: # fiya
                return False, self.penalties.fire
        return True, 0


"""
class TestEnv:
    def __init__(self, time_limit):
        self.x = 1
        self.y = 1
        self.time = 0
        self.state = self._get_state()
        self.actions = ['up', 'down', 'left', 'right', 'stay']
        self.timed_out = False
        self.time_limit = time_limit
        self.goal = (199, 199)
        self.penalties = PENALTIES()
    
    def _get_state(self):
        state = get_local_maze_information(self.x, self.y)
        # zero out fire to make it easier for us
        state[:, :, 1] = np.zeros_like(state[:, :, 1])
        return state

    def update(self, action):
        
        action, penalty = self._check_valid_action(action)
        # move agent according to chosen action
        self._move_agent(action)
        # increment time steps
        self.time += 1
        # update state information
        self.state = self._get_state()
        # check if we are revisiting a state
        penalty += self._check_path(self.x, self.y)
        # add new position to path
        self.path.append((self.x, self.y))
        return self.state, self._is_terminal(), penalty

    def reset(self):
        self.x=1
        self.y=1
        self.time=0
        self.state = self._get_state()
        self.timed_out=False
        self.path = [(self.x, self.y)]

    def _move_agent(self, action):
        if action == "up":
            self.y -= 1 
        if action == "down":
            self.y += 1
        if action == "right":
            self.x += 1
        if action == "left":
            self.x -= 1
        if action == "stay":
            pass

    def _check_valid_action(self, action):
        rel_x, rel_y = 1, 1 # relative position in state array
        valid = True # account for action validity
        if action == "up":
            valid = self._check_state(rel_x, rel_y-1)
        elif action == "down":
            valid = self._check_state(rel_x, rel_y+1)
        elif action == "left":
            valid = self._check_state(rel_x-1, rel_y)
        elif action == "right":
            valid = self._check_state(rel_x+1, rel_y)
        return (action, 0) if valid else ("stay", -0.01) 

    def _is_terminal(self):
        if (self.x == 199) & (self.y == 199):
            return 1
        elif self.time >= self.time_limit:
            self.timed_out = True
            return 1
        return 0

    def _check_state(self, rel_x, rel_y):
        if self.state[rel_x, rel_y, 0] == 0: # wall
                return False
        elif self.state[rel_x, rel_y, 0] == 1:
            if self.state[rel_x, rel_y, 1] > 0: # fiya
                return False
        return True
"""

class TestEnv(Env):
    def __init__(self, time_limit):
        super().__init__(time_limit)

    def _get_state(self):
        state = get_local_maze_information(self.x, self.y)
        # zero out fire to make it easier for us
        state[:, :, 1] = np.zeros_like(state[:, :, 1])
        return state