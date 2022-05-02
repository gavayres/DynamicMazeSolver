import numpy as np
from read_maze import get_local_maze_information

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
        return self.state, self._is_terminal(), penalty

    def reset(self):
        self.x=1
        self.y=1
        self.time=0
        self.state = self._get_state()
        self.timed_out=False

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
        return (action, 0) if valid else ("stay", -1) 

    def _is_terminal(self):
        if (self.x == 199) & (self.y == 199):
            return 1
        elif self.time >= self.time_limit:
            self.timed_out = True
            return 1
        return 0

    def _check_state(self, rel_x, rel_y):
        if self.state[rel_x][rel_y][0] == 0: # wall
            return False
        elif self.state[rel_x][rel_y][0] == 1:
            if self.state[rel_x][rel_y][1] > 0: # fiya
                return False
        return True


class TestEnv:
    def __init__(self, time_limit):
        self.x = 1
        self.y = 1
        self.old_x = 1
        self.old_y = 1
        self.time = 0
        self.state = self._get_state()
        self.actions = ['up', 'down', 'left', 'right', 'stay']
        self.timed_out = False
        self.time_limit = time_limit
        self.goal = (199, 199)
    
    def _get_state(self):
        state = get_local_maze_information(self.x, self.y)
        # zero out fire to make it easier for us
        state[:, :, 1] = np.zeros_like(state[:, :, 1])
        return state

    def update(self, action):
        """
        action is an integer index
        """
        action, penalty = self._check_valid_action(action)
        old_x, old_y = self.x, self.y
        self.old_x, self.old_y = old_x, old_y
        # move agent according to chosen action
        self._move_agent(action)
        # increment time steps
        self.time += 1
        # update state information
        self.state = self._get_state()
        return self.state, self._is_terminal(), penalty

    def reset(self):
        self.x=1
        self.y=1
        self.old_x=1
        self.old_y=1
        self.time=0
        self.timed_out=False
        self.state = self._get_state()

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