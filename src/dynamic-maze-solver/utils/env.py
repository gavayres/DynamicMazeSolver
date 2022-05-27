from collections import namedtuple
from functools import reduce
import logging
import numpy as np
from utils.helpers import populate_landmarks
from read_maze import get_local_maze_information

PENALTIES = namedtuple('Penalties', 
                       ['fire', 'wall', 'revisit', 'stay', 'blocked', 'landmark'], 
                       defaults=[-0.02, -0.04, -0.01, -0.02, -1, 0.04])

class Env:
    def __init__(self, time_limit, maze, fire=True, goal=(199,199), landmark_step=5):
        self.fire = fire
        self.x = 1
        self.y = 1
        self.time = 0
        self.state = self._get_state()
        self.actions = ['up', 'down', 'left', 'right', 'stay']
        self.timed_out = False
        self.time_limit = time_limit
        self.goal = goal
        self.penalties = PENALTIES()
        self.blocked = False
        self.landmarks = populate_landmarks(landmark_step, self.goal, maze)
        self.path = [(self.x, self.y)]

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
        if self.fire:
            return get_local_maze_information(self.x, self.y)
        else: 
            state = get_local_maze_information(self.x, self.y)
            no_fire_state = state[:,:,0]
            return np.expand_dims(no_fire_state, axis=-1)

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
        elif (x,y) in self.landmarks:
            return self.penalties.landmark
        else:
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
        elif action == "stay":
            penalty = self.penalties.stay
        return (action, penalty) if valid else ("stay", penalty) 

    def _is_terminal(self):
        if (self.x, self.y) == self.goal:
            print("Maze solved!\n")
            return 1
        elif self.time >= self.time_limit:
            self.timed_out = True
            return 1
        else:
            return 0

    def _check_state(self, rel_x, rel_y):
        if self.state[rel_x, rel_y, 0] == 0: # wall
            return False, self.penalties.wall
        elif self.state[rel_x, rel_y, 0] == 1:
            if self.fire:
                if self.state[rel_x, rel_y, 1] > 0: # fiya
                    return False, self.penalties.fire
                else: 
                    return True, 0
            else:
                return True, 0

class KillBlockedIn(Env):
    """
    If agent gets 'blocked in' then restart.
    Also, restrict the agents available actions at each time step
    so that they are not allowed to backtrack. 
    """
    def __init__(self, time_limit, fire):
        super().__init__(time_limit, fire)
        self.blocked = False

    def reset(self):
        self.x=1
        self.y=1
        self.time=0
        self.state = self._get_state()
        self.timed_out=False
        self.path = [(self.x, self.y)]
        self.blocked = False

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
        blocked = self._is_blocked()
        # check if we are revisiting a state
        # note, we move agent to the new x,y 
        # check if it has been here before 
        # and THEN add the current x,y to the path
        penalty += self._check_path(self.x, self.y)
        penalty += self.penalties.blocked if blocked else 0
        # add new position to path
        self.path.append((self.x, self.y))
        return self.state, self._is_terminal(), penalty

    def _is_terminal(self):
        if (self.x == 199) & (self.y == 199):
            return 1
        elif self.time >= self.time_limit:
            self.timed_out = True
            return 1
        elif self.blocked:
            print("I'm stuck!\n")
            return 1
        else:
            return 0
        
    def _is_blocked(self):
        """
        Return true if only available agent action is a revisit.
        """
        rel_x, rel_y = 1, 1 # relative position in state array
        is_revisit = lambda x,y: int(self.path.count((x,y)))
        is_wall = lambda x,y: int(self.state[x, y, 0] == 0)
        wall_or_revisit = 0
        for i in [-1, 1]:
            wall_or_revisit+=int(is_wall(rel_x, rel_y+i) or is_revisit(rel_x, rel_y+i))
            wall_or_revisit+=int(is_wall(rel_x+i, rel_y) or is_revisit(rel_x+i, rel_y))
        is_blocked = wall_or_revisit == 4
        self.blocked = is_blocked
        return is_blocked 