import numpy as np


class Reward:
    def _distance(self, x, y):
        pass

    def reward(self, state):
        return self._distance(state.x, state.y)


class ManhattanReward(Reward):
    def __init__(self, goal_pos) -> None:
        self.goal_pos = goal_pos
    
    """
    Returns negative manhattan distance 
    between two x,y co-ordinates.
    """
    def _distance(self, x, y):
        x_dist = abs(x - self.goal_pos[0])
        y_dist = abs(y - self.goal_pos[1])
        return -(x_dist + y_dist)

def manhattan_distance(pos1, pos2):
    """
    Arguments:
        pos1 (tuple[int])
        pos2 (tuple[int])
    returns:
        manhattan distance
    """
    x_dist = abs(pos1[0] - pos2[0])
    y_dist = abs(pos1[1] - pos2[1])
    return (x_dist + y_dist)

def manhattan_weights(x, y, goal):
    w_up = manhattan_distance((x, y-1), goal)
    w_down = manhattan_distance((x, y+1), goal)
    w_left = manhattan_distance((x-1, y), goal)
    w_right = manhattan_distance((x+1, y), goal)
    w_stay = manhattan_distance((x, y), goal)
    
    distances = [w_up, w_down, w_left, w_right, w_stay]
    weights = [1/distance if distance != 0 else 1 for distance in distances]
    return  [weight / np.sum(weights) for weight in weights] # normalise


"""
Give negative reward at every timestep.
Agent should hopefully learn to minimise
timesteps.
TODO: Scale reward by max possible reward (the timeout number of timesteps)
"""
class TimeReward(Reward):
    def reward(self, env):
        return - env.time

class BasicReward:
    """
    TODO: Change to some positive reward for reducing manhattan 
    distance and then a big one for finishing? 
    - Maybe a potential function for manhattan distance?
    """
    def reward(self, env):
        if (env._is_terminal()) & (env.timed_out == False):
            return 10
        elif env.timed_out:
            return -1
        # give more negative reward for moving further from the goal
        old_pos = (env.old_x, env.old_y)
        curr_pos = (env.x, env.y)
        if manhattan_distance(old_pos, env.goal) < manhattan_distance(curr_pos, env.goal):
            return -0.02 # penalise moving further away from the goal
        return -0.01  # expected reward of zero if manhattan shortest path taken?
        
