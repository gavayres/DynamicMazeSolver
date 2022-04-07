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

"""
Give negative reward at every timestep.
Agent should hopefully learn to minimise
timesteps.
"""
class TimeReward(Reward):
    def reward(self, env):
        return - env.time
        
