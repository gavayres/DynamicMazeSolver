import numpy as np
from utils.reward import ManhattanReward


class RandomAgent:
    def __init__(self, actions) -> None:
        self.actions = actions
        self.weighting_scheme = ManhattanReward((199, 199))

    def act(self, env):
        weights = self._manhattan_weights(env)
        return np.random.choice(self.actions, p=weights)

    def remember(self, replay):
        pass
    
    def _manhattan_weights(self, env):
        w_up = self.weighting_scheme._distance(env.x, env.y-1)
        w_down = self.weighting_scheme._distance(env.x, env.y+1)
        w_left = self.weighting_scheme._distance(env.x-1, env.y)
        w_right = self.weighting_scheme._distance(env.x+1, env.y)
        w_stay = self.weighting_scheme._distance(env.x, env.y)
        
        weights = [1/w_up, 1/w_down, 1/w_left, 1/w_right, 1/w_stay]
        return  [weight / np.sum(weights) for weight in weights] # normalise

