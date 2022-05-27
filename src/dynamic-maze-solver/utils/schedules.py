
def power_decay_schedule(episode_number: int,
                         decay_factor: float,
                         minimum_epsilon: float) -> float:
    """
    Power decay schedule found in other practical applications.
    Adapted from 
    https://github.com/davidrpugh/stochastic-expatriate-descent/blob/2020-04-03-deep-q-networks/_notebooks/2020-04-03-deep-q-networks.ipynb
    """
    return max(decay_factor**episode_number, minimum_epsilon)

def get_epsilon_decay_schedule():
    _epsilon_decay_schedule_kwargs = {
        "decay_factor": 0.99,
        "minimum_epsilon": 1e-2,
    }
    epsilon_decay_schedule = lambda n: power_decay_schedule(n, **_epsilon_decay_schedule_kwargs)
    return epsilon_decay_schedule
