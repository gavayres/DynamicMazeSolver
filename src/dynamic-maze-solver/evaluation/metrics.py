"""
Metrics of interest:
    - Average reward per episode
    - Number of invalid actions chosen per episode
    vs. number of episodes
    - Config settings for run
    - Training loss per 
        - episode
        - steps

    Should be saved to one foler, 
    then have w&b read from logs folder 
    and set up dashboarcs
    TODO:
        - what format does w&b need logs to be in?
        - what data structure to write to?
            - start with list, maybe try deque
            - w&b writes to logs from dictionary of 
            arrays / lists 
            e.g.
            # define our custom x axis metric
                wandb.define_metric("train/step")
            # set all other train/ metrics to use this step
                wandb.define_metric("train/*", step_metric="train/step")

"""

class Metric:
    def __init__(self) -> None:
        self.metrics_dict = {}

    def write(self, key, metric):
        self.metrics_dict[key].append(metric)

class EpisodeLoss(Metric):
    """
    Metric class for saving average 
    neural network loss per episode.
    """
    def __init__(self) -> None:
        super().__init__()
        self.metrics_dict['loss'] = []
        self.metrics_dict['episode'] = []

class Loss(Metric):
    def __init__(self) -> None:
        super().__init__()
        self.metrics_dict['loss'] = []
        