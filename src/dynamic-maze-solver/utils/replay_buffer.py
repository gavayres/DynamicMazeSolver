import random
from collections import deque


class ReplayBuffer:
    def __init__(self, size) -> None:
        self.buffer = deque(maxlen=size)

    def push(self, experience):
        self.buffer.append(experience)

    def ready(self, batch_size):
        return batch_size <= len(self.buffer)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
