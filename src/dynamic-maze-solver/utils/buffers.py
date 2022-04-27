import random
import numpy as np
import torch
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


class EpisodeBuffer:
    """
    TODO: Change types to be consistent NN input and 
    existing DQN class.

    A simple numpy replay buffer."""

    def __init__(self):
        self.obs = []
        self.action = []
        self.reward = []
        self.next_obs = []
        self.done = []

    def push(self, transition):
        self.obs.append(transition[0])
        self.action.append(torch.tensor(transition[1]))
        self.next_obs.append(transition[2])
        self.reward.append(torch.tensor(transition[3]))
        self.done.append(torch.tensor(transition[4]))

    def sample(self, random_update=False, lookup_step=None, idx=None):
        # stack our tensors, first dimension now sequence length
        obs = torch.stack(self.obs)#np.array(self.obs)
        action = torch.stack(self.action).unsqueeze(1)#np.array(self.action)
        reward = torch.stack(self.reward).unsqueeze(1)#np.array(self.reward)
        next_obs = torch.stack(self.next_obs)#np.array(self.next_obs)
        done = torch.stack(self.done).unsqueeze(1)#np.array(self.done)

        if random_update is True:
            obs = obs[idx:idx+lookup_step, :]
            action = action[idx:idx+lookup_step, :]
            reward = reward[idx:idx+lookup_step, :]
            next_obs = next_obs[idx:idx+lookup_step, :]
            done = done[idx:idx+lookup_step, :]

        return dict(obs=obs,
                    acts=action,
                    rews=reward,
                    next_obs=next_obs,
                    done=done)

    def __len__(self) -> int:
        return len(self.obs)


class EpisodeMemory:
    """
    TODO: Change types to be consisten with existing DQN.
    """
    def __init__(self, batch_size, max_epi_num, max_seq_len, random_update, lookup_size) -> None:
        self.batch_size = batch_size
        self.random_update = random_update
        self.lookup_size = lookup_size
        self.max_epi_num = max_epi_num
        self.max_seq_len = max_seq_len
        self.memory = deque(maxlen=self.max_epi_num)

    def push(self, episode):
        self.memory.append(episode)

    def sample(self):
        """
        Returns a set of experience corresponding 
        to a sequence in an episode. First we randomly choose episodes
        then from these episodes we get a sequence of experience and 
        add these random sequences together to a list which is then returned
        and used for training.
        """
        sampled_buffer=[]
        if self.random_update:
            # randomly choose exp sequences from sampled episodes
            sampled_episodes = random.sample(self.memory, self.batch_size)
            
            check_flag = True # check if every sample data to train is larger than batch size
            min_step = self.max_seq_len

            for episode in sampled_episodes:
                min_step = min(min_step, len(episode)) # get minimum step from sampled episodes

            for episode in sampled_episodes:
                if min_step > self.lookup_size: # sample buffer with lookup_step size
                    idx = np.random.randint(0, len(episode)-self.lookup_size+1)
                    sample = episode.sample(random_update=self.random_update, lookup_step=self.lookup_size, idx=idx)
                    sampled_buffer.append(sample)
                else:
                    idx = np.random.randint(0, len(episode)-min_step+1) # sample buffer with minstep size
                    sample = episode.sample(random_update=self.random_update, lookup_step=min_step, idx=idx)
                    sampled_buffer.append(sample)

        ##################### SEQUENTIAL UPDATE ############################           
        else: # Sequential update
            idx = np.random.randint(0, len(self.memory))
            sampled_buffer.append(self.memory[idx].sample(random_update=self.random_update))

        return sampled_buffer, len(sampled_buffer[0]['obs']) # buffers, sequence_length

    def ready(self, batch_size):
        return batch_size <= len(self.memory)


    def __len__(self):
        return len(self.memory)