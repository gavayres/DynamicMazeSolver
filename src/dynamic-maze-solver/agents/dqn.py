from collections import defaultdict
import numpy as np
import pickle
import torch
import wandb
from torch.nn import Sequential, Conv2d, Flatten, Linear, LeakyReLU, SmoothL1Loss
from torch.optim import Adam 
from utils.reward import manhattan_weights

"""
Helper function.
Convert replay buffer memory to 
a tensor stacked by num_samples.
So we can do mini-batch SGD as 
opposed to regular SGD.
"""
def batch_to_tensor(batch, device):
    state_t_list = []
    action_list = []
    state_tp1_list = []
    reward_list = []
    done_list = []
    for state_t, action_idx, state_tp1, reward, done in batch:
        state_t_list.append(state_t)
        action_list.append(torch.tensor(action_idx))
        state_tp1_list.append(state_tp1)
        reward_list.append(torch.tensor(reward))
        done_list.append(torch.tensor(done))
    return torch.stack(state_t_list).to(device), \
            torch.stack(action_list).unsqueeze(1).to(device), \
            torch.stack(state_tp1_list).to(device), \
           torch.stack(reward_list).unsqueeze(1).to(device), \
           torch.stack(done_list).unsqueeze(1).to(device)

"""
Agent implements Double Q Learning.
"""
class DQNAgent:
    def __init__(self,
    discount, 
    learning_rate,
    no_conv=False,
    fire=True
    ) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fire = fire
        self.input_size = 27 if self.fire else 18
        self.output_size = 5
        self.q_fn = self._q_fn(self.input_size, self.output_size, no_conv=no_conv)
        self.target_q_fn = self._q_fn(self.input_size, self.output_size, no_conv=no_conv)
        # send models to device
        self.q_fn.to(self.device).double()
        self.target_q_fn.to(self.device).double()
        self.discount = discount
        self.lr = learning_rate
        self.loss = SmoothL1Loss()
        self.optimizer = Adam(self.q_fn.parameters(), lr=self.lr)
        self.epsilon = 1
        self.no_conv = no_conv


    def _q_fn(self, input_size, output_size, no_conv=False):
        """
        Neural network for approximating q function.
        """
        if no_conv:
            net = Sequential(
                Flatten(),
                Linear(input_size, 256),
                LeakyReLU(),
                Flatten(),
                Linear(256, 256),
                LeakyReLU(),
                Linear(256, output_size)
            )
        else:    
            net = Sequential(
                Conv2d(2, 1,
                kernel_size=3,
                padding=1),
                LeakyReLU(),
                Flatten(),
                Linear(input_size[0]*input_size[1], 32),
                LeakyReLU(),
                Linear(32, output_size)
            )
        return net

    
    def update_target(self):
        self.target_q_fn.load_state_dict(self.q_fn.state_dict())

    def train(self, online_q_t, target_q):
        """
        Train agent given predictions from online and target networks.
        """
        loss = self.loss(online_q_t, target_q)
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.q_fn.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss.item()


    def replay(self, batch):
        """
        Sample experience from replay buffer and train agent.
        """
        state_t, action_idx, state_tp1, reward, done =  batch_to_tensor(batch, self.device)

        # prediction from 'online' network, used for action selection
        online_q_tp1 = self.q_fn(state_tp1.double())
        tp1_action = torch.argmax(online_q_tp1, dim=1).unsqueeze(1)
        # input to loss
        online_q_t = self.q_fn(state_t.double()).gather(dim=1, index=action_idx)
        # prediction from target network
        with torch.no_grad():
            target_tp1 = self.target_q_fn(state_tp1.double()).gather(dim=1, index=tp1_action)
        expected_q_t = reward + self.discount * (
            target_tp1*(1-done))
        # train network 
        loss = self.train(online_q_t.to(self.device), expected_q_t.to(self.device))
        return loss
        
    def act(self, state):
        # epsilon greedy exploration
        if np.random.rand() <= self.epsilon:
            return np.random.randint(0, 4)
        with torch.no_grad():
            q_vals = self.q_fn(
                    state.to(self.device).unsqueeze(0)
                    ) if self.no_conv else self.q_fn(
                    state.to(self.device)
                    )
        return torch.argmax(q_vals).item()

    def manhattan_act(self, state, x, y):
        """ 
        Guided epsilon greedy exploration using Manhattan distance 
        from goal state.
        """
        goal = (199, 199)
        if np.random.rand() <= self.epsilon:
            weights = manhattan_weights(x, y, goal)
            action = np.random.choice(list(range(5)), p=weights)
            return action
        q_vals = self.q_fn(state)
        return torch.argmax(q_vals).item()

    def save(self, checkpoint_dir, loss, episode, stats):
        # save online network
        torch.save({
            'episode': episode,
            'model_state_dict': self.q_fn.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            }, checkpoint_dir + "/online.pt")
        # save target network
        torch.save({
            'episode': episode,
            'model_state_dict': self.q_fn.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            }, checkpoint_dir + "/target.pt")
        # save training statistics to date
        with open(checkpoint_dir + '/stats.p', 'wb') as f:
            pickle.dump(stats, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, checkpoint_dir):
        # load online model
        online_checkpoint = torch.load(checkpoint_dir + "/online.pt")
        self.q_fn.load_state_dict(online_checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(online_checkpoint['optimizer_state_dict'])
        # load target model
        target_checkpoint = torch.load(checkpoint_dir + "/target.pt")
        self.target_q_fn.load_state_dict(target_checkpoint['model_state_dict'])
        episode = online_checkpoint['episode']
        loss = online_checkpoint['loss']
        # load stats
        try:
            with open(checkpoint_dir + '/stats.p', 'rb') as f:
                stats = pickle.load(f)
        except FileNotFoundError:
            print("No stats file exists, setting to empty dictionary.")
            stats = defaultdict(list)
        print(f"Loading agent last trained for {episode} episodes\n")
        return episode, stats



