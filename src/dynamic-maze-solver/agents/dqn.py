import numpy as np
import torch
from torch.nn import Sequential, Conv2d, Flatten, Linear, SmoothL1Loss, ReLU
from torch.optim import Adam 

device = torch.device('cpu')

def play():
    pass

def replay_buffer():
    pass

def q_function():
    pass

class DQNAgent:
    def __init__(self, state_size, num_actions, discount, learning_rate) -> None:
        self.q_fn = self._q_fn(state_size, num_actions).to(device).double()
        self.target_q_fn = self._q_fn(state_size, 1).to(device).double()
        self.discount = discount
        self.lr = learning_rate
        self.loss = SmoothL1Loss()
        self.optimizer = Adam(self.q_fn.parameters(), lr=self.lr)


    def _q_fn(self, input_size, output_size):
        """
        Neural network for approximaating q function.
        """
        net = Sequential(
            Conv2d(2, 1,
            kernel_size=3,
            padding=0),
            ReLU(),
            #Flatten(),
            Linear(1, output_size),
            #ReLU(),
            #Linear(output_size, 1)
        )
        return net

    
    def update_target(self):
        self.target_q_fn.load_state_dict(self.q_fun.state_dict())

    def train(self, state_t, target):
        # Compute Huber loss
        pred_reward = self.q_fn(state_t.double())
        loss = self.loss(pred_reward, target)
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.q_fn.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()


    def replay(self, batch):
        #TODO: Epsilon decay
        for state_t, action_idx, state_tp1, reward, done in batch:
            target = self.q_fn(state_t.double())
            if done:
                target[:,:, action_idx] = reward 
            else: 
                target_tp1 = self.target_q_fn(state_tp1.double())
                target[:,:, action_idx] = reward + self.discount * torch.argmax(target_tp1).item()
            # train network 
            #self.q_fn.fit(state_t, target, epochs=1, verbose=0)
            self.train(state_t, target)


    def act(self, state):
        q_vals = self.q_fn(state)
        return torch.argmax(q_vals).item()

