import numpy as np
import torch
from torch.nn import Sequential, Conv2d, Flatten, Linear, SmoothL1Loss
from torch.optim import Adam 

device = torch.device('cpu')

def play():
    pass

def replay_buffer():
    pass

def q_function():
    pass

class DQNAgent:
    def __init__(self, state_size, discount, learning_rate) -> None:
        self.q_fn = self._q_fn(state_size, 1).to(device)
        self.target_q_fn = self._q_fn(state_size, 1)
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
            kernel_size=3),
            Flatten(),
            Linear(3, output_size)
        )
        return net

    
    def update_target(self):
        self.target_q_fn.load_state_dict(self.q_fun.state_dict())

    def train(self, state_t, target):
        # Compute Huber loss
        loss = self.loss(state_t, target.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.q_fn.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()


    def replay(self, batch):
        #TODO: Epsilon decay
        for state_t, action, state_tp1, reward, done in batch:
            target = self.q_fn(state_t)
            if done:
                target[action] = reward 
            else: 
                target_p1 = self.target_q_fn(state_tp1)
                target[action] = reward + self.discount * np.argmax(target_p1)
            # train network 
            #self.q_fn.fit(state_t, target, epochs=1, verbose=0)
            self.train(state_t, target)

    def act(self, state):
        q_vals = self.q_fn(state)
        return np.argmax(q_vals)

