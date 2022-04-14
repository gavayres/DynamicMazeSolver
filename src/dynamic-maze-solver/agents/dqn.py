import numpy as np
import torch
import wandb
from torch.nn import Sequential, Conv2d, Flatten, Linear, SmoothL1Loss, ReLU
from torch.optim import Adam 


device = torch.device('cpu')

class DQNAgent:
    def __init__(self, 
    state_size, 
    num_actions, 
    discount, 
    learning_rate, 
    epsilon=1,
    ) -> None:
        self.q_fn = self._q_fn(state_size, num_actions).to(device).double()
        self.target_q_fn = self._q_fn(state_size, num_actions).to(device).double()
        self.discount = discount
        self.lr = learning_rate
        self.loss = SmoothL1Loss()
        self.optimizer = Adam(self.q_fn.parameters(), lr=self.lr)
        self.epsilon = epsilon
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.01


    def _q_fn(self, input_size, output_size):
        """
        Neural network for approximating q function.
        """
        net = Sequential(
            Conv2d(2, 1,
            kernel_size=3,
            padding=1),
            ReLU(),
            Flatten(),
            Linear(input_size[0]*input_size[1], output_size),
            ReLU(),
            Linear(output_size, output_size)
        )
        return net

    
    def update_target(self):
        self.target_q_fn.load_state_dict(self.q_fun.state_dict())

    def train(self, state_t, target):
        # preprocess input
        #preprocess(state_t)
        # Compute Huber loss
        # take negative of prediction! Hopefully this helps convergence
        pred_reward =  - self.q_fn(state_t.double())
        loss = self.loss(pred_reward, target)
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.q_fn.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        wandb.log({"loss":loss.item()})
        return loss.item()


    def replay(self, batch):
        """
        TODO: Batch training, 
            run through loop and append all this stuff into lists
            then perform batch inference.
        """
        # Double q learning
        batch_loss = []
        for state_t, action_idx, state_tp1, reward, done in batch:
            # prediction from 'online' network
            online_pred_tp1 = - self.q_fn(state_tp1.double())
            # prediction from target network
            target_tp1 = - self.target_q_fn(state_tp1.double())
            online_pred_tp1[:, action_idx] = reward + self.discount * target_tp1[:, 
                                                            torch.argmax(online_pred_tp1)
                                                            ].item()*(1-done)
            # train network 
            loss = self.train(state_t, online_pred_tp1)
            batch_loss.append(loss)
        # epsilon decay at the end of replay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return np.mean(batch_loss)
        



    def act(self, state):
        # epsilon greedy exploration
        if np.random.rand() <= self.epsilon:
            return np.random.randint(0, 4)
        q_vals = self.q_fn(state)
        return torch.argmax(q_vals).item()

