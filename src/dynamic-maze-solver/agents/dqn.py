import numpy as np
import torch
import wandb
from torch.nn import Sequential, Conv2d, Flatten, Linear, MSELoss, LeakyReLU, SmoothL1Loss
from torch.optim import Adam 
from utils.reward import manhattan_weights


#device = torch.device('cpu')

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
TODO: Normalise state input.
"""

class DQNAgent:
    def __init__(self, 
    state_size, 
    num_actions, 
    discount, 
    learning_rate, 
    epsilon=1,
    no_conv=False
    ) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.q_fn = self._q_fn(state_size, num_actions, no_conv=no_conv)
        self.target_q_fn = self._q_fn(state_size, num_actions, no_conv=no_conv)
        # send models to device
        self.q_fn.to(self.device).double()
        self.target_q_fn.to(self.device).double()
        self.discount = discount
        self.lr = learning_rate
        self.loss = SmoothL1Loss()#MSELoss()
        self.optimizer = Adam(self.q_fn.parameters(), lr=self.lr)
        self.epsilon = 1
        self.no_conv = no_conv
        #self.epsilon_decay = 0.99
        #self.epsilon_min = 0.01


    def _q_fn(self, input_size, output_size, no_conv=False):
        """
        Neural network for approximating q function.
        TODO: - how to initialise
              - is this the best architecture?
              - should this be the loss used, maybe MSE?
              - how to initialise?
              - Option to NOT USE CONV and just flatten instead
        """
        if no_conv:
            net = Sequential(
                Flatten(),
                Linear(input_size, 32),
                LeakyReLU(),
                Flatten(),
                Linear(32, 32),
                LeakyReLU(),
                Linear(32, output_size)
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
        TODO: Evaluation metrics.
        """
        # preprocess input
        #preprocess(state_t)
        # Compute Huber loss
        # take negative of prediction! Hopefully this helps convergence

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
        TODO: GPU training
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
        #mask = torch.zeros_like(online_pred_tp1)
        # boolean mask at column indices indicating action
        expected_q_t = reward + self.discount * (
            target_tp1*(1-done))



        #mask.index_fill_(1, action_idx, 1)
        #print(online_pred_tp1[mask.bool()].shape)
        #online_pred_tp1[mask.bool()]  = reward + self.discount * torch.masked_select(target_tp1[:, 
        #                                                    torch.argmax(online_pred_tp1, dim=1)
        #                                                    ], ~done)
        # train network 
        loss = self.train(online_q_t.to(self.device), expected_q_t.to(self.device))
        # epsilon decay
        #if self.epsilon > self.epsilon_min:
        #    self.epsilon *= self.epsilon_decay

        return loss
        
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
        """
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
        # guided epsilon greedy exploration
        goal = (199, 199)
        if np.random.rand() <= self.epsilon:
            weights = manhattan_weights(x, y, goal)
            action = np.random.choice(list(range(5)), p=weights)
            return action
        q_vals = self.q_fn(state)
        return torch.argmax(q_vals).item()

    def save(self, checkpoint_dir, loss, epoch):
        # save online network
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.q_fn.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            }, checkpoint_dir + "/online.pt")
        # save target network
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.q_fn.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            }, checkpoint_dir + "/target.pt")

    def load(self, checkpoint_dir):
        online_checkpoint = torch.load(checkpoint_dir + "/online.pt")
        self.q_fn.load_state_dict(online_checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(online_checkpoint['optimizer_state_dict'])
        epoch = online_checkpoint['epoch']
        loss = online_checkpoint['loss']


