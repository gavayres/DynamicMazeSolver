import numpy as np
import wandb
import torch
from torch.nn import Sequential, Conv2d, Flatten, Linear, MSELoss, LeakyReLU, LSTM, Module
from torch.optim import Adam 
from agents.dqn import DQNAgent
import logging

logger = logging.getLogger(__name__)

class DRQN(Module):
    def __init__(self, input_size, output_size, hidden_size=32) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.conv2d = Conv2d(2, 1, kernel_size=3, padding=1)
        self.relu = LeakyReLU()
        self.flatten = Flatten()
        self.linear = Linear(input_size[0]*input_size[1], self.hidden_size)
        self.lstm = LSTM(hidden_size, self.hidden_size, batch_first=True)
        self.out_linear = Linear(self.hidden_size, output_size)

    def forward(self, x, h, c):
        logger.debug(f"Input x shape{x.size()}\n")
        # reshape x to [BATCH_SIZE*SEQ_LEN, CHANNELS, ROWS, COLS]
        batch_size, seq_len = x.size()[0], x.size()[1]
        x = x.reshape(1, batch_size*seq_len, x.size()[-3], x.size()[-2], x.size()[-1]).squeeze(0)
        logger.debug(f"New x shape{x.size()}\n")
        x = self.conv2d(x)
        x = self.flatten(self.relu(x))
        x = self.linear(x)
        logger.debug(f"After conv, x shape{x.size()}\n")
        # reshape back to [BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE]
        x = x.reshape(batch_size, seq_len, self.hidden_size)
        logger.debug(f"Pre LSTM x shape{x.size()}\n")
        x, (h_out, c_out) = self.lstm(x, (h, c))
        x = self.out_linear(x)
        logger.debug(f"Output x shape{x.size()}\n")
        return x, h_out, c_out

    def init_hidden_state(self, batch_size, training=None):
        """
        Hidden state should be zeroed before training on an episode.
        If evaluating then batch size will be one because we are just 
        sequentially feeding in observations as the agent plays the game.
        """
        assert training is not None, "training step parameter should be dtermined"

        if training is True:
            return torch.zeros([1, batch_size, self.hidden_size]), torch.zeros([1, batch_size, self.hidden_size])
        else:
            return torch.zeros([1, 1, self.hidden_size]), torch.zeros([1, 1, self.hidden_size])


class DRQNAgent(DQNAgent):
    """
    Deep recurrent q network agent.
    """
    def __init__(self, state_size, num_actions, discount, learning_rate, epsilon=1) -> None:
        super().__init__(state_size, num_actions, discount, learning_rate, epsilon)

    def _q_fn(self, input_size, output_size, hidden_size=32):
        """
        Regular Q but with an lstm as the LAST layer to process 
        sequences.
        """
        net = DRQN(input_size=input_size, output_size=output_size, hidden_size=hidden_size)
        return net

    def replay(self, batch, seq_len):
        observations = []
        actions = []
        rewards = []
        next_observations = []
        dones = []
        batch_size = len(batch)

        for i in range(len(batch)):
            observations.append(batch[i]["obs"])
            actions.append(batch[i]["acts"])
            rewards.append(batch[i]["rews"])
            next_observations.append(batch[i]["next_obs"])
            dones.append(batch[i]["done"])

        logging.debug(f"obs type: {type(observations)}\n")
        logging.debug(f"obs shape: {type(observations[0])}\n")
        observations = torch.cat(observations, dim=0) #np.array(observations)
        actions = torch.cat(actions, dim=0)#np.array(actions)
        rewards = torch.cat(rewards, dim=0) #np.array(rewards)
        next_observations = torch.cat(next_observations, dim=0)#np.array(next_observations)
        dones = torch.cat(dones, dim=0) #np.array(dones)

        # should be [BATCH_SIZE, SEQ_LEN, FEATURE SIZE]
        logging.debug(f"Obs shape: {observations.shape}\n")

        device='cpu'

        feature_size=(2,3,3)

        observations = observations.reshape(batch_size,seq_len, *feature_size).double().to(device)
        actions = actions.reshape(batch_size,seq_len,-1).to(device)
        rewards = rewards.reshape(batch_size,seq_len,-1).double().to(device)
        next_observations = next_observations.reshape(batch_size,seq_len, *feature_size).double().to(device)
        dones = dones.reshape(batch_size,seq_len,-1).double().to(device)

        logging.debug(f"Tensor obs shape: {observations.shape}\n")
        #state_t, action_idx, state_tp1, reward, done =  batch_to_tensor(batch)
        # reset lstm hidden states for episode
        h_target, c_target = self.target_q_fn.init_hidden_state(batch_size=batch_size, training=True)
        h_online, c_online = self.q_fn.init_hidden_state(batch_size=batch_size, training=True)


        # prediction from 'online' network, used for action selection
        online_q_tp1, _, _ = self.q_fn(next_observations.double(), h_online.double(), c_online.double())
        logging.debug(f"Online q tp1 shape: {online_q_tp1.size()}\n")
        tp1_action = torch.argmax(online_q_tp1, dim=1).unsqueeze(1)
        logging.debug(f"Tp1 action shape: {online_q_tp1.size()}\n")
        # input to loss
        online_q_t, _, _ = self.q_fn(observations.double(), h_online.double(), c_online.double())
        online_action_q_t = online_q_t.gather(dim=1, index=actions) # take the q value which corresponded to the actions taken
        logging.debug(f"Online q shape: {online_q_t.size()}\n")
        # prediction from target network
        target_tp1, _, _ = self.target_q_fn(next_observations.double(), h_target.double(), c_target.double())
        target_action_tp1 = target_tp1.gather(dim=1, index=tp1_action)
        logging.debug(f"Target tp1 shape: {target_tp1.size()}\n")
        #mask = torch.zeros_like(online_pred_tp1)
        # boolean mask at column indices indicating action
        expected_q_t = rewards + self.discount * (
            target_action_tp1*(1-dones))

        loss = self.train(online_action_q_t, expected_q_t)
        return loss

    def act(self, state, h, c):
        # for input to nn have to add BATCH_SIZE and SEQ_LEN dimensions
        q_vals, h_new, c_new = self.q_fn(state.unsqueeze(0).unsqueeze(0), h.double(), c.double())
        logger.debug(f"q_vals shape: {q_vals.size()}\n")
        logger.debug(f"h shape {h_new.size()}\n")
        logger.debug(f"c shape: {c_new.size()}\n")
        # epsilon greedy exploration
        if np.random.rand() <= self.epsilon:
            return np.random.randint(0, 4), h_new, c_new
        logger.debug(f"Agent action: {torch.argmax(q_vals).item()}\n")
        logger.debug(f"Agent q vals: {q_vals}\n")
        return torch.argmax(q_vals).item(), h_new, c_new
