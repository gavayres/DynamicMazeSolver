import logging
import torch
from read_maze import load_maze, get_local_maze_information
from utils.env import Env
from utils.reward import TimeReward
from agents.random import RandomAgent
from agents.dqn import DQNAgent
from utils.replay_buffer import ReplayBuffer


EPISODES = 100
BATCH_SIZE = 50
# setup logger 
logging.basicConfig(filename='logs/run_logs.log', 
                    format='%(asctime)s %(message)s', 
                    datefmt='%m/%d/%Y %I:%M:%S %p', 
                    level=logging.DEBUG)

# load the maze into our environment
load_maze()

print(get_local_maze_information(1, 2))

time_reward = TimeReward()

replay_time=False

# helper function to convert a numpy array to a tensor
tensorify = lambda np_array: torch.from_numpy(np_array)
# reshape [BATCH, ROWS, COLS, CHANNELS] -> [BATCH, CHANNELS, ROWS, COLS]
reshape_input = lambda tensor: torch.permute(tensor, (2, 0, 1))


if __name__ == "__main__":
    env = Env()
    #agent = RandomAgent(env.actions)
    agent = DQNAgent((3,3,2), 0.9, 0.003)
    replay_buffer = ReplayBuffer(200)

    for episode in range(EPISODES):
        env.reset()
        state_t = env.state
        # convert to tensor
        state_t = tensorify(state_t)
        # reshape for net input and add batch_size dimension
        state_t = state_t.permute((2, 0, 1)).unsqueeze(0)
        print(state_t.size())
        done = False

        # inner loop, play game and record results
        while not done:
            action = agent.act(state_t)
            #action = agent.act(env) # just for random agent
            state_tp1, done = env.update(action)
            # convert to tensor
            state_tp1 = tensorify(state_tp1)

            reward = time_reward.reward(env)
            replay_buffer.push(
                (state_t, action, state_tp1, reward, done)
                )
            state_t = state_tp1.detach()
            print(type(state_t))

            logging.debug("Agent action: %s", action)
            logging.debug(f"Position: %d, %d", env.x, env.y)

            if replay_buffer.ready():
                # replay to update target network
                batch = replay_buffer.sample(batch_size=BATCH_SIZE)
                loss = agent.replay(batch)
            if done:
                agent.update_target()
