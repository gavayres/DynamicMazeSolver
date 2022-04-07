import logging
from read_maze import load_maze, get_local_maze_information
from utils.env import Env
from utils.reward import TimeReward
from agents.random import RandomAgent

EPISODES = 100
# setup logger 
logging.basicConfig(filename='run_logs.log', 
                    format='%(asctime)s %(message)s', 
                    datefmt='%m/%d/%Y %I:%M:%S %p', 
                    level=logging.DEBUG)

# load the maze into our environment
load_maze()

print(get_local_maze_information(1, 2))

ACTIONS = ['up', 'down', 'left', 'right', 'stay']
time_reward = TimeReward()

replay_time=False
if __name__ == "__main__":
    agent = RandomAgent(ACTIONS)
    env = Env()
    for episode in range(EPISODES):
        state_t = env.reset()
        done = False
        # inner loop, play game and record results
        while not done:
            #action = agent.act(state_t)
            action = agent.act(env) # just for random agent
            state_tp1, done = env.update(action)
            reward = time_reward.reward(env)
            agent.remember(
                (state_t, action, state_tp1, reward)
                )
            state_t = state_tp1

            logging.debug("Agent action: %s", action)
            logging.debug(f"Position: %d, %d", env.x, env.y)

            if replay_time:
                # replay to update target network
                loss = agent.replay()
