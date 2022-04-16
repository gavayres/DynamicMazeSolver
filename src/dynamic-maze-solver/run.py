import logging
import torch
import wandb
from read_maze import load_maze, get_local_maze_information
from utils.env import Env
from utils.reward import TimeReward
from agents.random import RandomAgent
from agents.dqn import DQNAgent
from utils.replay_buffer import ReplayBuffer
from utils.schedules import get_epsilon_decay_schedule
from evaluation.metrics import EpisodeLoss


wandb.init(project="dynamic-maze-solver", entity="gavayres")


"""
TODO: Epsilon decay as a function of episode?
TODO: Reward to be sum of time and negative manhattan distance?
    this would promote:
        reducing manhattan distance and reducing time
        Consideration:
            - don't want manhattan distance to dominate
            - maybe use inverse manhattan distance?
            - maybe give one big reward at end = -time
TODO: Learning rate scheduler
TODO: Play one full run of epsiodes.
TODO: Run on colab GPU. 
TODO: Iron out evaluation metrics. Specifically, 
    Time taken per episode. 
TODO: How to decide when environment solved?
TODO: When env solved call run_loop one more time
    BUT save path.

TODO: Pull everything out of main and put into configurable
    run_loop().

"""
EPISODES = 300
BATCH_SIZE = 64
BUFFER_SIZE=10000
LOG_INTERVAL=50
# setup logger 
logging.basicConfig(filename='logs/run_logs.log', 
                    format='%(asctime)s %(message)s', 
                    datefmt='%m/%d/%Y %I:%M:%S %p', 
                    level=logging.DEBUG)

# load the maze into our environment
load_maze()

print(get_local_maze_information(1, 2))

time_reward = TimeReward()

# helper function to convert a numpy array to a tensor
tensorify = lambda np_array: torch.from_numpy(np_array)
# reshape [BATCH, ROWS, COLS, CHANNELS] -> [BATCH, CHANNELS, ROWS, COLS]
reshape = lambda tensor: torch.permute(tensor, (2, 0, 1))

# epsilon schedule
epsilon_schedule = get_epsilon_decay_schedule()

"""
Run loop.
Example of how it would look.

"""
def run_loop(env, 
            agent, 
            replay_buffer, 
            episodes, 
            batch_size,
            log_interval,
            epsilon_decay_schedule,
            checkpoint_dir=None,
            evaluate=None
            ):
    # return stats (as a dictionary?)
    # think about how to process stats
    # if 'evaluate' then return 'final path' of run.
    # or shortest path?
    pass




if __name__ == "__main__":
    env = Env()
    #agent = RandomAgent(env.actions)
    agent = DQNAgent((3,3,2),len(env.actions), 0.9, 0.003)
    replay_buffer = ReplayBuffer(BUFFER_SIZE)
    loss_dict = EpisodeLoss()
    # log gradients of target and online q fun approximators
    wandb.watch(agent.q_fn, log_freq=1000)
    wandb.watch(agent.target_q_fn, log_freq=1000)

    for episode in range(EPISODES):
        episode_loss=0
        num_invalid_actions=0
        env.reset()
        state_t = env.state
        # convert to tensor
        state_t = tensorify(state_t)
        # reshape for net input and add batch_size dimension
        state_t = reshape(state_t).double()
        done = False
        agent.epsilon = epsilon_schedule(episode)

        # inner loop, play game and record results
        while not done:
            action_idx = agent.act(state_t.double())
            action = env.actions[action_idx]
            #action = agent.act(env) # just for random agent
            state_tp1, done, penalty = env.update(action)
            # convert to tensor
            state_tp1 = tensorify(state_tp1)
            # reshape
            state_tp1 = reshape(state_tp1)

            reward = time_reward.reward(env) + penalty # penalise invalid actions
            replay_buffer.push(
                (state_t, action_idx, state_tp1, reward, done)
                )
            state_t = state_tp1.detach()

            num_invalid_actions+=penalty

            if replay_buffer.ready(BATCH_SIZE):
                #print("Replaying episodes. \n")
                # replay to update target network
                batch = replay_buffer.sample(batch_size=BATCH_SIZE)
                loss = agent.replay(batch)
                if env.time % LOG_INTERVAL == 0:
                    wandb.log({"loss":loss})
                    logging.debug("Agent action: %s", action)
                    logging.debug(f"Position: %d, %d", env.x, env.y)

                episode_loss += loss

            if done:
                print("Episode over. Updating target. \n")

                print(f"Time taken: {env.time+1}\n")
                print(f"Number of invalid actions: {num_invalid_actions}\n")
                loss_dict.write("loss", episode_loss/(env.time + 1))
                agent.update_target()
