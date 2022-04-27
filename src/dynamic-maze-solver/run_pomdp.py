from collections import defaultdict
import logging
import torch
import wandb
import numpy as np
import yaml
import json
from agents.drqn import DRQNAgent
from read_maze import load_maze, get_local_maze_information
from utils.env import TestEnv, Env
from utils.reward import TimeReward, BasicReward
from agents.random import RandomAgent
from agents.dqn import DQNAgent
from utils.buffers import ReplayBuffer, EpisodeBuffer, EpisodeMemory
from utils.schedules import get_epsilon_decay_schedule
from evaluation.metrics import EpisodeLoss


wandb.init(project="dynamic-maze-solver", entity="gavayres")
# load config file
with open("agents/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# set random seed
np.random.RandomState(config['DQN']['random_seed'])

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

TODO: Timeout
TODO: Let agent move into fire but increment reward by how ling the fire remains 
    in the grid cell?



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

# counter
def counter():
    count=0
    while True:
        yield count
        count+=1

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
            save_interval,
            epsilon_decay_schedule,
            RewardClass,
            checkpoint_dir=None,
            evaluate=None,
            update_target_interval=10
            ):
    # return stats (as a dictionary?)
    # think about how to process stats
    # if 'evaluate' then return 'final path' of run.
    # or shortest path?
    stats = defaultdict(list)
    save_num = counter() # counter to record the number of saves
    loss=0

    for episode in range(episodes):
        episode_loss=[]
        num_invalid_actions=0
        env.reset()
        episode_record = EpisodeBuffer()
        # initialise hidden state for lstm
        h, c = agent.q_fn.init_hidden_state(batch_size=batch_size, training=False)
        state_t = env.state
        # convert to tensor
        state_t = tensorify(state_t)
        # reshape for net input and add batch_size dimension
        state_t = reshape(state_t).double()
        done = False
        agent.epsilon = epsilon_decay_schedule(episode)
        print(f"Epsilon: {agent.epsilon}\n")

        # inner loop, play game and record results
        while not done:
            #------ uncomment -----# 
            logging.debug(f"Initial h shape{h.size()}\n")
            logging.debug(f"Initial c shape {c.size()}\n")
            action_idx, h, c = agent.act(state_t.double(), h.double(), c.double())
            action = env.actions[action_idx]
            #------------------#
            state_tp1, done, penalty = env.update(action)
            # convert to tensor
            state_tp1 = tensorify(state_tp1)
            # reshape
            state_tp1 = reshape(state_tp1)

            reward = RewardClass.reward(env) + penalty # penalise invalid actions
            #TODO: Implement episode record
            episode_record.push(
                (state_t, action_idx, state_tp1, reward, done)
                )
            logging.debug(f"Size of episode record: {len(episode_record)}\n")
            state_t = state_tp1.detach()
            # record the number of invalid actions
            num_invalid_actions+=penalty



            if replay_buffer.ready(replay_buffer.batch_size):
                print("Replaying episodes. \n")
                # replay to update target network
                batch, seq_len = replay_buffer.sample()
                loss = agent.replay(batch, seq_len)
                episode_loss.append(loss) # record loss 
                if env.time % log_interval == 0:
                    wandb.log({"loss":loss})
                    logging.debug("Agent action: %s", action)
                    logging.debug(f"Position: %d, %d", env.x, env.y)
                if env.time % save_interval == 0:
                    agent.save(checkpoint_dir, loss, next(save_num))
                if (env.time +1) % update_target_interval == 0:
                    # update target network
                    agent.update_target()

            if done == 1:
                print("Episode over. Updating target. \n")
                print(f"Number of invalid actions: {num_invalid_actions}\n")
                # update stats
                stats["time"].append(env.time) # time taken to finish maze
                stats["invalid_actions"].append(num_invalid_actions) # number of invalid actions taken
                stats["average_loss"].append(np.mean(episode_loss))
                stats["std_loss"].append(np.std(episode_loss))
                agent.save(checkpoint_dir, loss, next(save_num))
        logging.debug(f"Size of episode memory: {len(replay_buffer)}\n")
        replay_buffer.push(episode_record)
    return stats





if __name__ == "__main__":
    checkpoint_dir = "./checkpoints"
    agent = DRQNAgent((3,3,2),5, config['DQN']['gamma'], config['DQN']['learning_rate'])
    wandb.watch(agent.q_fn)
    wandb.watch(agent.target_q_fn)
    stats = run_loop(env=TestEnv(time_limit=10000),
                #agent=RandomAgent(TestEnv(time_limit=10000).actions),
                agent=DRQNAgent((3,3,2),5, config['DQN']['gamma'], config['DQN']['learning_rate']),
                replay_buffer=EpisodeMemory(
                    batch_size=2, 
                    max_epi_num=10, 
                    max_seq_len=500,
                    random_update=True, 
                    lookup_size=250
                    ), 
                episodes=config['DQN']['episodes'],
                batch_size=config['DQN']['batch_size'],
                log_interval=config['DQN']['log_interval'],
                save_interval=config['DQN']['save_interval'],
                epsilon_decay_schedule=epsilon_schedule,
                RewardClass=BasicReward(),
                checkpoint_dir=checkpoint_dir,
                evaluate=False)
    # save stats
    with open('./logs/stats.json', "w") as stats_file:
        json.dump(stats, stats_file)
