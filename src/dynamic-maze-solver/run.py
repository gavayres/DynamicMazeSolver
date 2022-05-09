from collections import defaultdict
import logging
import torch
import wandb
import numpy as np
import yaml
import json
import argparse
from read_maze import load_maze, get_local_maze_information
from utils.env import TestEnv, Env
from utils.reward import CheeseReward, TimeReward, BasicReward, ManhattanReward
from agents.random import RandomAgent
from agents.dqn import DQNAgent
from utils.buffers import ReplayBuffer
from utils.schedules import get_epsilon_decay_schedule
from evaluation.metrics import EpisodeLoss

# set up argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--load_path', type=str, default=None, help='Load an agent from the given checkpoint path.')
args = parser.parse_args()

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

TODO: Pull everything out of main and put into configurable
    run_loop(). 

TODO: Timeout
TODO: Let agent move into fire but increment reward by how ling the fire remains 
    in the grid cell?

TODO: Check that agent is predicting if has been in a cell before or not.
TODO: What would an ideal policy do? It would follow the cheese and avoid the dead ends. 
If we have an extra penalty for dead ends then how would an agent learn to relate that to a state?
Add another indicator if one of the nearby grid cells is a dead end or not? 
Input would then by [3, 3, 4] first two dims are wall, fire indicators, third is memory, fourth is dead end.

TODO: Why is it that our loss converges to zero straight away?

TODO: How can we check what our agent is actually learning?

NOTE: I WASNT NORMALISING TH EFFING STATE IN THE EVALUATION STEP!!!!
        NO WONDER MY FRIGGING AGENT NEVER WENT ANYWHERE FFS!!

NOTE: How would an agent actually learn a good policy from 3by3 frames?
    It would have to learn that it should move into a region with some cheese?
    But cheese doesn't always lead to good long term reward....
    suppose in two identical 3by3 grids, one with high expected future reward and one with low
    How would an agent distinguish? Not possible.
    Have to encode position somehow..?
    Add x,y co-ordinate as input? 
        - two extra dimensions?
        - one for x co-ordinate
        - one for y co-ordinate
        Is this cheating in some sense?
        - Would be difficult though to make any sense out of previously unobserved 
        states
NOTE:
    Would there possibly be any better state representations?

NOTE: 
    Should I remove staying as an option to the agent?
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

def normalise_state(state):
    """
    Normalises state input by max possible values for fire channels
    """
    norm_state = state
    norm_state[:, :, 1] = norm_state[:, :, 1] / 2 # max number of seconds fire can appear for
    return norm_state

def get_agent_history(agent_pos, agent_path):
    x,y = agent_pos
    # want to check if pos close to x,y 
    # exist in list of tuples
    agent_history = np.zeros((3,3))
    in_path = lambda x, y: (x, y) in agent_path
    agent_history[0, 0] = in_path(x-1, y+1)
    agent_history[0, 1] = in_path(x, y+1)
    agent_history[0, 2] = in_path(x+1, y+1)
    agent_history[1, 0] = in_path(x-1, y)
    agent_history[1, 1] = 0 # current location so obvs in path
    agent_history[1, 2] = in_path(x+1, y)
    agent_history[2, 0] = in_path(x-1, y-1)
    agent_history[2,1] = in_path(x, y-1)
    agent_history[2,2] = in_path(x+1, y+1)
    return agent_history


def add_state_memory(state, agent_pos, agent_path):
    """
    Add binary indicator to each location in observation
    indicating if the agent has been to that location before.
    """
    # get array of indicators
    history = np.expand_dims(get_agent_history(agent_pos, agent_path), -1)
    new_state = np.concatenate((state, history), axis=-1)
    return new_state



def evaluate_loop(env, agent, RewardClass, state_memory=False):
    env.reset()
    state_t = env.state
    if state_memory:
        # add new dimension indicating if agent has been in loc
        state_t = add_state_memory(state_t, (env.x, env.y), env.path)
    state_t = tensorify(state_t)
    state_t = reshape(state_t).double()
    done = False
    episode_reward=0
    agent.epsilon = 1e-2 # set to min epsilon
    # run on maze
    while not done:
        action_idx = agent.act(state_t.double())
        action = env.actions[action_idx]
        state_tp1, done, penalty = env.update(action)
        if state_memory:
            # add new dimension indicating if agent has been in loc
            state_tp1 = add_state_memory(state_tp1, (env.x, env.y), env.path)
        state_tp1 = tensorify(state_tp1)
        state_tp1 = reshape(state_tp1)
        reward = RewardClass.reward(env) + penalty
        episode_reward+=reward
        state_t = state_tp1.detach()
        if done==1:
            print(f"Time taken: {env.time}\n")
            print(f"Agent position: {env.x}, {env.y}\n")
            print(f"Total reward: {episode_reward}")
            if env.timed_out == False:
                return 1 # agent completed the maze!
    return 0
            



"""
Run loop.
Example of how it would look.
"""
def train_loop(env, 
            agent, 
            replay_buffer, 
            episodes, 
            batch_size,
            log_interval,
            save_interval,
            epsilon_decay_schedule,
            RewardClass,
            checkpoint_dir=None,
            start_episode=0,
            evaluate_interval=3, # evaluate every three episodes
            stats=defaultdict(list),
            state_memory=False
            ):
    # return stats (as a dictionary?)
    # think about how to process stats
    # if 'evaluate' then return 'final path' of run.
    # or shortest path?
    total_success=0

    for episode in range(start_episode, episodes):
        episode_loss=[]
        num_invalid_actions=0
        episode_reward=0
        env.reset()
        state_t = env.state
        if state_memory:
            # add new dimension indicating if agent has been in loc
            state_t = add_state_memory(state_t, (env.x, env.y), env.path)
        #state_t = normalise_state(state_t) # NEW
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
            action_idx = agent.act(state_t.double())
            #action_idx = agent.manhattan_act(state_t.double(), env.x, env.y)
            #----------------------#
            #action = agent.act(env) # TESTING: DELETE LATER
            #action_idx=1 # TESTING: DELETE LATER
            #--- uncomment -- #
            action = env.actions[action_idx]
            #------------------#
            state_tp1, done, penalty = env.update(action)
            if state_memory:
                # add new dimension indicating if agent has been in loc
                state_tp1 = add_state_memory(state_tp1, (env.x, env.y), env.path)

            #state_tp1 = normalise_state(state_tp1) # NEW
            # convert to tensor
            state_tp1 = tensorify(state_tp1)
            # reshape
            state_tp1 = reshape(state_tp1)

            reward = RewardClass.reward(env) + penalty # penalise invalid actions
            # add to episode reward
            episode_reward+=reward
            replay_buffer.push(
                (state_t, action_idx, state_tp1, reward, done)
                )
            state_t = state_tp1.detach()
            # record the number of invalid actions
            num_invalid_actions+=penalty

            if replay_buffer.ready(batch_size):
                #print("Replaying episodes. \n")
                # replay to update target network
                batch = replay_buffer.sample(batch_size=batch_size)
                loss = agent.replay(batch)
                episode_loss.append(loss) # record loss 
                if env.time % log_interval == 0:
                    wandb.log({"loss":loss})
                    logging.debug("Agent action: %s", action)
                    logging.debug(f"Position: %d, %d", env.x, env.y)
                if env.time % save_interval == 0:
                    agent.save(checkpoint_dir, loss, episode, stats)
                    


            if done == 1:
                print("Episode over. Updating target. \n")
                print(f"Number of invalid actions: {num_invalid_actions}\n")
                # update stats
                stats["time"].append(env.time) # time taken to finish maze
                stats["total penalty"].append(num_invalid_actions) # sum of penalties incurred
                stats["average_loss"].append(np.mean(episode_loss))
                stats["std_loss"].append(np.std(episode_loss))
                stats["reward"].append(episode_reward)
                # update target network
                agent.update_target()
                agent.save(checkpoint_dir, loss, episode, stats)

        if episode % evaluate_interval == 0:
            success = evaluate_loop(env, agent, RewardClass, state_memory)
            total_success += success
            logging.debug(f"Number of times agent completed maze: {total_success}\n")
            if total_success >= 3:
                agent.save(checkpoint_dir, loss, episode, stats)
                stats["path"] = env.path
                return stats # maze solved
    return stats





if __name__ == "__main__":
    checkpoint_dir = "./checkpoints"
    #agent=DQNAgent((3,3,2),5, config['DQN']['gamma'], config['DQN']['learning_rate'], no_conv=False)
    agent=DQNAgent(27,5, config['DQN']['gamma'], config['DQN']['learning_rate'], no_conv=True)
    #agent=RandomAgent(TestEnv(time_limit=10000).actions)
    wandb.watch(agent.q_fn, idx=1)
    wandb.watch(agent.target_q_fn, idx=2)
    # Reward of Manhattan distance from goal state
    #RewardClass = ManhattanReward(goal_pos=(199,199))
    #RewardClass = BasicReward()
    RewardClass = CheeseReward()

    if args.load_path:
        episode, stats = agent.load(args.load_path)
    else:
        #begin training at episode 0
        episode=0
        stats=defaultdict(list)

    stats = train_loop(env=TestEnv(time_limit=config['Env']['time_limit']),
                agent=agent,
                replay_buffer=ReplayBuffer(config['DQN']['buffer_size']), 
                episodes=config['DQN']['episodes'],
                batch_size=config['DQN']['batch_size'],
                log_interval=config['DQN']['log_interval'],
                save_interval=config['DQN']['save_interval'],
                epsilon_decay_schedule=epsilon_schedule,
                RewardClass=RewardClass,
                checkpoint_dir=checkpoint_dir,
                start_episode=0,
                evaluate_interval=config['Env']['evaluate_interval'],
                state_memory=True)
    # save stats
    with open('./logs/stats.json', "w") as stats_file:
        json.dump(stats, stats_file)
