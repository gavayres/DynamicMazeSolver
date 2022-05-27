import logging
import torch
import wandb
import numpy as np
import yaml
import json
import argparse
from collections import defaultdict
from read_maze import load_maze, get_local_maze_information
from utils.env import Env
from utils.reward import CheeseReward
from agents.dqn import DQNAgent
from utils.buffers import ReplayBuffer
from utils.schedules import get_epsilon_decay_schedule
from utils.helpers import add_agent_position, add_state_memory

# set up argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--load_path', type=str, default=None, help='Load an agent from the given checkpoint path.')
parser.add_argument('--evaluate', type=bool, default=False, help='Run agent in evaluation mode.')
args = parser.parse_args()

#wandb.init(project="dynamic-maze-solver", entity="gavayres")
# load config file
with open("agents/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# set random seed
np.random.RandomState(config['DQN']['random_seed'])

# setup logger 
logging.basicConfig(filename='logs/run_logs.log', 
                    format='%(asctime)s %(message)s', 
                    datefmt='%m/%d/%Y %I:%M:%S %p', 
                    level=logging.DEBUG)

# load the maze into our environment
load_maze()

def construct_maze():
    """
    Construct global maze from local observations.
    Note this is never given to the agent, it is only
    used by the environment to populate landmarks.
    """
    rows, cols = (201, 201)
    constr = np.zeros((rows, cols))
    for j in range(1, cols, 2):
        for i in range(1, rows, 2):
            constr[i-1:i+2,j-1:j+2] = get_local_maze_information(i,j)[:,:, 0]
    return constr

print("Building environment. This takes just under 3 minutes.\n")
global_maze = construct_maze()

# helper function to convert a numpy array to a tensor
tensorify = lambda np_array: torch.from_numpy(np_array)
# reshape [BATCH, ROWS, COLS, CHANNELS] -> [BATCH, CHANNELS, ROWS, COLS]
reshape = lambda tensor: torch.permute(tensor, (2, 0, 1))

# epsilon schedule
epsilon_schedule = get_epsilon_decay_schedule()

def evaluate_loop(env, agent, RewardClass, state_memory=False, compass=False, verbose=False):
    env.reset()
    state_t = env.state
    if state_memory:
        # add new dimension indicating if agent has been in location
        state_t = add_state_memory(state_t, (env.x, env.y), env.path)
    if compass:
        state_t = add_agent_position(state_t, (env.x, env.y))
    state_t = tensorify(state_t)
    state_t = reshape(state_t).double()
    done = False
    episode_reward=0
    agent.epsilon = 0
    # run on maze
    while not done:
        action_idx = agent.act(state_t.double())
        action = env.actions[action_idx]
        if verbose:
            print(f"Agent observation: {get_local_maze_information(env.x, env.y)}\n")
            print(f"Agent action: {action}\n")
        state_tp1, done, penalty = env.update(action)
        if state_memory:
            # add new dimension indicating if agent has been in loc
            state_tp1 = add_state_memory(state_tp1, (env.x, env.y), env.path, env.landmarks)
        if compass:
            state_tp1 = add_agent_position(state_tp1, (env.x, env.y))
        state_tp1 = tensorify(state_tp1)
        state_tp1 = reshape(state_tp1)
        reward = RewardClass.reward(env) + penalty
        episode_reward+=reward
        state_t = state_tp1.detach()
        if done==1:
            print(f"Time taken: {env.time}\n")
            print(f"Agent position: {env.x}, {env.y}\n")
            print(f"Total reward: {episode_reward}")
            if (env.timed_out == False) & (env.blocked == False):
                if verbose:
                    print(f"Agent path through maze: {env.path}\n")
                return 1 # agent completed the maze!
            else: 
                if verbose:
                    print(f"Agent path through maze: {env.path}\n")
                return 0
            
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
            state_memory=False,
            compass=False,
            evaluate=False
            ):
    total_success=0
    """ If in evaluation mode, just run evaluation loop."""
    if evaluate:
        env.time_limit = 4000
        success = evaluate_loop(env, agent, RewardClass, state_memory, compass, verbose=True)
        return success

    for episode in range(start_episode, episodes):
        episode_loss=[]
        loss=0
        num_invalid_actions=0
        episode_reward=0
        env.reset()
        state_t = env.state
        if state_memory:
            # add new dimension indicating if agent has been in loc
            state_t = add_state_memory(state_t, (env.x, env.y), env.path, env.landmarks)
        if compass:
            state_t = add_agent_position(state_t, (env.x, env.y))
        # convert to tensor
        state_t = tensorify(state_t)
        # reshape for net input and add batch_size dimension
        state_t = reshape(state_t).double()
        done = False
        agent.epsilon = epsilon_decay_schedule(episode)
        print(f"Epsilon: {agent.epsilon}\n")

        # inner loop, play game and record results
        while not done:
            action_idx = agent.act(state_t.double())
            action = env.actions[action_idx]
            state_tp1, done, penalty = env.update(action)
            if state_memory:
                # add new dimension indicating if agent has been in loc
                state_tp1 = add_state_memory(state_tp1, (env.x, env.y), env.path, env.landmarks)
            if compass:
                state_tp1 = add_agent_position(state_tp1, (env.x, env.y))
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
            success = evaluate_loop(env, agent, RewardClass, state_memory, compass)
            total_success += success
            logging.debug(f"Number of times agent completed maze: {total_success}\n")
            if total_success >= 1:
                agent.save(checkpoint_dir, loss, episode, stats)
                stats["path"] = env.path
                return stats # maze solved
    return stats





if __name__ == "__main__":
    checkpoint_dir = "./checkpoints"
    agent=DQNAgent(config['DQN']['gamma'], config['DQN']['learning_rate'], no_conv=True, fire=False)
    #wandb.watch(agent.q_fn, idx=1)
    #wandb.watch(agent.target_q_fn, idx=2)
    # Bonus reward for entering new states
    RewardClass = CheeseReward()
    evaluate = False
    if args.load_path:
        episode, stats = agent.load(args.load_path)
    else:
        #begin training at episode 0
        episode=0
        stats=defaultdict(list)

    if args.evaluate:
        evaluate=True

    stats = train_loop(env=Env(
                    time_limit=config['Env']['time_limit'], 
                    maze=global_maze, 
                    fire=False, 
                    goal=(199,199), 
                    landmark_step='all'),
                agent=agent,
                replay_buffer=ReplayBuffer(config['DQN']['buffer_size']), 
                episodes=config['DQN']['episodes'],
                batch_size=config['DQN']['batch_size'],
                log_interval=config['DQN']['log_interval'],
                save_interval=config['DQN']['save_interval'],
                epsilon_decay_schedule=epsilon_schedule,
                RewardClass=RewardClass,
                checkpoint_dir=checkpoint_dir,
                start_episode=17,
                evaluate_interval=config['Env']['evaluate_interval'],
                state_memory=True,
                compass=False,
                evaluate=evaluate)
    # save stats
    with open('./logs/stats.json', "w") as stats_file:
        json.dump(stats, stats_file)
