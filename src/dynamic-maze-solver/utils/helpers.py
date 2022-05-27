import numpy as np
"""
Helper functions.
"""

def add_state_memory(state, agent_pos, agent_path, landmarks=None):
    """
    Add binary indicator to each location in observation
    indicating if the agent has been to that location before.
    """
    # get array of indicators
    history = np.expand_dims(get_agent_history(agent_pos, agent_path, landmarks), -1)
    new_state = np.concatenate((state, history), axis=-1)
    return new_state

def get_agent_history(agent_pos, agent_path, landmarks=None):
    x,y = agent_pos
    # want to check if pos close to x,y 
    # exist in list of tuples
    agent_history = np.zeros((3,3))
    in_path = lambda x, y: int(agent_path.count((x,y)) > 0)
    is_landmark = lambda x,y: int(landmarks.count((x,y)) > 0)
    agent_history[0, 0] = in_path(x-1, y+1)
    agent_history[0, 1] = in_path(x, y+1)
    agent_history[0, 2] = in_path(x+1, y+1)
    agent_history[1, 0] = in_path(x-1, y)
    agent_history[1, 1] = 0 # current location so obvs in path
    agent_history[1, 2] = in_path(x+1, y)
    agent_history[2, 0] = in_path(x-1, y-1)
    agent_history[2,1] = in_path(x, y-1)
    agent_history[2,2] = in_path(x+1, y+1)
    if landmarks:
        agent_history[0, 0] = 0.5 if is_landmark(x-1, y+1) & (agent_history[0, 0] != 1) else 1
        agent_history[0, 1] = 0.5 if is_landmark(x, y+1) & (agent_history[0, 1] != 1) else 1
        agent_history[0, 2] = 0.5 if is_landmark(x+1, y+1) & (agent_history[0, 2] != 1) else 1
        agent_history[1, 0] = 0.5 if is_landmark(x-1, y) & (agent_history[1, 0] != 1) else 1
        agent_history[1, 1] = 0 # current location so obvs in path
        agent_history[1, 2] = 0.5 if is_landmark(x+1, y) & (agent_history[1, 2] != 1) else 1
        agent_history[2, 0] = 0.5 if is_landmark(x-1, y-1) & (agent_history[2, 0] != 1) else 1
        agent_history[2,1] = 0.5 if is_landmark(x, y-1) & (agent_history[2,1] != 1) else 1
        agent_history[2,2] = 0.5 if is_landmark(x+1, y+1) & (agent_history[2,2] != 1) else 1
    return agent_history

def add_agent_position(state, agent_pos):
    location = np.zeros((3,3, 1))
    location[0,0, 0] = agent_pos[0] / 199
    location[0,1, 0] = agent_pos[1] / 199
    new_state = np.concatenate((state, location), axis=-1)
    return new_state

def normalise_state(state):
    """
    Normalises state input by max possible values for fire channels
    """
    norm_state = state
    norm_state[:, :, 1] = norm_state[:, :, 1] / 2 # max number of seconds fire can appear for
    return norm_state


def populate_landmarks(landmark_step, goal, maze):
    shortest_path = BFS(maze, (1,1), goal)
    # add a landmark every landmark_step
    landmarks = shortest_path if landmark_step == 'all' else [
        shortest_path[i] \
        for i in range(
            0, 
            len(shortest_path), 
            landmark_step)]
    return landmarks


def BFS(maze, start, end):
    '''Brute-Force Search, adapted from https://github.com/s-pangburn/python-bfs/blob/master/source.py
    Args:
        maze(list): the maze to be navigated
        start(tuple): the starting coordinates (row, col)
        end(tuple): the end coordinates (row, col)
    Returns:
        shortest path from start to end
    '''
    queue = [start]
    visited = set()

    while len(queue) != 0:
        if queue[0] == start:
            path = [queue.pop(0)]  # Required due to a quirk with tuples in Python
        else:
            path = queue.pop(0)
        front = path[-1]
        if front == end:
            return path
        elif front not in visited:
            for adjacentSpace in getAdjacentSpaces(maze, front, visited):
                newPath = list(path)
                newPath.append(adjacentSpace)
                queue.append(newPath)
            visited.add(front)
    return None

def getAdjacentSpaces(maze, space, visited):
    ''' Returns all legal spaces surrounding the current space
    :param space: tuple containing coordinates (row, col)
    :return: all legal spaces
    '''
    spaces = list()
    spaces.append((space[0]-1, space[1]))  # Up
    spaces.append((space[0]+1, space[1]))  # Down
    spaces.append((space[0], space[1]-1))  # Left
    spaces.append((space[0], space[1]+1))  # Right

    final = list()
    for i in spaces:
        if maze[i[0]][i[1]] != 0 and i not in visited:
            final.append(i)
    return final