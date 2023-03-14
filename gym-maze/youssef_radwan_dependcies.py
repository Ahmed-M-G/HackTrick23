import pygame
import random
import numpy as np
import os
import time
import sys
import numpy as np
import copy

import gym
from gym import error, spaces, utils
from gym.utils import seeding

from gym_maze.envs.maze_view_2d import MazeView2D, RescueItem

class MazeEnv(gym.Env):
    metadata = {
        "render.modes": ["human", "rgb_array"],
    }

    ACTION = ["N", "S", "E", "W"]

    def __init__(self, maze_file=None, maze_cells=None, maze_size=None, mode=None, enable_render=True, rescue_item_locations=None, has_loops=False):

        self.viewer = None
        self.enable_render = enable_render
        
        self.rescue_item_locations= rescue_item_locations

        if hasattr(maze_cells, 'shape'):
            self.maze_view = MazeView2D(maze_name="OpenAI Gym - Maze (10 x 10)",
                                        maze_cells=maze_cells,
                                        screen_size=(640, 640), 
                                        enable_render=enable_render, rescue_item_locations=self.rescue_item_locations)
        
        elif maze_file:
            self.maze_view = MazeView2D(maze_name="OpenAI Gym - Maze (%s)" % maze_file,
                                        maze_file_path=maze_file,
                                        screen_size=(640, 640), 
                                        enable_render=enable_render, rescue_item_locations=self.rescue_item_locations)
        elif maze_size:
            self.maze_view = MazeView2D(maze_name="OpenAI Gym - Maze (%d x %d)" % maze_size,
                                        maze_size=maze_size, screen_size=(640, 640),
                                        has_loops=has_loops,
                                        enable_render=enable_render, rescue_item_locations=self.rescue_item_locations)
        else:
            raise AttributeError("One must supply either a maze_file path (str) or the maze_size (tuple of length 2)")

        self.maze_size = self.maze_view.maze_size

        # forward or backward in each dimension
        self.action_space = spaces.Discrete(2*len(self.maze_size))

        # observation is the x, y coordinate of the grid
        low = np.zeros(len(self.maze_size), dtype=int)
        high =  np.array(self.maze_size, dtype=int) - np.ones(len(self.maze_size), dtype=int)
        self.observation_space = spaces.Box(low, high, dtype=np.int64)

        # initial condition
        self.state = None
        self.steps_beyond_done = None
        self.steps = None

        # Simulation related variables.
        self.seed()
        self.reset()

        # Just need to initialize the relevant attributes
        self.configure()
        # self.maze_view.maze.save_maze('./hacktrick.npy')

    def __del__(self):
        if self.enable_render is True:
            self.maze_view.quit_game()

    def configure(self, display=None):
        self.display = display

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    # <<<<<<<<<<<<
    def reward(self):
        if np.array_equal(self.maze_view.robot, [9,9]):  # find end point
            reward = 100
        elif tuple(self.maze_view.robot) in self.valid_items: # find item
            reward = 50
        elif np.array_equal(self.maze_view.robot, self.prev_position): # wall
            reward = -20
        else:
            # print(self.maze_view.get_rescue_items_locations())
            # get the negative manhaten distance 
            reward = np.sum(-1*np.array(self.maze_view.get_rescue_items_locations()[0])) + -2 * np.linalg.norm(self.maze_view.robot-[9,9])
        return reward

# 1- penality redandant movments 
# 2- penality visite the same cell ketttter
# 3- reward get an item  XXX
# 4- penality in case doing a wrong action (like go throw wall) XXX

    def step(self, action):
        info = {}
        self.prev_action = action
        self.prev_position = copy.deepcopy(self.maze_view.robot)
        self.maze_view.move_robot(self.ACTION[action])
        info['rescued_items'] = self.maze_view.rescued_items
        self.state = self.get_current_flatten_state()
        
        truncated = False
        info['riddle_type'] = 0
        info['riddle_question'] = 0

        reward = self.reward()

        if tuple(self.maze_view.robot) in self.valid_items:
            del self.valid_items[tuple(self.maze_view.robot)]
            # print(">>>>>>  valid items: ", self.valid_items)

        self.steps +=1
        terminated = False
        if np.array_equal(self.maze_view.robot, [9,9]) or self.steps == 5000:
            terminated = True

        return self.state, reward, terminated, info

    def get_current_state(self):
        info = {}
        info['rescued_items'] = self.maze_view.rescued_items
        self.state = self.get_current_flatten_state()
        truncated = False
        info['riddle_type'] = None
        info['riddle_question'] = None
        reward = self.reward()
        terminated = False
        return self.state, reward, terminated, truncated, info


    def get_current_flatten_state(self):
        state = []
        state.extend(self.maze_view.robot)
        state.extend(self.maze_view.get_rescue_items_locations()[0])
        for direction in self.maze_view.get_rescue_items_locations()[1]:
            state.extend(direction)
        return np.array(state).squeeze()
    
    def reset(self):
        self.maze_view.reset_robot()
        self.state = self.get_current_flatten_state()
        self.valid_items = copy.deepcopy(self.rescue_item_locations)
        self.prev_position = self.maze_view.robot
        self.prev_action = None
        self.steps_beyond_done = None
        self.steps = 0
        self.terminated = False
        self.truncated = False
        self.maze_view.reset_rescue_items()
        return self.state

    def is_game_over(self):
        return self.maze_view.game_over

    def render(self, mode="human", close=False):
        if close:
            self.maze_view.quit_game()
        return self.maze_view.update(mode)


class MazeEnvSample5x5(MazeEnv):

    def __init__(self, enable_render=True, maze_cells=None, rescue_item_locations=None):
        super(MazeEnvSample5x5, self).__init__(maze_file="maze2d_5x5.npy", enable_render=enable_render, maze_cells=maze_cells, rescue_item_locations= rescue_item_locations)


class MazeEnvRandom5x5(MazeEnv):

    def __init__(self, enable_render=True, maze_cells=None, rescue_item_locations=None):
        super(MazeEnvRandom5x5, self).__init__(maze_size=(5, 5), enable_render=enable_render, maze_cells=maze_cells, mode='plus', rescue_item_locations= rescue_item_locations)


class MazeEnvSample10x10(MazeEnv):

    def __init__(self, enable_render=True, maze_cells=None, rescue_item_locations=None, maze_file=None):
        super(MazeEnvSample10x10, self).__init__(maze_file=maze_file, enable_render=enable_render, maze_cells=maze_cells, rescue_item_locations= rescue_item_locations)


class MazeEnvRandom10x10(MazeEnv):

    def __init__(self, enable_render=True, maze_cells=None, rescue_item_locations=None):
        super(MazeEnvRandom10x10, self).__init__(maze_size=(10, 10), enable_render=enable_render, maze_cells=maze_cells, mode='plus', rescue_item_locations=rescue_item_locations)


class MazeEnvSample3x3(MazeEnv):

    def __init__(self, enable_render=True, maze_cells=None, rescue_item_locations=None):
        super(MazeEnvSample3x3, self).__init__(maze_file="maze2d_3x3.npy", enable_render=enable_render, maze_cells=maze_cells, rescue_item_locations= rescue_item_locations)


class MazeEnvRandom3x3(MazeEnv):

    def __init__(self, enable_render=True, maze_cells=None, rescue_item_locations=None):
        super(MazeEnvRandom3x3, self).__init__(maze_size=(3, 3), enable_render=enable_render, maze_cells=maze_cells, rescue_item_locations= rescue_item_locations)


class MazeEnvSample100x100(MazeEnv):

    def __init__(self, enable_render=True, maze_cells=None, rescue_item_locations=None):
        super(MazeEnvSample100x100, self).__init__(maze_file="maze2d_100x100.npy", enable_render=enable_render, maze_cells=maze_cells, rescue_item_locations= rescue_item_locations)


class MazeEnvRandom100x100(MazeEnv):

    def __init__(self, enable_render=True, maze_cells=None, rescue_item_locations=None):
        super(MazeEnvRandom100x100, self).__init__(maze_size=(100, 100), enable_render=enable_render, maze_cells=maze_cells, rescue_item_locations= rescue_item_locations)


class MazeEnvRandom10x10Plus(MazeEnv):

    def __init__(self, enable_render=True, maze_cells=None, rescue_item_locations=None):
        super(MazeEnvRandom10x10Plus, self).__init__(maze_size=(10, 10), mode="plus", enable_render=enable_render, maze_cells=maze_cells, rescue_item_locations= rescue_item_locations)


class MazeEnvRandom20x20Plus(MazeEnv):

    def __init__(self, enable_render=True, maze_cells=None, rescue_item_locations=None):
        super(MazeEnvRandom20x20Plus, self).__init__(maze_size=(20, 20), mode="plus", enable_render=enable_render, maze_cells=maze_cells, rescue_item_locations= rescue_item_locations)


class MazeEnvRandom30x30Plus(MazeEnv):
    def __init__(self, enable_render=True, maze_cells=None, rescue_item_locations=None):
        super(MazeEnvRandom30x30Plus, self).__init__(maze_size=(30, 30), mode="plus", enable_render=enable_render, maze_cells=maze_cells, rescue_item_locations= rescue_item_locations)