import gym
from gym import error, spaces, utils
from gym.utils import seeding
from tqdm import tqdm
from keras.layers import PReLU, BatchNormalization, Normalization
from keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from rl.agents.dqn import DQNAgent
from rl.agents.cem import CEMAgent
from rl.agents.sarsa import SARSAAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory, EpisodeParameterMemory
import tensorflow as tf
import sys
import numpy as np
import math
import random
import json
import requests
import gym_maze
from gym_maze.envs.maze_manager import MazeManager
from youssef_radwan_dependcies import MazeEnvRandom10x10, WIND_VISITED_MAT_SIZE
from riddle_solvers import *
import time
import traceback
import copy
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from rl.callbacks import ModelIntervalCheckpoint
import warnings
warnings.filterwarnings("ignore")

def randomize_rescue_items(maze_size):
        rescue_items_dict = {}
        riddle_types = ['server', 'cipher', 'pcap', 'captcha']
        random.shuffle(riddle_types)

        for riddle_type in riddle_types:
            position = (random.randrange(0, maze_size-1), random.randrange(0, maze_size-1))
            while(position==(0,0) or position==(9,9)):
                position = (random.randrange(0, maze_size-1), random.randrange(0, maze_size-1))
                
            rescue_items_dict[position] = riddle_type
        return rescue_items_dict


def build_model(actions):
    model = Sequential()   
    model.add(Dense(32, activation=LeakyReLU(alpha=0.24), input_shape=(1,DIMETIONS)))
    model.add(Flatten())
    # model.add(Normalization())
    model.add(Dense(32, activation=LeakyReLU(alpha=0.24)))
    # model.add(Normalization())
    model.add(Dense(actions, activation='linear'))
    return model


def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = EpisodeParameterMemory(limit=50000, window_length=1)
    dqn = SARSAAgent(model, actions, policy=policy, test_policy=None, gamma=0.99, nb_steps_warmup=10, train_interval=1)
    return dqn


if __name__ == "__main__":
    sample_maze = np.load("hackathon_sample.npy")
    agent_id = "9" # add your agent id here
    maze_size = 10
    render = True
    env = MazeEnvRandom10x10(maze_cells=sample_maze, rescue_item_locations=randomize_rescue_items(maze_size), enable_render=render)
    
    n_state, reward, terminated, info = env.step(1)
    DIMETIONS = len(n_state)
    print(">>>>>>  ", DIMETIONS)
    actions = env.action_space.n
    model = build_model(actions)
    Adam._name = 'IBM :)'
    dqn = build_agent(model, actions)
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=2e-3)
    dqn.compile(optimizer)

    model_checkpoint_callback = ModelIntervalCheckpoint(
        filepath='./models/sarsa_model/',
        interval=10000, verbose=1)

    
    history = dqn.fit(env, nb_steps=200000, visualize=render, verbose=1, log_interval=5000, callbacks=[model_checkpoint_callback])

    # print(history.history)
    # losses = history.history['episode_reward']
    # plt.plot(losses)
    scores = dqn.test(env, nb_episodes=1, visualize=True)
    # print(np.mean(scores.history['episode_reward']))
    dqn.save_weights('./models/saved_model/', overwrite=True)
    # dqn.load_weights('./models/saved_model/')
    model.save('./models/saved_model_model/')
