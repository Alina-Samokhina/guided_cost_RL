import torch
import numpy as np
import gym

def get_cumulative_rewards(rewards, gamma=0.99):
    G = np.zeros_like(rewards, dtype = float)
    G[-1] = rewards[-1]
    for idx in range(-2, -len(rewards)-1, -1):
        G[idx] = rewards[idx] + gamma * G[idx+1]
    return G 

def to_one_hot(y_tensor, ndims):
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    y_one_hot = torch.zeros(
        y_tensor.size()[0], ndims).scatter_(1, y_tensor, 1)
    return y_one_hot

def conv2d_size_out(size, kernel_size, stride):
    return (size - (kernel_size - 1) - 1) // stride  + 1

def make_env_cartpole(name, seed=None):
    env = gym.make(name).unwrapped
    if seed is not None:
        env.seed(seed)
    return env