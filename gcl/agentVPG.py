import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import conv2d_size_out


class AgentVPG(nn.Module):
    def __init__(self, state_shape, n_actions, mode = 'toy'):
        
        super().__init__()
        self.state_shape = state_shape
        self.n_actions = n_actions
        if mode == 'toy':
            self.model = nn.Sequential(
              nn.Linear(in_features = state_shape[0], out_features = 128),
              nn.ReLU(),
              nn.Linear(in_features = 128 , out_features = 64),
              nn.ReLU(),
              nn.Linear(in_features = 64 , out_features = 2)
            )
        elif mode == 'atari':
            conv_output = conv2d_size_out(
                conv2d_size_out(
                    conv2d_size_out(
                        conv2d_size_out(64, 3, 2), 
                        3, 2), 
                    3, 2), 
                1, 1
            )         
            self.model = nn.Sequential(
              nn.Conv2d(in_channels=state_shape[0], out_channels=16, kernel_size=3, stride=2),
              nn.ReLU(),
              nn.Conv2d(16, 32, 3, stride=2),
              nn.ReLU(),
              nn.Conv2d(32, 64, 3, stride=2),
              nn.ReLU(),nn.Linear(conv_output , 256),
              nn.Linear(256 , n_actions),
                
            )
    def forward(self, x):
        logits = self.model(x)
        return logits
    
    def predict_probs(self, states):
        states = torch.FloatTensor(states)
        logits = self.model(states).detach()
        probs = F.softmax(logits, dim = -1).numpy()
        return probs
    
    def generate_session(self, env, t_max=1000):
        states, actions, rewards = [], [], []
        s = env.reset()

        for t in range(t_max):
            action_probs = self.predict_probs(np.array([s]))[0]
            a = np.random.choice(self.n_actions,  p = action_probs)
            new_s, r, done, info = env.step(a)

            states.append(s)
            actions.append(a)
            rewards.append(r)

            s = new_s
            if done:
                break

        return states, actions, rewards