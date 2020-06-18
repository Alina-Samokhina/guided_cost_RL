import numpy as np
import torch
import torch.nn as nn 

from .utils import get_cumulative_rewards, to_one_hot


def train_vpg_on_session(model, env, states, actions, rewards, optimizer, gamma=0.99, entropy_coef=1e-2, scheduler = None):

    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.int32)
    cumulative_returns = np.array(get_cumulative_rewards(rewards, gamma))
    cumulative_returns = torch.tensor(cumulative_returns, dtype=torch.float32)

    logits = model(states)
    probs = nn.functional.softmax(logits, -1)
    log_probs = nn.functional.log_softmax(logits, -1)

    log_probs_for_actions = torch.sum(
        log_probs * to_one_hot(actions, env.action_space.n), dim=1)
   
    entropy = -torch.mean(torch.sum(probs*log_probs), dim = -1 )
    loss = -torch.mean(log_probs_for_actions*cumulative_returns -entropy*entropy_coef)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if scheduler:
        scheduler.step()

    return np.sum(rewards)

def train_gcl_on_session(
    model,
    env, 
    cost_f, 
    demo_traj,
    sampled_traj,
    optimizer_model, 
    optimizer_cost,
    scheduler_model = None, 
    scheduler_cost = None,
    gamma = 0.99, 
    entropy_coef = 1e-2
):

    states, actions, rewards = sampled_traj
    states_demo, actions_demo, rewards_demo = demo_traj
    
    rewards = torch.tensor(rewards, dtype=torch.float32)
    rewards_demo = torch.tensor(rewards_demo, dtype=torch.float32)
    
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.int32)
    states_demo = torch.tensor(states_demo, dtype=torch.float32)
    actions_demo = torch.tensor(actions_demo, dtype=torch.int32)
    
    costs = cost_f(states)
    costs_demo = cost_f(states_demo)
    
    logits = model(states)
    probs = nn.functional.softmax(logits, -1)
    logits_demo = model(states_demo)
    probs_demo = nn.functional.softmax(logits_demo, -1)
    
       
    loss_IOC = torch.mean(costs_demo) + \
        torch.log(torch.mean((torch.exp(-costs))/(probs))+torch.mean((torch.exp(-costs_demo))/(probs_demo)))    
    loss_IOC.backward()
    optimizer_cost.step()
    optimizer_cost.zero_grad()
        
        
    if scheduler_cost:
        scheduler_cost.step()
           
    costs = cost_f(states).detach().numpy() 
    cumulative_returns = np.array(get_cumulative_rewards(costs, gamma))
    cumulative_returns = torch.tensor(cumulative_returns, dtype=torch.float32)

    logits = model(states)
    probs = nn.functional.softmax(logits, -1)
    log_probs = nn.functional.log_softmax(logits, -1)

    log_probs_for_actions = torch.sum(
        log_probs * to_one_hot(actions, env.action_space.n), dim=1)
   
    entropy = -torch.mean(torch.sum(probs*log_probs), dim = -1 )
    loss = -torch.mean(log_probs_for_actions*cumulative_returns -entropy*entropy_coef) 

    loss.backward()
    optimizer_model.step()
    optimizer_model.zero_grad()
    if scheduler_model:
        scheduler_model.step()

    return np.sum(rewards.detach().numpy()), np.sum(costs)