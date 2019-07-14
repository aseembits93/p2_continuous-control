import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment
import ipdb as pdb
#%matplotlib inline

from ddpg_agent import Agent
env = UnityEnvironment(file_name='Reacher_Linux_v2/Reacher.x86_64')
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=False)[brain_name]
n_agents = len(env_info.agents)
#seed = random.randint(1,100)

seed = 89

print("Seed : {}".format(seed))
agent = Agent(state_size=33, action_size=4, random_seed=seed)
agent.actor_local.load_state_dict(torch.load('checkpoint_actor.pth'))
agent.critic_local.load_state_dict(torch.load('checkpoint_critic.pth'))


for t in range(200):
    states = env_info.vector_observations
    actions = agent.act(states, add_noise=False)
    env_info = env.step(actions)[brain_name]
    next_states, rewards, dones = env_info.vector_observations, env_info.rewards, env_info.local_done
    if np.any(dones):
        break 


