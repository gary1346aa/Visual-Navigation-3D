import gym
from gym_minigrid.wrappers import *
import torch
import time
import random

model = torch.load(r'../data/MiniGrid-MultiRoom-N4-S5-v0-RNDPPO/MiniGrid-MultiRoom-N4-S5-v0-RNDPPO_s0/pyt_save/model.pt')
env = gym.make('MiniGrid-MultiRoom-N4-S5-v0')
env.seed(random.randint(0, 100))
action_dict = {i: a.name for i, a in enumerate(env.actions)}

epi_return, epi_count = 0, 0
o = env.reset()['image'].flatten()

while epi_count < 10:
    a, v, logp = model.step(torch.as_tensor(o, dtype=torch.float32))
    env.render()
    o, r, done, _ = env.step(a) # take a random action
    o = o['image'].flatten()
    epi_return += r

    #print(action_dict[a.item(0)])
    if done:
        print(f'Done episode #{epi_count+1}, episodic return = {epi_return:.3f}')
        o = env.reset()['image'].flatten()
        epi_return = 0
        epi_count += 1
        continue

env.close()