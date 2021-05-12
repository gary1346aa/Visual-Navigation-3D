import torch
import time
import random
import cv2
import numpy as np
from maze3d.maze_env import MazeBaseEnv, make


model = torch.load(r'../data/ppo/ppo_s0/pyt_save/model.pt')
env = MazeBaseEnv(make("MazeBoardRandom"), render_res = (64, 64))

epi_return, epi_count = 0, 0
o, _ = env.reset()

while epi_count < 10:
    o = o.astype(np.float32) / 255.
    o = o.transpose(2,0,1)
    state = torch.as_tensor(o[np.newaxis,...], dtype=torch.float32)
    a, v, logp = model.step(state)
    env.render()
    o, r, done, _ = env.step(a) # take a random action
    epi_return += r
    cv2.waitKey(10)
    if done:
        print(f'Done episode #{epi_count+1}, episodic return = {epi_return:.3f}')
        o = env.reset()
        epi_return = 0
        epi_count += 1
        continue

env.close()