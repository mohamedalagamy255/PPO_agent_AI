#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# environment
import gym
import numpy as np
from util import gray_scale, resize, resize_gray
import PIL
from PIL import Image
from config import *
from atari_wrappers import *


class Env():
    def __init__(self, agent, env_id, seed):
        env = gym.make(env_name)
        # env =  NoopResetEnv(env , noop_max=env_id)
        env = WarpFrame(env, width=IMG_W, height=IMG_H, grayscale=True)
        env = FrameStack(env, k=k)  # return (IMG_H , IMG_W ,k)
        self.env = env

        self.agent = agent
        self.env_id = env_id
        self.seed = seed

    def run(self):
        np.random.seed(self.seed)
        self.env.seed(self.seed)
        count              = 1                # use to count episode
        state              = self.env.reset()
        one_episode_reward = 0 
        step               = 0                # use to count step in one epoch
        done               = True
        while True:
            step                 += 1
            a                     = self.agent.choice_action(state, done)
            state_, r, done, info = self.env.step(a)
            one_episode_reward   += r

            if done:
                state_ = self.env.reset()

            # This can limit max step
            # if step >= 60000:
            #    done = True

            self.agent.observe(state, a, r, state_, done)

            state      = state_

            if done:
                print(str(self.env_id) + ":" + str(count) + "      :       " + str(one_episode_reward))
                count             += 1
                one_episode_reward = 0
                state              = state_
                step               = 0

