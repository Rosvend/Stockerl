import gymnasium as gym
from gymnasium import spaces
import numpy as np 


class CustomEnv(gym.Env):
    def __init__(self, df, initial_balance=1000000):
        super(CustomEnv,self).__init__()
        self.df = df 
        self.initial_balance = initial_balance

        #actions (1=buy all, 0=hold, -1=sell all)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        #observations (10 prices + invested value + cash)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(12,), dtype=np.float32)