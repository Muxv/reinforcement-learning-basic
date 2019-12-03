import numpy as np
import gym
from pampy import match, _
from gym import spaces
from gym.utils import seeding



class Recyclingbot(gym.Env):
    def __init__(self):
        super().__init__()
        self.params_dict = {
            'alpha': 0.3,
            'beta': 0.7,
            'r_wait': 0,
            'r_search': 2,
            'r_forcedcharge': -3,
            'r_charge': 0
        }
        self.reward_range = [
            self.params_dict['r_wait'],
            self.params_dict['r_search'],
            self.params_dict['r_forcedcharge'],
            self.params_dict['r_charge']
        ]
        self.action_space = spaces.Discrete(3)
        self.action_name = ['search', 'wait', 'charge']
        self.observation_space = spaces.Discrete(2)
        self.observation_name = ['high', 'low']
        self.reset()


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self._state = 0

    def step(self, action):
        """
        Args:
            action (object): an action provided by the agent
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        alpha = self.params_dict['alpha']
        beta = self.params_dict['beta']
        if self._state == 0:  # high
            _state_change = np.random.choice([True, False], p=[alpha, 1-alpha])
        else:  # low
            _state_change = np.random.choice([True, False], p=[beta, 1-beta])

        if action in self.action_name:
            action = self.action_name.index(action)

        # avoid the illegal action input
        assert self.action_space.contains(action)

        next_state, reward = match(
            (self._state, action),

            # wait
            (_, 1), (self._state,
                     self.params_dict['r_wait']),
            # search when high
            (0, 0), (1-self._state if _state_change else self._state,
                     self.params_dict['r_search'],),
            # charge when high
            (0, 2), (self._state,
                     0),
            # search when low
            (1, 0), (1-self._state if _state_change else self._state,
                     self.params_dict['r_forcedcharge'] if _state_change else self.params_dict['r_search']),
            # charge when low
            (1, 2), (1-self._state,
                     0),
        )
        done = True
        print("After action [{}], state from [{}] change to [{}], get reward [{}]".format(
            self.action_name[action],
            self.observation_name[self._state],
            self.observation_name[next_state],
            str(reward)
        ))
        self._state = next_state

        return (next_state, reward, done, None)

    def environment_transfer(self, action, state_old, state_new, describe=False):
        """
        Args:
            action (object): an action provided by the agent
            state (object): an observation of the enviroment
        Returns:
            transfer percentage (float): the percentage of transfer from state_old to state_new by certain action  
        Usage:
            Use for DP to calculate the bellman function
        """
        if action in self.action_name:
            action = self.action_name.index(action)
        if state_old in self.observation_name:
            state_old = self.observation_name.index(state_old)
        if state_new in self.observation_name:
            state_new = self.observation_name.index(state_new)
        
        assert self.action_space.contains(action)
        assert self.observation_space.contains(state_old)
        assert self.observation_space.contains(state_new)
        

        percentage = match((action, state_old, state_new),
                           # high -> high by search
                           (0, 0, 0), self.params_dict['alpha'],
                           # high -> low by search
                           (0, 0, 1), 1 - self.params_dict['alpha'],
                           # high -> high by wait
                           (1, 0, 0), 1.0,
                           # high -> high by charge
                           (2, 0, 0), 1.0,
                           # low -> low by search
                           (0, 1, 1), self.params_dict['beta'],
                           # low -> high by search
                           (0, 1, 0), 1 - self.params_dict['beta'],
                           # low -> low by wait
                           (1, 1, 1), 1.0,
                           # low -> high by charge
                           (2, 1, 0), 1.0,
                           (_, _, _), 0.0
                           )
        if describe:
            info = [self.observation_name[state_old],
                    self.observation_name[state_new],
                    self.action_name[action],
                    percentage]
            print(
                "The percentage of state from [{}] to [{}] by action [{}] is [{}]".format(*info))
        return percentage

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, s):
        self._state = s


if __name__ == '__main__':
    bot = Recyclingbot()
    bot.step('wait')
    bot.step('wait')
    bot.environment_transfer(
        bot.action_space.sample(),
        bot.observation_space.sample(),
        bot.observation_space.sample(),
        describe=True
    )
    

