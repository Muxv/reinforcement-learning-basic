import numpy as np
import gym
from pampy import match, _
from gym import spaces


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
        self.state = 0  # 0    ,  1

    def seed(self, seed=None):
        if seed == None:
            np.random.seed(10)
        else:
            np.random.seed(seed)
        return

    def reset(self):
        self.state = 0

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
        if self.state == 0:  # high
            state_change = np.random.choice([True, False], p=[alpha, 1-alpha])
        else:  # low
            state_change = np.random.choice([True, False], p=[beta, 1-beta])

        if action in self.action_name:
            action = self.action_name.index(action)

        next_state, reward = match(
            (self.state, action),

            # wait
            (_, 1), (self.state,
                     self.params_dict['r_wait']),
            # search when high
            (0, 0), (1-self.state if state_change else self.state,
                     self.params_dict['r_search'],),
            # charge when high
            (0, 2), (self.state,
                     0),
            # search when low
            (1, 0), (1-self.state if state_change else self.state,
                     self.params_dict['r_forcedcharge'] if state_change else self.params_dict['r_search']),
            # charge when low
            (1, 2), (1-self.state,
                     0),
        )
        done = True
        print("After action [{}], state from [{}] change to [{}], get reward [{}]".format(
            self.action_name[action],
            self.observation_name[self.state],
            self.observation_name[next_state],
            str(reward)
        ))
        self.set_state(next_state)

        return (next_state, reward, done, None)

    def set_state(self, state):
        self.state = state

if __name__ == '__main__':
    bot = Recyclingbot()
    bot.step('wait')
    bot.step('wait')
    bot.step('search')
    bot.step('search')
    bot.step('search')
    bot.step('search')
    bot.step('search')
