from typing import Optional, Union, List, Tuple

import gym
from gym import spaces
from gym.core import RenderFrame, ActType, ObsType


class StockEnvTrain(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df, day=0):
        self.day = day
        self.df = df
        # Action Space
        # action_space normalization and shape is STOCK_DIM
        self.action_space = spaces.Box(low=-1, high=1, shape=(STOCK_DIM,))

        # State Space
        # Shape = 181: [Current Balance]+[prices 1–30]+[owned shares 1–30]
        # +[macd 1–30]+ [rsi 1–30] + [cci 1–30] + [adx 1–30]
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(181,))
        # load data from a pandas dataframe
        self.data = self.df.loc[self.day, :]
        self.terminal = False
        # initalize state
        self.state = [INITIAL_ACCOUNT_BALANCE] + \
                     self.data.adjcp.values.tolist() + \
                     [0] * STOCK_DIM + \
                     self.data.macd.values.tolist() + \
                     self.data.rsi.values.tolist() + \
                     self.data.cci.values.tolist() + \
                     self.data.adx.values.tolist()
        # initialize reward
        self.reward = 0
        self.cost = 0
        # memorize all the total balance change
        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        self.rewards_memory = []
        self.trades = 0
        # self.reset()
        self._seed()

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        pass

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        pass