# Gym stuff
import gym
from gym import Env, spaces
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete

# Helpers
import numpy as np
import random
import os

# Stable Baselines
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy


class StockEnv(Env):

    def __init__(self, df, window_size, frame_bound):

        self.df = df
        self.window_size = window_size
        self.frame_bound = frame_bound
        self.prices, self.signal_features = self._process_data()
        self.shape = (window_size, self.signal_features.shape[1])
        self._start_tick = self._current_tick = self.window_size

        # spaces
        self.action_space = Discrete(3) # Actions we can take: Buy, Sell, Hold
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float64)

        # Set starting capital
        self.cash = 5000
        self.shares = 0
        self.portfolio_value = 0
        self._calculate_total_value(self._start_tick)

        # length of time i.e. how many days of trading
        self.total_days = len(self.prices) - 1 - self.window_size

        # episode
        self._last_trade_tick = self._current_tick - 1
        self._end_tick = len(self.prices) - 1
        self._portfolio_value_last_tick = 0
        self._done = None
        self._total_reward = 0
        self._total_profit = 0  # unit

    def _calculate_total_value(self, tick):
        self._calculate_portfolio_value(tick)
        self.total_value = self.cash + self.portfolio_value
        return self.total_value

    def _calculate_portfolio_value(self, tick):
        current_price = self.prices[tick]
        self.portfolio_value = current_price * self.shares
        return self.portfolio_value

    def step(self, action):
        self._done = False
        self._current_tick += 1

        if self._current_tick == self._end_tick:
            self._done = True

        reward = 0
        current_price = self.prices[self._current_tick]
        last_trade_price = self.prices[self._last_trade_tick]
        price_diff = current_price - last_trade_price
        last_value = self._calculate_total_value(self._current_tick)

        # Apply action
        # 0 = Buy
        # 1 = Sell
        # 2 = Hold
        if action == 0:
            # Buy the stock
            if self.cash >= current_price:
                self._last_trade_tick = self._current_tick
                self.shares += 1
                self.cash -= current_price
        if action == 1:
            # Sell the stock
            if self.shares > 0:
                self._last_trade_tick = self._current_tick
                self.shares -= 1
                self.cash += current_price
                reward = price_diff
        if action == 2:
            # Hold
            pass

        self._calculate_total_value(self._current_tick)
        reward += self.total_value - last_value

        self._total_reward += reward
        self._calculate_portfolio_value(self._current_tick)
        self._calculate_total_value(self._current_tick)

        obs = self._get_observation()

        info = dict(
            total_reward=self._total_reward,
            total_value=self.total_value,
            cash=self.cash,
            portfolio_value=self.portfolio_value,
            shares=self.shares
        )

        # Return step information
        return obs, reward, self._done, info

    def render(self):
        # Implement viz
        pass

    def reset(self):
        self.__init__(self.df, self.window_size, self.frame_bound)
        self._current_tick = self._start_tick
        return self._get_observation()

    def _get_observation(self):
        return self.signal_features[(self._current_tick - self.window_size + 1):self._current_tick + 1]

    def _process_data(self):
        start = self.frame_bound[0] - self.window_size
        end = self.frame_bound[1]
        prices = self.df.loc[:, 'Open'].to_numpy()[start:end]
        signal_features = self.df.loc[:, ['Open', 'Volume', 'SMA', 'RSI', 'OBV']].to_numpy()[start:end]
        return prices, signal_features

    # def render(self, mode='human'):
    #
    #     def _plot_position(position, tick):
    #         color = None
    #         if position == Positions.Short:
    #             color = 'red'
    #         elif position == Positions.Long:
    #             color = 'green'
    #         if color:
    #             plt.scatter(tick, self.prices[tick], color=color)
    #
    #     if self._first_rendering:
    #         self._first_rendering = False
    #         plt.cla()
    #         plt.plot(self.prices)
    #         start_position = self._position_history[self._start_tick]
    #         _plot_position(start_position, self._start_tick)
    #
    #     _plot_position(self._position, self._current_tick)
    #
    #     plt.suptitle(
    #         "Total Reward: %.6f" % self._total_reward + ' ~ ' +
    #         "Total Profit: %.6f" % self._total_profit
    #     )
    #
    #     plt.pause(0.01)
    #
    # def render_all(self, mode='human'):
    #     window_ticks = np.arange(len(self._position_history))
    #     plt.plot(self.prices)
    #
    #     short_ticks = []
    #     long_ticks = []
    #     for i, tick in enumerate(window_ticks):
    #         if self._position_history[i] == Positions.Short:
    #             short_ticks.append(tick)
    #         elif self._position_history[i] == Positions.Long:
    #             long_ticks.append(tick)
    #
    #     plt.plot(short_ticks, self.prices[short_ticks], 'ro')
    #     plt.plot(long_ticks, self.prices[long_ticks], 'go')
    #
    #     plt.suptitle(
    #         "Total Reward: %.6f" % self._total_reward + ' ~ ' +
    #         "Total Profit: %.6f" % self._total_profit
    #     )