# Gym stuff
from enum import Enum

import gym
from finta import TA
from gym import Env, spaces
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete, flatten, flatten_space
from gym.utils import seeding
from keras import models

from leaves.databitch import DataBitch

# Helpers
import numpy as np
import random
import os

# Stable Baselines
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy


class Actions(Enum):
    Sell = 0
    Buy = 1


class Positions(Enum):
    Short = 0
    Long = 1


class StockEnv(Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, data, window_size=10, model=None):

        self.seed()
        self.model = model
        self.data = data
        self.df = data.df
        self.window_size = window_size
        self.predictions = []
        self.prices, self.signal_features = self._process_data()
        self.shape = (window_size, self.signal_features.shape[1])
        self._start_tick = self._current_tick = 0

        # spaces
        self.action_space = Discrete(3)  # Actions we can take: Buy, Sell, Hold
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float64)

        # Set starting capital
        self.cash = 5000
        self.shares = 0
        self.portfolio_value = 0
        self._calculate_total_value()

        # episode
        self._last_trade_tick = self._current_tick - 1
        self._end_tick = len(self.prices) - self.window_size
        self._portfolio_value_last_tick = 0
        self._done = None
        self._total_reward = 0
        self._total_profit = 0  # unit

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _calculate_total_value(self):
        self.total_value = self.cash + self._calculate_portfolio_value()
        return self.total_value

    def _calculate_portfolio_value(self):
        current_price = self.prices[self._current_tick]
        self.portfolio_value = current_price * self.shares
        return self.portfolio_value

    def step(self, action):
        self._done = False

        reward = 0
        current_price = self.prices[self._current_tick + self.window_size]
        last_trade_price = self.prices[self._last_trade_tick + self.window_size]
        price_diff = current_price - last_trade_price
        last_value = self._calculate_total_value()

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

        self._current_tick += 1
        self._calculate_total_value()
        reward += self.total_value - last_value
        # self._total_reward += reward

        obs = self._get_observation()

        info = dict(
            total_reward=self._total_reward,
            total_value=self.total_value,
            cash=self.cash,
            portfolio_value=self.portfolio_value,
            shares=self.shares
        )

        if self._current_tick == self._end_tick:
            self._done = True

        # Return step information
        return obs, reward, self._done, info

    def render(self):
        # Implement viz
        pass

    def reset(self):
        self.model = self.model
        self.data = self.data
        self.df = self.data.df
        self.window_size = self.window_size
        self.predictions = []
        self.prices, self.signal_features = self._process_data()
        self.shape = (self.window_size, self.signal_features.shape[1])
        self._start_tick = self._current_tick = 0

        # spaces
        self.action_space = Discrete(3)  # Actions we can take: Buy, Sell, Hold
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float64)

        # Set starting capital
        self.cash = 5000
        self.shares = 0
        self.portfolio_value = 0
        self._calculate_total_value()

        # episode
        self._last_trade_tick = self._current_tick - 1
        self._end_tick = len(self.prices) - self.window_size
        self._portfolio_value_last_tick = 0
        self._done = None
        self._total_reward = 0
        self._total_profit = 0  # unit
        return self._get_observation()

    def _get_observation(self):
        return self.signal_features[self._current_tick:self._current_tick + self.window_size]

    def _calculate_values(self):
        # Calculate SMA, RSI, and OBV
        self.df.loc[:, 'SMA'] = TA.SMA(self.df, 12)
        self.df.loc[:, 'RSI'] = TA.RSI(self.df)
        self.df.loc[:, 'OBV'] = TA.OBV(self.df)
        self.df.fillna(0, inplace=True)

    def _process_data(self):
        start = self.data.n_past - 1
        prices = self.df.loc[:, 'Open'].to_numpy()[start:]
        self._calculate_values()
        if self.predictions:
            signal_features = self.df.loc[:, ['Open', 'Volume', 'SMA', 'RSI', 'OBV', 'prediction']].to_numpy()[start:]
        else:
            signal_features = self.df.loc[:, ['Open', 'Volume', 'SMA', 'RSI', 'OBV']].to_numpy()[start:]
        return prices, signal_features

    def predict(self):
        predictions = [0] * (self.data.n_past - 1)
        for i in range(self.data.n_past, len(self.df) + 1):
            pred_input = np.array([self.data.training_set_scaled[i - self.data.n_past:i, :]])
            prediction = self.model.predict(pred_input, verbose=0)
            prediction = self.data.pred_scaler.inverse_transform(prediction)[0][0]
            predictions.append(prediction)
        self.predictions = self.df['prediction'] = predictions
        self._process_data()

    def split_data(self, validation=None, train=None):
        if train:
            self.data.df = self.df.loc[:round(len(self.df) * train), :]
        if validation:
            self.data.df = self.df.loc[round(len(self.df) - len(self.df) * validation):, :]
        self.reset()

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
