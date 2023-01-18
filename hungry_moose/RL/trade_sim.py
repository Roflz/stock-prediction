import os

from gym_anytrading.envs import StocksEnv
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import StopTrainingOnRewardThreshold, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv

from .stock_env import StockEnv
from gym_anytrading.datasets import FOREX_EURUSD_1H_ASK, STOCKS_GOOGL
from finta import TA


class TradeSim:

    def __init__(self, df, window_size, frame_bound, model=None):
        self.model = model
        self.df = df
        self.window_size = window_size
        self.log_path = os.path.join('training', 'logs')
        self.model_path = os.path.join('training', 'models')
        self.frame_bound = frame_bound
        self.callbacks = []
        self.env = self.reset_env()

        self.df.sort_values('Date', ascending=True, inplace=True)

    def calculate_values(self):
        # Calculate SMA, RSI, and OBV
        self.df['SMA'] = TA.SMA(self.df, 12)
        self.df['RSI'] = TA.RSI(self.df)
        self.df['OBV'] = TA.OBV(self.df)
        self.df.fillna(0, inplace=True)
        print(self.df.head(15))

    def set_env(self):
        self.env = StockEnv(self.df, self.window_size, self.frame_bound)
        return self.env

    def reset_env(self):
        self.env = StockEnv(self.df, self.window_size, self.frame_bound)
        return self.env

    def make_env(self):
        self.env = DummyVecEnv([lambda: self.env])
        return self.env

    def add_callbacks(self, eval_freq: int, reward_threshold=0, stop_callback=False):
        if stop_callback:
            stop_callback = StopTrainingOnRewardThreshold(reward_threshold=reward_threshold, verbose=1)
            eval_callback = EvalCallback(self.env,
                                         callback_on_new_best=stop_callback,
                                         eval_freq=10000,
                                         best_model_save_path=self.model_path,
                                         verbose=1)
            self.callbacks += stop_callback
            self.callbacks += eval_callback
        else:
            eval_callback = EvalCallback(self.env,
                                         eval_freq=10000,
                                         best_model_save_path=self.model_path,
                                         verbose=1)
            self.callbacks += eval_callback

    def train_model(self, algorithm, policy, timesteps):
        if algorithm == "A2C":
            self.model = A2C(policy, self.env, verbose=1, tensorboard_log=self.log_path)
            self.model.learn(total_timesteps=timesteps, callback=self.callbacks)
        if algorithm == "PPO":
            self.model = PPO(policy, self.env, verbose=1, tensorboard_log=self.log_path)
            self.model.learn(total_timesteps=timesteps, callback=self.callbacks)

    def test_random(self, episodes: int):
        for episode in range(1, episodes + 1):
            obs = self.env.reset()
            done = False
            score = 0

            while not done:
                # env.render()
                action = self.env.action_space.sample()
                obs, reward, done, info = self.env.step(action)
                score += reward
            print(info)
            print('Episode:{} Score:{}'.format(episode, score))
        self.env.close()

    def save_model(self, name):
        self.model.save(os.path.join(self.model_path, name))

    def test_model(self, episodes):
        for episode in range(1, episodes + 1):
            obs = self.env.reset()
            done = False
            score = 0

            while not done:
                # env.render()
                action = self.model.predict(obs)
                obs, reward, done, info = self.env.step(action[0].min())
                score += reward
            print(info)
            print('Episode:{} Score:{}'.format(episode, score))
        self.env.close()
