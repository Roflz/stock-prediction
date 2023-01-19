import os

from gym_anytrading.envs import StocksEnv
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import StopTrainingOnRewardThreshold, EvalCallback, \
    StopTrainingOnNoModelImprovement
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv

from .stock_env import StockEnv
from leaves.databitch import DataBitch
from gym_anytrading.datasets import FOREX_EURUSD_1H_ASK, STOCKS_GOOGL
from finta import TA


class TradeSim:

    def __init__(self, env, model=None):
        self.env = env
        self.model = model
        self.log_path = os.path.join('training', 'logs')
        self.model_path = os.path.join('training', 'models')
        self.callbacks = None

        self.env.df.sort_values('Date', ascending=True, inplace=True)

    def set_env(self):
        self.env = StockEnv(self.env.data, window_size=self.env.window_size, model=self.env.model)
        return self.env

    def reset_env(self):
        self.env = StockEnv(self.env.data, window_size=self.env.window_size, model=self.env.model)
        return self.env

    def make_env(self):
        self.env = DummyVecEnv([lambda: self.env])
        return self.env

    def add_callbacks(self, reward_threshold=0, stop_callback=False):
        stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=10, min_evals=5, verbose=1)
        if stop_callback:
            stop_callback = StopTrainingOnRewardThreshold(reward_threshold=reward_threshold, verbose=1)
            eval_callback = EvalCallback(self.env,
                                         callback_on_new_best=stop_callback,
                                         eval_freq=10000,
                                         best_model_save_path=self.model_path,
                                         callback_after_eval=stop_train_callback,
                                         verbose=1)
            self.callbacks += stop_callback
            self.callbacks += eval_callback
        else:
            eval_callback = EvalCallback(self.env,
                                         eval_freq=10000,
                                         best_model_save_path=self.model_path,
                                         callback_after_eval=stop_train_callback,
                                         verbose=1)
            self.callbacks = [eval_callback]

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
            history = {0: 0, 1: 0, 2: 0}

            while not done:
                # env.render()
                action = self.env.action_space.sample()
                obs, reward, done, info = self.env.step(action)
                history[action] += 1
            print(info)
            print('Episode: {}. Buy: {}, Sell: {}, Hold: {}'.format(episode, history[0], history[1], history[2]))
        self.env.close()

    def save_model(self, name):
        self.model.save(os.path.join(self.model_path, name))

    def test_model(self, episodes):
        for episode in range(1, episodes + 1):
            obs = self.env.reset()
            done = False
            history = {0: 0, 1: 0, 2: 0}

            while not done:
                # env.render()
                action = self.model.predict(obs)
                obs, reward, done, info = self.env.step(action[0].min())
                history[action[0].min()] += 1
            print(info)
            print('Episode: {}. Buy: {}, Sell: {}, Hold: {}'.format(episode, history[0], history[1], history[2]))
        self.env.close()
