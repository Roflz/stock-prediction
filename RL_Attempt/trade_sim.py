import os

from gym_anytrading.envs import StocksEnv
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import StopTrainingOnRewardThreshold, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv

from stock_env import StockEnv
from gym_anytrading.datasets import FOREX_EURUSD_1H_ASK, STOCKS_GOOGL
from finta import TA

df = STOCKS_GOOGL
window_size = 10
log_path = os.path.join('training', 'logs')
model_path = os.path.join('training', 'models')

df.sort_values('Date', ascending=True, inplace=True)

# Calculate SMA, RSI, and OBV
df['SMA'] = TA.SMA(df, 12)
df['RSI'] = TA.RSI(df)
df['OBV'] = TA.OBV(df)
df.fillna(0, inplace=True)
print(df.head(15))


def add_signals(env):
    start = env.frame_bound[0] - env.window_size
    end = env.frame_bound[1]
    prices = env.df.loc[:, 'Open'].to_numpy()[start:end]
    signal_features = env.df.loc[:, ['Open', 'Volume', 'SMA', 'RSI', 'OBV']].to_numpy()[start:end]
    return prices, signal_features


class MyCustomEnv(StocksEnv):
    _process_data = add_signals


env = StockEnv(df, window_size, frame_bound=(21, 2000))
# env = MyCustomEnv(df=df, window_size=10, frame_bound=(21, 2000))

episodes = 5
for episode in range(1, episodes + 1):
    obs = env.reset()
    done = False
    score = 0

    while not done:
        # env.render()
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        score += reward
    print(info)
    print('Episode:{} Score:{}'.format(episode, score))
env.close()

# env_maker = lambda: env
# env = DummyVecEnv([env_maker])

# Stops the model if reward is X amount, I think that it only checks when the eval_callback is called
# stop_callback = StopTrainingOnRewardThreshold(reward_threshold=1000, verbose=1)
# Evaluates the model every X timesteps
eval_callback = EvalCallback(env,
                             # callback_on_new_best=stop_callback,
                             eval_freq=10000,
                             best_model_save_path=model_path,
                             verbose=1)

# Test Model
model = A2C("MlpPolicy", env, verbose=1, tensorboard_log=log_path)
model.learn(total_timesteps=100000, callback=eval_callback)

# Save model
model.save(os.path.join(model_path, "test_model"))
# print(evaluate_policy(model, env, n_eval_episodes=10, render=False))

env = StockEnv(df, window_size, frame_bound=(2001, 2300))
# env = MyCustomEnv(df=df, window_size=10, frame_bound=(2001, 2300))
episodes = 5
for episode in range(1, episodes + 1):
    obs = env.reset()
    done = False
    score = 0

    while not done:
        # env.render()
        action = model.predict(obs)
        obs, reward, done, info = env.step(action[0].min())
        score += reward
    print(info)
    print('Episode:{} Score:{}'.format(episode, score))
env.close()
