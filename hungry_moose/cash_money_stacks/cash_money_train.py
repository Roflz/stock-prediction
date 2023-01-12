# Goal:
# Make a model that chooses when to buy/sell a stock based off of predictions and/or other data
#
# Data we can use to go off of:
#   Stock price predictions from models
#   Current amount of money
#   Stock prices from the last X days
#   Prediction accuracy last X days
#   Prediction accuracy in total
#   Model MSE
#
#
# What we need to choose:
#   Which stocks to buy
#   Which stocks to sell
#   Which stocks to hold
#   How much stock to buy/sell
#
# Approach:
#   Make choices periodically, to start off we can act once a day to buy/sell/hold
#   Train a model that optimizes to make the most money based off the data we have, and the actions we take
#
# The model:
#   Need to train it on historical data, and optimize for maximum profit over time
#       Do we need to set it a future date that it is optimizing for?
#           i.e. do we train it to make the max increase in $$ tomorrow? Or max increase for a day in the future...?
#           Brainstorm...:
#               Predicting next day:
#                   Seems more universal... Predicting just 1 day in the future now shouldn't be too to much different
#                   from predicting 1 day in the future years ago
#               Predicting for max $$ over time:
#                   Might be more prone to overfitting on training data. If we train it on 8 years of data and train it
#                   to have the most $$ outcome at the end of that. That might not work quite the same for predicting in
#                   the present...
#                   This method is likely to make more $$ over time if it was done perfectly
#               What if we did something like train it to have the highest slope... d$/dt
#   Model should make decisions on each stock every day

# May need different kind of machine learning model:
# possibilities:
#   Supervised or Unsupervised Classification - similar to titty train but predicts a class label
#       Class labels could be buy/sell/hold
#       Might need to be unsupervised bc we dont necessarily have known answers in our inputs...
#       We might be able to create answers though...
#
# Reinforcement Learning: Reinforcement learning describes a class of problems where an agent operates in an
#   environment and must learn to operate using feedback. Reinforcement learning is learning what to do — how to map
#   situations to actions—so as to maximize a numerical reward signal. The learner is not told which actions to take,
#   but instead must discover which actions yield the most reward by trying them.


# region Parameters
# years = 10
# n_past = 300
# n_future = 1
# features = ["Open", "Close", "High", "Low", "Volume"]
# model_dict = {}
# epochs = 200
# batch_size = 32
# endregion

# import gym
# import random
#
# env = gym.make('CartPole-v1')
# states = env.observation_space.shape
# actions = env.action_space.n
#
# # episodes = 10
# # for episode in range(1, episodes + 1):
# #     state = env.reset()
# #     done = False
# #     score = 0
# #
# #     while not done:
# #         env.render()
# #         action = random.choice([0, 1])
# #         n_state, reward, done, trunc, info = env.step(action)
# #         score += reward
# #     print('Episode:{} Score:{}'.format(episode, score))
#
# import numpy as np
# from keras import Sequential
# from keras.layers import Dense, Flatten
# from keras.optimizers import Adam
#
#
# def build_model(states, actions):
#     model = Sequential()
#     model.add(Flatten(input_shape=states))
#     model.add(Dense(24, activation='relu'))
#     model.add(Dense(24, activation='relu'))
#     model.add(Dense(actions, activation='linear'))
#     return model
#
#
# # model = build_model(states, actions)
# # model.summary()
#
# from rl.agents import DQNAgent
# from rl.policy import BoltzmannQPolicy
# from rl.memory import SequentialMemory
#
#
# def build_agent(model, actions):
#     policy = BoltzmannQPolicy()
#     memory = SequentialMemory(limit=50000, window_length=1)
#     dqn = DQNAgent(model=model, memory=memory, policy=policy,
#                    nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
#     return dqn
#
#
# dqn = build_agent(build_model(states, actions), actions)
# dqn.compile(Adam(lr=1e-3), metrics=['mae'])
# dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)
#
# import gym
# from gym import Env
# import numpy as np
# from gym.spaces import Discrete, Box
# import random
#
#
# # create a custom class
# class ShowerEnv:
#     def __init__(self):
#         # Actions we can take, possible value 0,1,2 (Buy, Sell, Hold)
#         self.action_space = Discrete(3)
#         # This holds our current value. i.e. the current cash moneys we have, and/or current value of stock(s) we have
#         # Low/High gives us the minimum and maximum that our value can be
#         self.observation_space = Box(low=np.float32(np.array([0])), high=np.float32(np.array([100])),
#                                      dtype=np.float32)  # numbers between [0 100] continuous
#         # This is the starting value. Here, will be between [38-3.38+3]
#         self.state = 38 + random.randint(-3, 3)
#         # This is the length we are running for. Probably will be the amount of days worth of data that we are
#         # training this with
#         self.shower_length = 60  # duration of  temperature
#
#     def step(self, shower_action):
#         # if action =0, then decrease temperature,
#         # if action=1, leave unchanged
#         # if action=2, increase
#         self.state += shower_action - 1
#         # Reduce shower length by 1 second
#         self.shower_length -= 1
#
#         # Calculate Reward
#         if 37 <= self.state <= 39:
#             reward = 1
#         else:
#             reward = -1
#         if self.shower_length <= 0:
#             done = True
#         else:
#             done = False
#
#         # For us:
#         # if we make money
#         # reward = 1
#         # else
#         # reward = -1
#
#         info = ()
#         info = {}
#         return self.state, reward, done, info
#
#     def render(self):
#         pass
#
#     def reset(self):
#         # Reset temp
#         self.state = 38 + random.randint(-3, 3)
#         # Reset shower time
#         self.shower_length = 60
#         return self.state
#
#
# env = ShowerEnv()
# episodes = 10
# for episode in range(1, episodes + 1):
#     state = env.reset()
#     done = False
#     score = 0
#     while not done:
#         action = env.action_space.sample()
#         n_state, reward, done, info = env.step(action)
#         score += reward
#     print('Episode:{} Score:{}'.format(episode, score))
#
# import tensorflow as tf
# from keras.models import Sequential
# from keras.layers import Dense, Flatten
# from keras.optimizers import Adam
#
# env = ShowerEnv()
# states = env.observation_space.shape
# actions = env.action_space.n
#
#
# def build_model(states, actions):
#     model = Sequential()
#     model.add(Dense(units=24, activation='relu', input_shape=states))
#     model.add(Dense(units=24, activation='relu'))
#     model.add(Dense(actions, activation='linear'))
#     return model
#
#
# # model =build_model(states,actions)
# # model.compile(optimizer=Adam(learning_rate=1e-3), metrics=['mae'])
# # del model
# # print(model.summary())
# from rl.agents import DQNAgent
# from rl.policy import BoltzmannQPolicy
# from rl.memory import SequentialMemory
#
#
# def build_agent(model, actions):
#     policy = BoltzmannQPolicy()
#     memory = SequentialMemory(limit=50000, window_length=1)
#     dqn = DQNAgent(model=model, memory=memory, policy=policy, nb_actions=actions, nb_steps_warmup=10,
#                    target_model_update=1e-2)
#     return dqn
#
#
# dqn = build_agent(build_model(states, actions), actions)
# dqn.compile(optimizer=Adam(learning_rate=1e-3), metrics=['mae'])
# dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)
#
# scores = dqn.test(env, nb_episodes=100, visualize=False)
# print(np.mean(scores.history['episode_reward']))

# _ = dqn.test(env, nb_episodes=15, visualize=True)


# Gym stuff
import gym
import gym_anytrading

# Stable baselines - rl stuff
# import tensorflow.contrib.layers as tf_layers
from tensorflow.python.compiler.tensorrt
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C

# Processing libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv('data/gmedata.csv')
df.head()

df['Date'] = pd.to_datetime(df['Date'])
df.dtypes

df.set_index('Date', inplace=True)
df.head()

env = gym.make('stocks-v0', df=df, frame_bound=(5,100), window_size=5)

env.signal_features

env.action_space

state = env.reset()
while True:
    action = env.action_space.sample()
    n_state, reward, done, info = env.step(action)
    if done:
        print("info", info)
        break

plt.figure(figsize=(15, 6))
plt.cla()
env.render_all()
plt.show()

env_maker = lambda: gym.make('stocks-v0', df=df, frame_bound=(5,100), window_size=5)
env = DummyVecEnv([env_maker])

model = A2C('MlpLstmPolicy', env, verbose=1)
model.learn(total_timesteps=1000000)

env = gym.make('stocks-v0', df=df, frame_bound=(90,110), window_size=5)
obs = env.reset()
while True:
    obs = obs[np.newaxis, ...]
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    if done:
        print("info", info)
        break

plt.figure(figsize=(15,6))
plt.cla()
env.render_all()
plt.show()