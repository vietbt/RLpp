import gym
from sac import SAC

env = gym.make('Pendulum-v0')
model = SAC(env)
model.learn(total_timesteps=50000)
