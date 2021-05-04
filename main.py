import gym
from sac import SAC

env = gym.make('Pendulum-v0')
print("action_space:", env.action_space)
print("observation_space:", env.observation_space)
model = SAC(env)
model.learn(total_timesteps=500000)
