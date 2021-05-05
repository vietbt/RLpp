import gym
from sac import SAC
import sys



if sys.argv[1] == "1":
    env = gym.make('BreakoutNoFrameskip-v4')
    model = SAC(env, discrete=True, feature_extraction="cnn")
else:
    env = gym.make('Pendulum-v0')
    model = SAC(env, discrete=False, feature_extraction="mlp")



model.learn(total_timesteps=500000)
