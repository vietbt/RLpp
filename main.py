import gym
from sac import SAC
import sys



if sys.argv[1] == "1":
    env = gym.make('BreakoutNoFrameskip-v4')
    model = SAC(env, discrete=True, feature_extraction="cnn")
elif sys.argv[1] == "2":
    env = gym.make('CartPole-v0')
    model = SAC(env, discrete=True, feature_extraction="mlp", regularized=False)
else:
    env = gym.make('Pendulum-v0')
    model = SAC(env, discrete=False, feature_extraction="mlp")



model.learn(total_timesteps=500000)
