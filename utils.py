import numpy as np
import tensorflow as tf
from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete

def scale_action(action_space, action):
    if isinstance(action_space, Discrete):
        return action
    low, high = action_space.low, action_space.high
    return 2.0 * ((action - low) / (high - low)) - 1.0

def unscale_action(action_space, scaled_action):
    if isinstance(action_space, Discrete):
        return scaled_action
    low, high = action_space.low, action_space.high
    return low + 0.5*(scaled_action + 1.0)*(high - low)

def get_schedule_fn(value_schedule):
    return constfn(float(value_schedule))

def constfn(val):
    def func(_):
        return val
    return func

def apply_squashing_func(mu, pi, logp_pi, EPS=1e-6):
    deterministic_policy = tf.tanh(mu)
    policy = tf.tanh(pi)
    logp_pi -= tf.reduce_sum(tf.log(1 - policy**2 + EPS), axis=1)
    return deterministic_policy, policy, logp_pi

def gaussian_likelihood(x, mu, log_std, EPS=1e-6):
    pre_sum = -0.5*(((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1)

def gaussian_entropy(log_std):
    return tf.reduce_sum(log_std + 0.5*np.log(2*np.pi*np.e), axis=-1)

def clip_but_pass_gradient(x, lower=-1., upper=1.):
    clip_up = tf.cast(x > upper, tf.float32)
    clip_low = tf.cast(x < lower, tf.float32)
    return x + tf.stop_gradient((upper-x)*clip_up + (lower-x)*clip_low)

def observation_input(ob_space, batch_size=None, name='Ob', scale=False):
    if isinstance(ob_space, Discrete):
        observation_ph = tf.placeholder(shape=(batch_size,), dtype=tf.int32, name=name)
        processed_observations = tf.cast(tf.one_hot(observation_ph, ob_space.n), tf.float32)
        return observation_ph, processed_observations
    elif isinstance(ob_space, Box):
        observation_ph = tf.placeholder(shape=(batch_size,) + ob_space.shape, dtype=ob_space.dtype, name=name)
        processed_observations = tf.cast(observation_ph, tf.float32)
        if (scale and not np.any(np.isinf(ob_space.low)) and not np.any(np.isinf(ob_space.high)) and
           np.any((ob_space.high - ob_space.low) != 0)):
            processed_observations = ((processed_observations - ob_space.low) / (ob_space.high - ob_space.low))
        return observation_ph, processed_observations
    elif isinstance(ob_space, MultiBinary):
        observation_ph = tf.placeholder(shape=(batch_size, ob_space.n), dtype=tf.int32, name=name)
        processed_observations = tf.cast(observation_ph, tf.float32)
        return observation_ph, processed_observations
    elif isinstance(ob_space, MultiDiscrete):
        observation_ph = tf.placeholder(shape=(batch_size, len(ob_space.nvec)), dtype=tf.int32, name=name)
        processed_observations = tf.concat([
            tf.cast(tf.one_hot(input_split, ob_space.nvec[i]), tf.float32) for i, input_split
            in enumerate(tf.split(observation_ph, len(ob_space.nvec), axis=-1))
        ], axis=-1)
        return observation_ph, processed_observations

def get_trainable_vars(name):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)