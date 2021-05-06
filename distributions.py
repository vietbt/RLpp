from gym import spaces
from layers import linear
import tensorflow as tf
import numpy as np

class DiagGaussianProbabilityDistribution:
    def __init__(self, flat):
        self.flat = flat
        mean, logstd = tf.split(axis=len(flat.shape) - 1, num_or_size_splits=2, value=flat)
        self.mean = mean
        self.logstd = logstd
        self.std = tf.exp(logstd)
    
    @classmethod
    def proba_distribution_from_latent(cls, size, pi_latent_vector, vf_latent_vector, init_scale=1.0, init_bias=0.0):
        mean = linear(pi_latent_vector, 'pi', size, init_scale=init_scale, init_bias=init_bias)
        logstd = tf.get_variable(name='pi/logstd', shape=[1, size], initializer=tf.zeros_initializer())
        pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
        q_values = linear(vf_latent_vector, 'q', size, init_scale=init_scale, init_bias=init_bias)
        return cls(pdparam), mean, q_values

    def sample(self):
        return self.mean + self.std * tf.random_normal(tf.shape(self.mean), dtype=self.mean.dtype)

    def mode(self):
        return self.mean
        
    def neglogp(self, x):
        return 0.5 * tf.reduce_sum(tf.square((x - self.mean) / self.std), axis=-1) \
               + 0.5 * np.log(2.0 * np.pi) * tf.cast(tf.shape(x)[-1], tf.float32) \
               + tf.reduce_sum(self.logstd, axis=-1)

    def entropy(self):
        return tf.reduce_sum(self.logstd + .5 * np.log(2.0 * np.pi * np.e), axis=-1)


class SquashedGaussianDistribution:
    def __init__(self, flat):
        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20
        self.EPS = 1e-6
        self.flat = flat
        mean, logstd = tf.split(axis=len(flat.shape) - 1, num_or_size_splits=2, value=flat)
        self.mean = mean
        self.logstd = tf.clip_by_value(logstd, self.LOG_STD_MIN, self.LOG_STD_MAX)
        self.std = tf.exp(logstd)
        
    
    @classmethod
    def proba_distribution_from_latent(cls, size, pi_latent_vector, vf_latent_vector, init_scale=1.0, init_bias=0.0):
        mean = linear(pi_latent_vector, 'pi', size, init_scale=init_scale, init_bias=init_bias)
        logstd = linear(pi_latent_vector, 'logstd', size, init_scale=init_scale, init_bias=init_bias)
        # logstd = tf.get_variable(name='pi/logstd', shape=[1, size], initializer=tf.zeros_initializer())
        pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
        q_values = linear(vf_latent_vector, 'q', size, init_scale=init_scale, init_bias=init_bias)
        return cls(pdparam), mean, q_values

    def sample(self):
        return tf.tanh(self.mean + self.std * tf.random_normal(tf.shape(self.mean), dtype=self.mean.dtype))
    
    def mode(self):
        return tf.tanh(self.mean)
        
    def neglogp(self, x):
        neglogp_likelihood = 0.5 * tf.reduce_sum(tf.square((tf.atanh(x)-self.mean)/(self.std+self.EPS)), axis=-1) \
               + 0.5 * np.log(2.0 * np.pi) * tf.cast(tf.shape(x)[-1], tf.float32) \
               + tf.reduce_sum(self.logstd, axis=-1)
        policy = self.sample()
        return neglogp_likelihood + tf.reduce_sum(tf.log(1 - policy**2 + self.EPS), axis=-1)


    def entropy(self):
        return tf.reduce_sum(self.logstd + .5 * np.log(2.0 * np.pi * np.e), axis=-1)


class CategoricalProbabilityDistribution:
    def __init__(self, logits):
        self.logits = logits
    
    @classmethod
    def proba_distribution_from_latent(cls, n_cat, pi_latent_vector, vf_latent_vector, init_scale=1.0, init_bias=0.0):
        pdparam = linear(pi_latent_vector, 'pi', n_cat, init_scale=init_scale, init_bias=init_bias)
        q_values = linear(vf_latent_vector, 'q', n_cat, init_scale=init_scale, init_bias=init_bias)
        return cls(pdparam), pdparam, q_values

    def sample(self):
        uniform = tf.random_uniform(tf.shape(self.logits), dtype=self.logits.dtype)
        return tf.argmax(self.logits - tf.log(-tf.log(uniform)), axis=-1)

    def mode(self):
        return tf.argmax(self.logits, axis=-1)
        
    def prob(self):
        logits = self.logits - tf.reduce_max(self.logits, axis=-1, keep_dims=True)
        prob = tf.nn.softmax(logits, -1)
        prob = tf.maximum(prob, 1e-8)
        return prob

    def neglogp(self, actions):
        actions = tf.cast(actions, tf.int32)
        prob = self.prob()
        onehot = tf.one_hot(actions, self.logits.get_shape().as_list()[-1])
        neglogpac = -tf.log(tf.reduce_sum(tf.multiply(prob, onehot), axis=-1))
        return neglogpac

    def entropy(self):
        prob = self.prob()
        entropy = -tf.reduce_sum(prob*tf.log(prob), axis=-1)
        return entropy


class MultiCategoricalProbabilityDistribution:
    def __init__(self, nvec, flat):
        self.flat = flat
        self.categoricals = list(map(CategoricalProbabilityDistribution, tf.split(flat, nvec, axis=-1)))

    @classmethod
    def proba_distribution_from_latent(cls, n_vec, pi_latent_vector, vf_latent_vector, init_scale=1.0, init_bias=0.0):
        pdparam = linear(pi_latent_vector, 'pi', sum(n_vec), init_scale=init_scale, init_bias=init_bias)
        q_values = linear(vf_latent_vector, 'q', sum(n_vec), init_scale=init_scale, init_bias=init_bias)
        return cls(n_vec, pdparam), pdparam, q_values

    def sample(self):
        return tf.stack([p.sample() for p in self.categoricals], axis=-1)

    def mode(self):
        return tf.stack([p.mode() for p in self.categoricals], axis=-1)
        
    def neglogp(self, actions):
        return tf.add_n([p.neglogp(px) for p, px in zip(self.categoricals, tf.unstack(x, axis=-1))])

    def entropy(self):
        return tf.add_n([p.entropy() for p in self.categoricals])


class BernoulliProbabilityDistribution:
    def __init__(self, logits):
        self.logits = logits
        self.probabilities = tf.sigmoid(logits)
    
    @classmethod
    def proba_distribution_from_latent(cls, size, pi_latent_vector, vf_latent_vector, init_scale=1.0, init_bias=0.0):
        pdparam = linear(pi_latent_vector, 'pi', size, init_scale=init_scale, init_bias=init_bias)
        q_values = linear(vf_latent_vector, 'q', size, init_scale=init_scale, init_bias=init_bias)
        return cls(pdparam), pdparam, q_values

    def sample(self):
        samples_from_uniform = tf.random_uniform(tf.shape(self.probabilities))
        return tf.cast(math_ops.less(samples_from_uniform, self.probabilities), tf.float32)

    def mode(self):
        return tf.round(self.probabilities)

    def neglogp(self, x):
        return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=tf.cast(x, tf.float32)), axis=-1)

    def entropy(self):
        return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.probabilities), axis=-1)


def make_proba_dist(ac_space, pi_latent_vector, vf_latent_vector, init_scale=1.0, init_bias=0.0, use_squashed=False):
    if isinstance(ac_space, spaces.Box):
        if use_squashed:
            return SquashedGaussianDistribution.proba_distribution_from_latent(ac_space.shape[0], pi_latent_vector, vf_latent_vector, init_scale, init_bias)
        else:
            return DiagGaussianProbabilityDistribution.proba_distribution_from_latent(ac_space.shape[0], pi_latent_vector, vf_latent_vector, init_scale, init_bias)
    elif isinstance(ac_space, spaces.Discrete):
        return CategoricalProbabilityDistribution.proba_distribution_from_latent(ac_space.n, pi_latent_vector, vf_latent_vector, init_scale, init_bias)
    elif isinstance(ac_space, spaces.MultiDiscrete):
        return MultiCategoricalProbabilityDistribution.proba_distribution_from_latent(ac_space.nvec, pi_latent_vector, vf_latent_vector, init_scale, init_bias)
    elif isinstance(ac_space, spaces.MultiBinary):
        return BernoulliProbabilityDistribution.proba_distribution_from_latent(ac_space.n, pi_latent_vector, vf_latent_vector, init_scale, init_bias)