import tensorflow as tf
import joblib
from layers import mlp_extractor, nature_cnn, linear
from utils import gaussian_likelihood, gaussian_entropy, apply_squashing_func, observation_input

class SACAgent:
    def __init__(self, sess, env, net_arch=[64, 64], scale=False, feature_extraction="mlp", act_fun=tf.nn.relu, layer_norm=False, mode="self_regularized"):
        self.sess = sess
        self.env = env
        with tf.variable_scope("input", reuse=False):
            self.obs_ph, self.processed_obs_ph = observation_input(env.observation_space, scale=scale, name="policy_tf")
            self.next_obs_ph, self.processed_next_obs_ph = observation_input(env.observation_space, scale=scale, name="target_policy")
            self.actions_ph = tf.placeholder(tf.float32, shape=(None,) + env.action_space.shape, name='action_ph')

        kwargs = dict(feature_extraction=feature_extraction, net_arch=net_arch, act_fun=act_fun, layer_norm=layer_norm)
        with tf.variable_scope("model", reuse=False):    
            _, self.policy_out, self.logp_pi, self.entropy = self.make_actor(self.processed_obs_ph, **kwargs)
            self.qf1, self.qf2, self.value_fn = self.make_critics(self.processed_obs_ph, self.actions_ph, create_qf=True, create_vf=True, **kwargs)
            self.qf1_pi, self.qf2_pi, _  = self.make_critics(self.processed_obs_ph, self.policy_out, create_qf=True, create_vf=False, reuse=True, **kwargs)
            if mode == "self_regularized":
                self.next_qf1, self.next_qf2, self.next_value_fn  = self.make_critics(self.processed_next_obs_ph, self.actions_ph, create_qf=True, create_vf=True, reuse=True, **kwargs)

        with tf.variable_scope("target", reuse=False):
            _, _, self.value_target = self.make_critics(self.processed_next_obs_ph, create_qf=False, create_vf=True, **kwargs)
            

        self.params = tf.trainable_variables()

    def step(self, obs):
        return self.sess.run(self.policy_out, {self.obs_ph: obs})

    def make_actor(self, obs, reuse=False, scope="pi", feature_extraction="cnn", net_arch=[64, 64], act_fun=tf.nn.relu, layer_norm=True, LOG_STD_MAX=2, LOG_STD_MIN=-20):
        with tf.variable_scope(scope, reuse=reuse):
            if feature_extraction == "cnn":
                pi_h = nature_cnn(obs)
            else:
                pi_h = tf.layers.flatten(obs)
            pi_h = mlp_extractor(pi_h, net_arch, act_fun, layer_norm=layer_norm)
            mu = linear(pi_h, "mu", self.env.action_space.shape[0])
            log_std = linear(pi_h, "log_std", self.env.action_space.shape[0])
        log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = std = tf.exp(log_std)
        pi = mu + tf.random_normal(tf.shape(mu)) * std
        logp_pi = gaussian_likelihood(pi, mu, log_std)
        entropy = gaussian_entropy(log_std)
        deterministic_policy, policy, logp_pi = apply_squashing_func(mu, pi, logp_pi)
        return deterministic_policy, policy, logp_pi, entropy

    def make_critics(self, obs, action=None, reuse=False, scope="values_fn", create_vf=True, create_qf=True, feature_extraction="cnn", net_arch=[64, 64], act_fun=tf.nn.relu, layer_norm=True):
        value_fn, qf1, qf2 = None, None, None
        with tf.variable_scope(scope, reuse=reuse):
            if feature_extraction == "cnn":
                critics_h = nature_cnn(obs)
            else:
                critics_h = tf.layers.flatten(obs)
            if create_vf:
                with tf.variable_scope('vf', reuse=reuse):
                    vf_h = mlp_extractor(critics_h, net_arch, act_fun, layer_norm=layer_norm)
                    value_fn = linear(vf_h, "vf", 1)
            if create_qf:
                qf_h = tf.concat([critics_h, action], axis=-1)
                with tf.variable_scope('qf1', reuse=reuse):
                    qf1_h = mlp_extractor(qf_h, net_arch, act_fun, layer_norm=layer_norm)
                    qf1 = linear(qf1_h, "qf1", 1)
                with tf.variable_scope('qf2', reuse=reuse):
                    qf2_h = mlp_extractor(qf_h, net_arch, act_fun, layer_norm=layer_norm)
                    qf2 = linear(qf2_h, "qf2", 1)
        return qf1, qf2, value_fn

    def get_params(self):
        return self.sess.run(self.params)

    def load_params(self, params):
        restores = []
        for p, param in zip(self.params, params):
            restores.append(p.assign(param))
        self.sess.run(restores)

    def save_params(self, path):
        params = self.sess.run(self.params)
        joblib.dump(params, path)