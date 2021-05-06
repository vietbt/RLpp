import numpy as np
import tensorflow as tf
from utils import unscale_action, scale_action, get_schedule_fn
from buffers import ReplayBuffer
from agents import SACAgent
from models import SACModel

class SAC:
    def __init__(self, env, gamma=0.99, tau=0.005, learning_rate=3e-4, buffer_size=50000,
                 learning_starts=100, train_freq=1, batch_size=64,
                 target_update_interval=1, gradient_steps=1, target_entropy='auto', ent_coef='auto', 
                 random_exploration=0.0, discrete=True, regularized=True, feature_extraction="cnn"):
        self.env = env
        self.learning_starts = learning_starts
        self.random_exploration = random_exploration
        self.train_freq = train_freq
        self.target_update_interval = target_update_interval
        self.batch_size = batch_size
        self.gradient_steps = gradient_steps
        self.learning_rate = learning_rate

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf.Session(graph=self.graph)
            self.replay_buffer = ReplayBuffer(buffer_size)
            self.agent = SACAgent(self.sess, env, discrete=discrete, regularized=regularized, feature_extraction=feature_extraction)
            self.model = SACModel(self.sess, self.agent, target_entropy, ent_coef, gamma, tau)
            with self.sess.as_default():
                self.sess.run(tf.global_variables_initializer())
                self.sess.run(self.model.target_init_op)
        self.num_timesteps = 0

    def train(self, learning_rate):
        batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones = self.replay_buffer.sample(self.batch_size)
        # print("batch_actions:", batch_actions.shape)
        # print("self.agent.actions_ph:", self.agent.actions_ph)
        
        feed_dict = {
            self.agent.obs_ph: batch_obs,
            self.agent.next_obs_ph: batch_next_obs,
            self.model.rewards_ph: batch_rewards.reshape(self.batch_size, -1),
            self.model.terminals_ph: batch_dones.reshape(self.batch_size, -1),
            self.model.learning_rate_ph: learning_rate
        }
        if not self.agent.discrete:
            feed_dict[self.agent.actions_ph] = batch_actions
        else:
            batch_actions = batch_actions.reshape(-1)
            feed_dict[self.agent.actions_ph] = batch_actions
        policy_loss, qf1_loss, qf2_loss, value_loss, *values = self.sess.run(self.model.step_ops, feed_dict)
        return policy_loss, qf1_loss, qf2_loss

    def learn(self, total_timesteps):
        learning_rate = get_schedule_fn(self.learning_rate)
        episode_rewards = [0]
        mb_losses = []
        obs = self.env.reset()
        for step in range(total_timesteps):
            if self.num_timesteps < self.learning_starts or np.random.rand() < self.random_exploration:
                unscaled_action = self.env.action_space.sample()
                action = scale_action(self.env.action_space, unscaled_action)
            else:
                action = self.agent.step(obs[None]).flatten()
                unscaled_action = unscale_action(self.env.action_space, action)
            # print("\nunscaled_action:", unscaled_action)
            new_obs, reward, done, _ = self.env.step(unscaled_action)
            self.num_timesteps += 1
            self.replay_buffer.add(obs, action, reward, new_obs, done)
            obs = new_obs
            
            if self.num_timesteps % self.train_freq == 0:
                for grad_step in range(self.gradient_steps):
                    if not self.replay_buffer.can_sample(self.batch_size) or self.num_timesteps < self.learning_starts:
                        break
                    frac = 1.0 - step / total_timesteps
                    current_lr = learning_rate(frac)
                    mb_losses.append(self.train(current_lr))
                    if (step + grad_step) % self.target_update_interval == 0:
                        self.sess.run(self.model.target_update_op)

            episode_rewards[-1] += reward
            if done:
                obs = self.env.reset()
                episode_rewards.append(0)
                
            mean_reward = round(float(np.mean(episode_rewards[-101:-1])), 1)
            loss_str = "/".join([f"{x:.3f}" for x in np.mean(mb_losses, 0)]) if len(mb_losses) > 0 else "NaN"
            print(f"Step {step} - reward: {mean_reward} - loss: {loss_str}", end="\n" if step%500==0 else "\r")