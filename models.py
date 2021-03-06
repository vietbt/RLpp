import tensorflow as tf
import numpy as np
from utils import get_trainable_vars

class SACModel:
    def __init__(self, sess, policy, target_entropy='auto', ent_coef='auto', gamma=0.99, tau=0.005):
        self.sess = sess
        self.policy = policy
        with tf.variable_scope("loss", reuse=False):
            self.terminals_ph = tf.placeholder(tf.float32, shape=(None, None), name='terminals')
            self.rewards_ph = tf.placeholder(tf.float32, shape=(None, None), name='rewards')
            self.learning_rate_ph = tf.placeholder(tf.float32, [], name='learning_rate_ph')

            self.entropy = tf.reduce_mean(policy.entropy)
            
            if target_entropy == 'auto':
                if policy.discrete:
                    target_entropy = -np.log(1.0/policy.env.action_space.n) * 0.98
                else:
                    target_entropy = -np.prod(policy.env.action_space.shape).astype(np.float32)
            else:
                target_entropy = float(target_entropy)
            
            if isinstance(ent_coef, str) and ent_coef.startswith('auto'):
                init_value = 1.0
                if '_' in ent_coef:
                    init_value = float(ent_coef.split('_')[1])
                log_ent_coef = tf.get_variable('log_ent_coef', dtype=tf.float32, initializer=np.log(init_value).astype(np.float32))
                ent_coef = tf.exp(log_ent_coef)
            else:
                ent_coef = float(ent_coef)


            if policy.regularized:
                q_backup = tf.stop_gradient(self.rewards_ph + gamma*(1-self.terminals_ph)*policy.next_value_fn)
            else:
                q_backup = tf.stop_gradient(self.rewards_ph + gamma*(1-self.terminals_ph)*policy.value_target)


            qf1_loss = 0.5*tf.reduce_mean((q_backup-policy.qf1)**2)
            qf2_loss = 0.5*tf.reduce_mean((q_backup-policy.qf2)**2)

            if policy.regularized:
                qf1_loss += 0.5*tf.reduce_mean((policy.next_qf1 - policy.qf1)**2)
                qf2_loss += 0.5*tf.reduce_mean((policy.next_qf2 - policy.qf2)**2)
            
            if policy.discrete:
                policy_out = -policy.pd.neglogp(policy.actions_ph)

            ent_coef_loss, entropy_optimizer = None, None
            if not isinstance(ent_coef, float):
                if policy.discrete:
                    ent_coef_loss = tf.reduce_mean(
                        tf.reduce_sum(
                            tf.multiply(
                                tf.stop_gradient(policy_out), 
                                -log_ent_coef * tf.stop_gradient(policy.logp_pi + target_entropy)
                            ),
                            axis=-1
                        )
                    )
                else:
                    ent_coef_loss = -tf.reduce_mean(log_ent_coef * tf.stop_gradient(policy.logp_pi + target_entropy))
                entropy_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
            
            if policy.discrete:
                policy_loss = tf.reduce_mean(
                    tf.reduce_sum(
                        tf.multiply(
                            policy_out, 
                            ent_coef * policy.logp_pi - tf.stop_gradient(policy.qf1)
                        ),
                        axis=-1
                    )
                )
            else:
                policy_loss = tf.reduce_mean(ent_coef*policy.logp_pi - policy.qf1_pi)
            
            if not policy.discrete:
                min_qf_pi = tf.minimum(policy.qf1_pi, policy.qf2_pi)
                v_backup = tf.stop_gradient(min_qf_pi - ent_coef*policy.logp_pi)
                value_loss = 0.5*tf.reduce_mean((policy.value_fn-v_backup)**2)
                values_losses = qf1_loss + qf2_loss + value_loss
                self.step_ops = [policy_loss, qf1_loss, qf2_loss, value_loss]
            else:
                values_losses = qf1_loss + qf2_loss
                self.step_ops = [policy_loss, qf1_loss, qf2_loss]

            policy_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
            policy_train_op = policy_optimizer.minimize(policy_loss, var_list=get_trainable_vars('model/pi'))

            value_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
            values_params = get_trainable_vars('model/values_fn')
            source_params = get_trainable_vars("model/values_fn")
            target_params = get_trainable_vars("target/values_fn")

            self.target_update_op = [tf.assign(target, (1-tau)*target + tau*source) for target, source in zip(target_params, source_params)]
            self.target_init_op = [tf.assign(target, source) for target, source in zip(target_params, source_params)]

            with tf.control_dependencies([policy_train_op]):
                train_values_op = value_optimizer.minimize(values_losses, var_list=values_params)
                self.step_ops += [policy.qf1, policy.qf2, policy.value_fn, policy.logp_pi, policy.entropy, policy_train_op, train_values_op]

                if ent_coef_loss is not None:
                    with tf.control_dependencies([train_values_op]):
                        ent_coef_op = entropy_optimizer.minimize(ent_coef_loss, var_list=log_ent_coef)
                        self.step_ops += [ent_coef_op, ent_coef_loss, ent_coef]
            