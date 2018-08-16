import sys
import gym.spaces
import itertools
import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.layers as layers
from collections import namedtuple
from utils.dqn_utils import *

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs", "lr_schedule"])


def learn(env,
          options,
          q_func,
          optimizer_spec,
          session,
          scope_name,
          exploration=LinearSchedule(300000, 0.1),
          stopping_criterion=None,
          replay_buffer_size=10000,
          batch_size=32,
          gamma=0.99,
          learning_starts=5000,
          learning_freq=1,
          frame_history_len=1,
          target_update_freq=1000,
          grad_norm_clipping=10):


    assert type(env.observation_space) == gym.spaces.Box
    assert type(env.action_space) == gym.spaces.Discrete

    if len(env.observation_space.shape) == 1:
        # This means we are running on low-dimensional observations (e.g. RAM)
        input_shape = env.observation_space.shape
    else:
        img_h, img_w, img_c = env.observation_space.shape
        input_shape = (img_h, img_w, frame_history_len * img_c)  # size_x, size_y,

    # taking into account options
    num_actions = env.action_space.n + len(options)

    ###  1. Set up placeholders
    with tf.variable_scope(scope_name):
        #         with tf.variable_scope("obs_t_ph"):
        # placeholder for current observation (or state)
        obs_t_ph = tf.placeholder(tf.uint8, [None] + list(input_shape), name="obs_t_ph")
        # casting to float on GPU ensures lower data transfer times.
        obs_t_float = tf.realdiv(tf.cast(obs_t_ph, tf.float32), 255.0, name='obs_t_float')

        pred_q = q_func(obs_t_float, num_actions, scope="q_func", reuse=False)
        pred_ac = tf.argmax(pred_q, axis=1, name="pred_ac")

    # placeholder for current action
    act_t_ph = tf.placeholder(tf.int32, [None], name="act_t_ph")

    # placeholder for current reward
    rew_t_ph = tf.placeholder(tf.float32, [None], name="rew_t_ph")

    with tf.variable_scope("obs_tp1_ph"):
        # placeholder for next observation (or state)
        obs_tp1_ph = tf.placeholder(tf.uint8, [None] + list(input_shape), name="obs_tp1_ph")
        obs_tp1_float = tf.cast(obs_tp1_ph, tf.float32) / 255.0

    # placeholder for end of episode mask
    done_mask_ph = tf.placeholder(tf.float32, [None], name="done_mask_ph")

    # placeholder for the time the option took
    opt_steps = tf.placeholder(tf.float32, [None], name="opt_steps")

    with tf.variable_scope("pred_q_a"):
        pred_q_a = tf.reduce_sum(pred_q * tf.one_hot(act_t_ph, depth=num_actions), axis=1, name='pred_q_a')

    target_q = q_func(obs_tp1_float, num_actions, scope="target_q_func", reuse=False)

    with tf.variable_scope("target_q_a"):
        target_q_a = rew_t_ph + (1 - done_mask_ph) * tf.pow(gamma, opt_steps) * tf.reduce_max(target_q, axis=1)

    with tf.variable_scope("Compute_bellman_error"):
        total_error = tf.reduce_sum(huber_loss(pred_q_a - tf.stop_gradient(target_q_a)), name='total_error')

    with tf.variable_scope("Hold_the_var"):
        # Hold all of the variables of the Q-function network and target network, respectively.
        q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_name + '/q_func')
        target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_q_func')

    # construct optimization op (with gradient clipping)
    learning_rate = tf.placeholder(tf.float32, (), name="learning_rate")
    with tf.variable_scope("Optimizer"):
        optimizer = optimizer_spec.constructor(learning_rate=learning_rate, **optimizer_spec.kwargs)
        train_fn = minimize_and_clip(optimizer, total_error,
                                     var_list=q_func_vars, clip_val=grad_norm_clipping)

    # update_target_fn will be called periodically to copy Q network to target Q network
    update_target_fn = []
    for var, var_target in zip(sorted(q_func_vars, key=lambda v: v.name),
                               sorted(target_q_func_vars, key=lambda v: v.name)):
        update_target_fn.append(var_target.assign(var))

    with tf.variable_scope("Update_target_fn"):
        update_target_fn = tf.group(*update_target_fn, name='update_target_fn')

    # construct the replay buffer with options
    replay_buffer = ReplayBufferOptions(replay_buffer_size, frame_history_len)

    ###############
    # RUN ENV     #
    ###############
    model_initialized = False
    num_param_updates = 0
    mean_episode_reward = -float('nan')
    best_mean_episode_reward = -float('inf')
    last_obs = env.reset()
    LOG_EVERY_N_STEPS = 500

    # saver = tf.train.Saver()
    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_name))

    for t in itertools.count():
        ### 1. Check stopping criterion
        if stopping_criterion is not None and stopping_criterion(env, t):
            break

        ### 2. Step the env and store the transition

        # Store the latest observation that was recorded from the simulator.
        idx = replay_buffer.store_frame(last_obs)

        # Epsilon greedy exploration
        if not model_initialized or random.random() < exploration.value(t):
            action = random.randint(0, num_actions - 1)
        else:
            obs = replay_buffer.encode_recent_observation()
            action = session.run(pred_ac, {obs_t_ph: [obs]})[0]

        if action < env.action_space.n:
            next_obs, reward, done, info = env.step(action)
            opt_steps_n = 1
        else:
            # here the execution of the option
            next_obs, reward, done, opt_steps_n, info = options[action - env.action_space.n].step(env, isoption=True)
            env._episode_length += 1

        # Store the outcome
        replay_buffer.store_effect(idx, action, reward, done, opt_steps_n)
        last_obs = env.reset() if done else next_obs

        ### 3. Perform experience replay and train the network.

        if (t > learning_starts and t % learning_freq == 0 and
                replay_buffer.can_sample(batch_size)):

            # 3.a sample a batch of transitions
            obs_batch, act_batch, rew_batch, next_obs_batch, done_batch, opt_steps_batch = replay_buffer.sample(
                batch_size)

            # 3.b initialize the model if haven't
            if not model_initialized:
                initialize_interdependent_variables(session, tf.global_variables(), {
                    obs_t_ph: obs_batch,
                    obs_tp1_ph: next_obs_batch,
                })
                session.run(update_target_fn)
                model_initialized = True

            # 3.c train the model
            _, error = session.run([train_fn, total_error], {
                obs_t_ph: obs_batch,
                act_t_ph: act_batch,
                rew_t_ph: rew_batch,
                obs_tp1_ph: next_obs_batch,
                opt_steps: opt_steps_batch,
                done_mask_ph: done_batch,
                learning_rate: optimizer_spec.lr_schedule.value(t)
            })

            # 3.d periodically update the target network
            if t % target_update_freq == 0:
                session.run(update_target_fn)
                num_param_updates += 1

        ### 4. Log progress
        episode_rewards = env.get_episode_rewards()
        episode_lengths = env.get_episode_lengths()

        if len(episode_rewards) > 0 and len(episode_rewards) <= 50:
            mean_episode_reward = np.mean(episode_rewards)
            mean_episode_length = np.mean(episode_lengths)

            max_episode_reward = np.max(episode_rewards)
            min_episode_length = np.min(episode_lengths)

            min_episode_reward = np.min(episode_rewards)
            max_episode_length = np.max(episode_lengths)

        elif len(episode_rewards) > 50:
            mean_episode_reward = np.mean(episode_rewards[-50:])
            mean_episode_length = np.mean(episode_lengths[-50:])

            max_episode_reward = np.max(episode_rewards[-50:])
            min_episode_length = np.min(episode_lengths[-50:])

            min_episode_reward = np.min(episode_rewards[-50:])
            max_episode_length = np.max(episode_lengths[-50:])

        best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)

        if t % LOG_EVERY_N_STEPS == 0 and model_initialized:
            print("Timestep %d" % (t,))
            print("mean reward (50 episodes) %f" % mean_episode_reward)
            print("mean length (50 episodes) %f" % mean_episode_length)
            print("max_episode_reward (50 episodes) %f" % max_episode_reward)
            print("min_episode_length (50 episodes) %f" % min_episode_length)
            print("min_episode_reward (50 episodes) %f" % min_episode_reward)
            print("max_episode_length (50 episodes) %f" % max_episode_length)
            print("best mean reward %f" % best_mean_episode_reward)
            print("episodes %d" % len(episode_rewards))
            print("exploration %f" % exploration.value(t))
            print("learning_rate %f" % optimizer_spec.lr_schedule.value(t))
            print("\n")
            sys.stdout.flush()

    meta_graph_def = tf.train.export_meta_graph(filename=scope_name + '/dqn_graph.ckpt.meta', export_scope=scope_name)
    save_path = saver.save(session, scope_name + '/dqn_graph.ckpt', write_meta_graph=False)
    print("Model saved in path: %s" % save_path)


