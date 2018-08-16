import argparse
import gym
from gym import wrappers
import os.path as osp
import random
import numpy as np
import itertools
import tensorflow as tf
import tensorflow.contrib.layers as layers
import time
import datetime

import os
import sys

nb_dir = os.path.split(os.getcwd())[0]
if nb_dir not in sys.path:
    sys.path.append(nb_dir)
import utils.plotting as plotting

from DQN import dqn
from utils.dqn_utils import *
# from atari_wrappers import *
# from environments.arm_env.arm_env import ArmEnv
from environments.arm_env_dqn_go_down import ArmEnvDQN_1


def arm_model(img_in, num_actions, scope, reuse=False):
    # as described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
    with tf.variable_scope(scope, reuse=reuse):
        out = img_in
        with tf.variable_scope("convnet"):
            # original architecture
            out = layers.convolution2d(out, num_outputs=32, kernel_size=6, stride=3, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
        out = layers.flatten(out)
        with tf.variable_scope("action_value"):
            out = layers.fully_connected(out, num_outputs=256, activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)

        return out


def arm_learn(env, session, scope_name, num_timesteps, spec_file=None, exp_dir=None):
    lr_multiplier = 1.0
    lr_schedule = PiecewiseSchedule([
        (0, 1e-4 * lr_multiplier),
        (num_timesteps / 40, 1e-4 * lr_multiplier),
        (num_timesteps / 8, 5e-5 * lr_multiplier),
    ],
        outside_value=5e-5 * lr_multiplier)

    optimizer = dqn.OptimizerSpec(
        constructor=tf.train.AdamOptimizer,
        kwargs=dict(epsilon=1e-4),
        lr_schedule=lr_schedule
    )

    def stopping_criterion(t):
        return t >= num_timesteps

    exploration_schedule = PiecewiseSchedule(
        [
            (0, 1.0),
            (num_timesteps / 20, 0.3),
            (num_timesteps / 4, 0.01),
        ], outside_value=0.01
    )

    dqn.learn(
        env,
        q_func=arm_model,
        optimizer_spec=optimizer,
        session=session,
        scope_name=scope_name,
        exploration=exploration_schedule,
        stopping_criterion=stopping_criterion,
        replay_buffer_size=1000000,
        batch_size=32,
        gamma=0.99,
        learning_starts=5000,
        learning_freq=1,
        frame_history_len=1,
        target_update_freq=200,
        grad_norm_clipping=10,
        log_every_n_steps=500,
        spec_file=spec_file,
        exp_dir=exp_dir
    )

    ep_rew = env.get_episode_rewards()
    ep_len = env.get_episode_lengths()

    return ep_rew, ep_len


def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.physical_device_desc for x in local_device_protos if x.device_type == 'GPU']


def set_global_seeds(i):
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.set_random_seed(i)
    np.random.seed(i)
    random.seed(i)


def get_session():
    tf.reset_default_graph()
    session = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    print("AVAILABLE GPUS: ", get_available_gpus())
    session = tf.Session()
    return session


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


def main():
    env = ArmEnvDQN_1(episode_max_length=100,
                      size_x=6,
                      size_y=4,
                      cubes_cnt=4,
                      scaling_coeff=3,
                      action_minus_reward=-1,
                      finish_reward=100,
                      tower_target_size=4)

    # create a new folder for this experiment
    os.chdir('../experiments/DQN with options/')
    dir_name = "experiment1/option1"  # + str(datetime.datetime.now())[:-10]
    createFolder(dir_name)
    os.chdir('../../DQN with Options/')

    f = open('../experiments/DQN with options/' + dir_name + '/specifications.txt', 'a').close()
    env.write_env_spec('../experiments/DQN with options/' + dir_name + '/specifications.txt')

    session = get_session()

    start = time.time()
    ep_rew, ep_len = arm_learn(env, session, scope_name="option1", num_timesteps=40000,
                               spec_file='../experiments/DQN with options/' + dir_name + '/specifications.txt',
                               exp_dir='../experiments/DQN with options/' + dir_name)

    end = time.time()
    print((end - start) / 60)

    stats = plotting.EpisodeStats(
        episode_lengths=ep_len,
        episode_rewards=ep_rew)
    plotting.plot_episode_stats(stats, save_fig=True,
                                fig_dir='../experiments/DQN with options/' + dir_name + '/',
                                fig_name='smoothed_')


#     tf.summary.FileWriter("option1_6_4_4_logs", tf.get_default_graph()).close()

if __name__ == "__main__":
    main()
