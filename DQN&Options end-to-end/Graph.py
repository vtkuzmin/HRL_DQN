import os
import sys

import argparse
import gym
from gym import wrappers
import os.path as osp
import random
import numpy as np
import itertools
import tensorflow as tf
import tensorflow.contrib.layers as layers

nb_dir = os.path.split(os.getcwd())[0]
if nb_dir not in sys.path:
    sys.path.append(nb_dir)

from environments.arm_env_dqn import ArmEnvDQN

def get_session():
    tf.reset_default_graph()
    session = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    session = tf.Session()
    return session

env = ArmEnvDQN(episode_max_length=200,
                 size_x=6,
                 size_y=4,
                 cubes_cnt=4,
                 scaling_coeff=3,
                 action_minus_reward=-1,
                 finish_reward=200,
                 tower_target_size=4)


def conv_model(input_data, scope, flatten=True, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        out = input_data
        out = layers.convolution2d(out, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
        out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
        out = layers.convolution2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
        if flatten:
            out = layers.flatten(out)
        return out


def mlp_model(input_data, output_len, scope, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        out = input_data
        out = layers.fully_connected(out, num_outputs=256, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=output_len, activation_fn=None)
        return out


with tf.Session() as session:
    frame_history_len = 1
    num_options = 2

    if len(env.observation_space.shape) == 1:
        # This means we are running on low-dimensional observations (e.g. RAM)
        input_shape = env.observation_space.shape
    else:
        img_h, img_w, img_c = env.observation_space.shape
        input_shape = (img_h, img_w, frame_history_len * img_c)  # size_x, size_y,

    num_actions = env.action_space.n

    # INPUT DATA: previous action and image
    prev_action = tf.placeholder(tf.float32, [None, num_options + 1], name="prev_action")

    with tf.variable_scope('input_image'):
        # placeholder for current observation (or state)
        obs_t_ph = tf.placeholder(tf.uint8, [None] + list(input_shape), name="obs_t_ph")
        # casting to float on GPU ensures lower data transfer times.
        obs_t_float = tf.realdiv(tf.cast(obs_t_ph, tf.float32), 255.0, name='obs_t_float')

    # CONVOLUTION
    convolution = conv_model(obs_t_float, scope="convolution", reuse=False)

    # MANAGER
    with tf.variable_scope("manager"):
        manager = mlp_model(convolution, num_options + 1, scope="manager", reuse=False)
        manager_pred_ac = tf.argmax(manager, axis=1, name="manager_pred_ac")
        manager_one_hot = tf.one_hot(manager_pred_ac, depth=num_options + 1, name="manager_one_hot")

    # NETs to check if the option is terminated
    options_checkers = [mlp_model(convolution, 1, scope='opt{0}_checker'.format(i + 1), reuse=False)
                        for i in range(num_options)]

    with tf.variable_scope("check_option"):
        options_check = tf.concat(options_checkers, 1, name="options_check")
        cond = tf.cast(tf.reduce_sum(tf.multiply(options_check, prev_action[:, 1:]), axis=1), tf.bool, name='cond')

    # SELECT on whether the option terminated
    with tf.variable_scope("subselect"):
        one_hot0 = tf.where(cond, manager_one_hot, prev_action, name="select1")

    # SELECT on if ot was option or not
    with tf.variable_scope("select_task"):
        one_hot = tf.where(tf.cast(prev_action[:, 0], tf.bool), manager_one_hot, one_hot0, name="select2")

    tasks = [mlp_model(convolution, num_actions, scope='task{0}'.format(i), reuse=False)
             for i in range(num_options + 1)]

    with tf.variable_scope("action"):
        action = tf.boolean_mask(tasks, tf.cast(one_hot, tf.bool), name="get_task")

    summary_writer = tf.summary.FileWriter("graph_logs", graph=tf.get_default_graph())