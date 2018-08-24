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
                size_x=4,
                size_y=3,
                cubes_cnt=3,
                scaling_coeff=3,
                action_minus_reward=-1,
                finish_reward=200,
                tower_target_size=3)


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
    options_checkers = [tf.argmax(mlp_model(convolution, 2, scope='opt{0}_checker'.format(i + 1), reuse=False), axis=1)
                        for i in range(num_options)]

    for i in range(len(options_checkers)):
        options_checkers[i] = tf.reshape(options_checkers[i], (tf.shape(options_checkers[i])[0], 1))

    with tf.variable_scope("check_option"):
        options_check = tf.cast(tf.concat(options_checkers, 1, name="options_check"), tf.float32)
        cond = tf.cast(tf.reduce_sum(tf.multiply(options_check, prev_action[:, 1:]), axis=1), tf.bool, name='cond')
    # cond = tf.cast(opt_check2, tf.bool, name = 'cond')

    # SELECT on whether the option terminated
    with tf.variable_scope("subselect"):
        one_hot0 = tf.where(cond, manager_one_hot, prev_action, name="select1")

    # SELECT on if it was option or not
    with tf.variable_scope("select_task"):
        one_hot = tf.where(tf.cast(prev_action[:, 0], tf.bool), manager_one_hot, one_hot0, name="select2")

    tasks = [mlp_model(convolution, num_actions, scope='task{0}'.format(i), reuse=False)
             for i in range(num_options + 1)]

    with tf.variable_scope("action"):
        pred_q = tf.boolean_mask(tf.transpose(tasks, perm=[1, 0, 2]), tf.cast(one_hot, tf.bool), name="get_task")
        pred_ac = tf.argmax(pred_q, axis=1, name="pred_ac")

    session.run(tf.global_variables_initializer())

    saver1 = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="convolution"))
    saver2 = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="task0"))
    saver3 = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="task1"))
    saver4 = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="task2"))
    saver5 = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="opt1_checker"))
    saver6 = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="opt2_checker"))

    saver1.restore(session, '../experiments/DQN&Options end-to-end/experiment task0/saved_model/conv_graph.ckpt')
    saver2.restore(session, '../experiments/DQN&Options end-to-end/experiment task0/saved_model/flat_graph.ckpt')
    saver3.restore(session, '../experiments/DQN&Options end-to-end/experiment task1/saved_model/graph.ckpt')
    saver4.restore(session, '../experiments/DQN&Options end-to-end/experiment task2/saved_model/graph.ckpt')
    saver5.restore(session, '../experiments/DQN&Options end-to-end/experiment checker1/saved_model/graph.ckpt')
    saver6.restore(session, '../experiments/DQN&Options end-to-end/experiment checker2/saved_model/graph.ckpt')

    env.step(3)
    env.step(3)
    env.render()

    # print(session.run(manager_one_hot,
    #                   {obs_t_ph: [env.get_evidence_for_image_render(), env.get_evidence_for_image_render()],
    #                    prev_action: [[1, 0, 0], [1, 0, 0]]}))
    print(session.run(options_checkers,
                      {obs_t_ph: [env.get_evidence_for_image_render()],
                       prev_action: [[1, 0, 0]]}))
    print(session.run(options_check,
                      {obs_t_ph: [env.get_evidence_for_image_render(), env.get_evidence_for_image_render()],
                       prev_action: [[1, 0, 0], [1, 0, 0]]}))
    # #     print(session.run(opt_check2, {obs_t_ph: [env.get_evidence_for_image_render(), env.get_evidence_for_image_render()], prev_action: [[1,0,0], [1,0,0]]}))
    print(session.run(cond, {obs_t_ph: [env.get_evidence_for_image_render(), env.get_evidence_for_image_render()],
                             prev_action: [[1, 0, 0], [1, 0, 0]]}))
    # print(session.run(one_hot0, {obs_t_ph: [env.get_evidence_for_image_render(), env.get_evidence_for_image_render()],
    #                              prev_action: [[1, 0, 0], [1, 0, 0]]}))
    # print(session.run(one_hot, {obs_t_ph: [env.get_evidence_for_image_render(), env.get_evidence_for_image_render()],
    #                             prev_action: [[1, 0, 0], [1, 0, 0]]}))
    # print(session.run(tasks, {obs_t_ph: [env.get_evidence_for_image_render(), env.get_evidence_for_image_render()],
    #                           prev_action: [[1, 0, 0], [1, 0, 0]]}))
    # print(len(session.run(tasks, {obs_t_ph: [env.get_evidence_for_image_render(), env.get_evidence_for_image_render()],
    #                               prev_action: [[1, 0, 0], [1, 0, 0]]})))
    # print(session.run(tf.cast(one_hot, tf.bool),
    #                   {obs_t_ph: [env.get_evidence_for_image_render(), env.get_evidence_for_image_render()],
    #                    prev_action: [[1, 0, 0], [1, 0, 0]]}))
    # print(session.run(pred_ac, {obs_t_ph: [env.get_evidence_for_image_render(), env.get_evidence_for_image_render()],
    #                             prev_action: [[1, 0, 0], [1, 0, 0]]}))

    # summary_writer = tf.summary.FileWriter("graph_logs", graph=tf.get_default_graph())
