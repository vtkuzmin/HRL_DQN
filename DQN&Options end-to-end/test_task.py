import argparse
import gym
from gym import wrappers
import os.path as osp
import random
import numpy as np
import itertools
import tensorflow as tf
import tensorflow.contrib.layers as layers

import os
import sys
nb_dir = os.path.split(os.getcwd())[0]
if nb_dir not in sys.path:
    sys.path.append(nb_dir)
# from utils import plotting

# from DQN import dqn
# from DQN.dqn_utils import *
from environments.arm_env_dqn import ArmEnvDQN
from environments.arm_env_dqn_go_down import ArmEnvDQN_1
from environments.arm_env_dqn_lift_cube import ArmEnvDQN_2

def encode_observation(frame):
    img_h, img_w = frame.shape[1], frame.shape[2]
    return frame.transpose(1, 2, 0, 3).reshape(img_h, img_w, -1)


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



def main():
    env1 = ArmEnvDQN(episode_max_length=200,
                    size_x=4,
                    size_y=3,
                    cubes_cnt=3,
                    scaling_coeff=3,
                    action_minus_reward=-1,
                    finish_reward=200,
                    tower_target_size=3)

    env2 = ArmEnvDQN_1(episode_max_length=200,
                    size_x=4,
                    size_y=3,
                    cubes_cnt=3,
                    scaling_coeff=3,
                    action_minus_reward=-1,
                    finish_reward=200,
                    tower_target_size=3)

    env3 = ArmEnvDQN_2(episode_max_length=200,
                       size_x=4,
                       size_y=3,
                       cubes_cnt=3,
                       scaling_coeff=3,
                       action_minus_reward=-1,
                       finish_reward=200,
                       tower_target_size=3)
    # print(env.reset())

    # First let's load meta graph and restore weights
    # saver = tf.train.import_meta_graph('option_lift_cube.ckpt.meta')

    #     saver2 = tf.train.import_meta_graph('/tmp/option_lift_cube.ckpt.meta')
    #     saver.restore(session, tf.train.latest_checkpoint('./'))
    frame_history_len = 1
    img_h, img_w, img_c = env1.observation_space.shape
    input_shape = (img_h, img_w, frame_history_len * img_c)  # size_x, size_y,
    num_actions = env1.action_space.n

    #     # placeholder for current observation (or state)
    #     obs_t_ph = tf.placeholder(tf.uint8, [None] + list(input_shape))
    #     # casting to float on GPU ensures lower data transfer times.
    #     obs_t_float = tf.cast(obs_t_ph, tf.float32) / 255.0



    #     pred_q = q_func(obs_t_float, num_actions, scope="q_func", reuse=False)
    #     pred_ac = tf.argmax(pred_q, axis=1)
    # graph = tf.get_default_graph()

    obs_t_ph = tf.placeholder(tf.uint8, [None] + list(input_shape), name="obs_t_ph")
    # casting to float on GPU ensures lower data transfer times.
    obs_t_float = tf.realdiv(tf.cast(obs_t_ph, tf.float32), 255.0, name='obs_t_float')



    conv = conv_model(obs_t_float, scope="convolution", reuse=False)
    pred_q = mlp_model(conv, num_actions, scope="task2", reuse=False)
    pred_ac = tf.argmax(pred_q, axis=1, name="pred_ac")

    #     obs_t_float2 = graph.get_tensor_by_name("obs_t_ph_lift:0")

    ## How to access saved operation
    #     pred_ac2 = graph.get_tensor_by_name("pred_ac_lift:0")

    episode_reward = 0
    episode_length = 0
    last_obs = env3.reset()

    session = tf.Session()

    #     saver2.restore(session, "/tmp/option_lift_cube.ckpt")
    saver1 = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="convolution"))
    saver2 = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="task2"))

    saver1.restore(session, '../experiments/DQN&Options end-to-end/experiment task0/saved_model/conv_graph.ckpt')
    saver2.restore(session, '../experiments/DQN&Options end-to-end/experiment task2/saved_model/graph.ckpt')

    for t in itertools.count():

        env3.render()
        obs = encode_observation(np.array([last_obs]))
        action = session.run(pred_ac, {obs_t_float: [obs]})[0]

        next_obs, reward, done, info = env3.step(action)

        episode_reward += reward
        episode_length += 1

        if done or episode_length == 100:
            env3.render()
            break

        last_obs = next_obs
    print(episode_reward, episode_length)


#     episode_reward = 0
#     episode_length = 0
#     last_obs = env2.reset()
#     for t in itertools.count():

#         env2.render()
#         obs = encode_observation(np.array([last_obs]))
#         action = session.run(pred_ac2, {obs_t_float2: [obs]})[0]

#         next_obs, reward, done, info = env2.step(action)

#         episode_reward += reward
#         episode_length += 1

#         if done or episode_length == 500:
#             env2.render()
#             break

#         last_obs = next_obs
#     print(episode_reward, episode_length)


if __name__ == "__main__":
    main()