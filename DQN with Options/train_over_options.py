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
import utils.plotting as plotting

# import dqn_with_options_v2 as dqn
import dqn_with_options as dqn
from utils.dqn_utils import *
#from atari_wrappers import *
#from environments.arm_env.arm_env import ArmEnv
from environments.arm_env_dqn_with_options import ArmEnvDQN
from option_class import option


def arm_model(img_in, num_actions, scope, reuse=False):
    # as described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
    with tf.variable_scope(scope, reuse=reuse):
        out = img_in
        with tf.variable_scope("convnet"):
            # original architecture
            out = layers.convolution2d(out, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
        out = layers.flatten(out)
        with tf.variable_scope("action_value"):
            out = layers.fully_connected(out, num_outputs=256, activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)

        return out


def arm_learn(env, options, session, num_timesteps):
    # This is just a rough estimate
    num_iterations = float(num_timesteps) / 4.0

    lr_multiplier = 1.0
    lr_schedule = PiecewiseSchedule([
        (0, 1e-4 * lr_multiplier),
        (num_iterations / 10, 1e-4 * lr_multiplier),
        (num_iterations / 2, 5e-5 * lr_multiplier),
    ],
        outside_value=5e-5 * lr_multiplier)
    optimizer = dqn.OptimizerSpec(
        constructor=tf.train.AdamOptimizer,
        kwargs=dict(epsilon=1e-4),
        lr_schedule=lr_schedule
    )

    def stopping_criterion(env, t):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return t >= num_timesteps

    exploration_schedule = PiecewiseSchedule(
        [
            (0, 1.0),
            (100000, 0.3),
            (200000, 0.1),
            (500000, 0.01),
        ], outside_value=0.01
    )

    dqn.learn(
        env,
        options=options,
        q_func=arm_model,
        optimizer_spec=optimizer,
        session=session,
        scope_name='over_options_8_6_6',
        exploration=exploration_schedule,
        stopping_criterion=stopping_criterion,
        replay_buffer_size=1000000,
        batch_size=32,
        gamma=0.99,
        learning_starts=5000,
        learning_freq=1,
        frame_history_len=1,
        target_update_freq=500,
        grad_norm_clipping=10
    )

    ep_rew = env.get_episode_rewards()
    ep_len = env.get_episode_lengths()
    env.close()
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
    #     tf_config = tf.ConfigProto(
    #         inter_op_parallelism_threads=1,
    #         intra_op_parallelism_threads=1)
    session = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    print("AVAILABLE GPUS: ", get_available_gpus())
    session = tf.Session()
    return session


def main():
    # Get Atari games.
    # benchmark = gym.benchmark_spec('Atari40M')
    #
    # # Change the index to select a different game.
    # task = benchmark.tasks[3]
    #
    # # Run training
    #     seed = 0  # Use a seed of zero (you may want to randomize the seed!)
    #     set_global_seeds(seed)
    # env = get_env(task, seed)
    env = ArmEnvDQN(episode_max_length=300,
                    size_x=8,
                    size_y=6,
                    cubes_cnt=6,
                    scaling_coeff=3,
                    action_minus_reward=-1,
                    finish_reward=1000,
                    tower_target_size=5)
    session = get_session()

    def stop_cond1(env):
        if env._arm_x + 1 < env._size_x:
            if env._grid[env._arm_x + 1, env._arm_y] == 1 and env._arm_x + 2 >= env._size_x:
                return True
            if env._grid[env._arm_x + 1, env._arm_y] == 1 and env._arm_x + 2 < env._size_x:
                if env._grid[env._arm_x + 2, env._arm_y] == 1:
                    return True
        else:
            return True
        return False

    def stop_cond2(env):
        if env._arm_x == 0 and env._grid[1, env._arm_y] == 1 and env._grid[2, env._arm_y] == 0:
            return True
        return False


        # initialize options

    #     option(env, stop_cond2, path = "option2_v2_8_6_6/dqn_graph.ckpt", import_scope = "option2_v2_8_6_6")
    #     option(env, stop_cond1, path = "option1_8_6_6/dqn_graph.ckpt", import_scope = "option1_8_6_6"),
    options = [option(env, stop_cond1, path="option1_8_6_6/dqn_graph.ckpt", import_scope="option1_8_6_6"),
               option(env, stop_cond2, path="option2_8_6_6/dqn_graph.ckpt", import_scope="option2_8_6_6")]

    ep_rew, ep_len = arm_learn(env, options, session, num_timesteps=1500000)

    thefile = open('ep_rew_8_6_6.txt', 'w')
    for item in ep_rew:
        thefile.write("%s\n" % item)

    thefile2 = open('ep_len_8_6_6.txt', 'w')
    for item in ep_len:
        thefile2.write("%s\n" % item)

    stats = plotting.EpisodeStats(
        episode_lengths=ep_len,
        episode_rewards=ep_rew)
    plotting.plot_episode_stats(stats)


#     tf.summary.FileWriter("logs", tf.get_default_graph()).close()

if __name__ == "__main__":
    main()