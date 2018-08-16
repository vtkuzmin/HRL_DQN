import os
import sys
import itertools
from utils.dqn_utils import *

nb_dir = os.path.split(os.getcwd())[0]
if nb_dir not in sys.path:
    sys.path.append(nb_dir)


# import gym.spaces
# import numpy as np
# import random
# import tensorflow as tf
# import tensorflow.contrib.layers as layers
# from collections import namedtuple


class option(object):
    def __init__(self, env, stop_condition, path, import_scope, name=None, max_t=25):

        self.max_t = max_t
        self.g = tf.Graph()
        self.sess = tf.Session(graph=self.g)
        self.stop_condition = stop_condition

        with self.g.as_default():
            self.saver = tf.train.import_meta_graph(path + ".meta", import_scope=import_scope)
            self.saver.restore(self.sess, path)

        self.obs_t_float = self.g.get_tensor_by_name(import_scope + "/obs_t_float:0")
        self.pred_ac = self.g.get_tensor_by_name(import_scope + "/pred_ac:0")

    def encode_observation(self, frame):
        img_h, img_w = frame.shape[1], frame.shape[2]
        return frame.transpose(1, 2, 0, 3).reshape(img_h, img_w, -1)

    def step(self, env):
        """Executes the option"""
        opt_reward = env._action_minus_reward
        opt_length = 0.0
        last_obs = env.get_evidence_for_image_render()
        #         env.render()

        for t in itertools.count():
            obs = self.encode_observation(np.array([last_obs]))
            action = self.sess.run(self.pred_ac, {self.obs_t_float: [obs]})[0]

            next_obs, reward, done, info = env.step(action)

            opt_reward += reward
            opt_length += 1

            #             env.render()

            if done or self.stop_condition(env) or t >= self.max_t:
                #                 env.render()
                break

            last_obs = next_obs

        return next_obs, opt_reward, done, opt_length, info

