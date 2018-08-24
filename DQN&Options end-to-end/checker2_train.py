import sys
import gym.spaces
import itertools
import os
import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.layers as layers
from utils.dqn_utils import *

import datetime


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


def train(conv_net,
          checker_mlp,
          session,
          epochs,
          X_train,
          y_train,
          batch_size=16,
          exp_dir=None):
    input_shape = X_train[0].shape

    obs_t_ph = tf.placeholder(tf.uint8, [None] + list(input_shape), name="obs_t_ph")
    obs_t_float = tf.realdiv(tf.cast(obs_t_ph, tf.float32), 255.0, name='obs_t_float')

    target = tf.placeholder(tf.float32, [None, 2], name="target")

    conv = conv_net(obs_t_float, scope="convolution", reuse=False)
    pred_y = checker_mlp(conv, 2, scope="opt2_checker", reuse=False)

    with tf.variable_scope("Compute_loss"):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred_y, labels=target)
        cost = tf.reduce_mean(cross_entropy)

    with tf.variable_scope("Compute_loss"):
        y_target = tf.argmax(target, axis=1, name="y_target")
        y_pred = tf.argmax(pred_y, axis=1, name="y_pred")
        correct_prediction = tf.equal(y_pred, y_target)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")

    with tf.variable_scope("Hold_the_var"):
        # Hold all of the variables of the Q-function network and target network, respectively.
        vars1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="opt2_checker")

    optimizer = tf.train.AdamOptimizer().minimize(cost, var_list=vars1)

    saver1 = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="convolution"))
    saver2 = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="opt2_checker"))

    session.run(tf.global_variables_initializer())
    saver1.restore(session, '../experiments/DQN&Options end-to-end/experiment task0/saved_model/conv_graph.ckpt')

    iterations = int(len(X_train) / batch_size)

    for epoch in range(epochs):

        for batch in range(iterations):
            # 3.a sample a batch of transitions
            idx0 = 0 + batch * batch_size
            idx1 = max(batch_size + batch * batch_size, len(X_train))

            obs_batch, target_batch = X_train[idx0:idx1], y_train[idx0:idx1]

            # 3.c train the model
            _, loss, train_accuracy = session.run([optimizer, cost, accuracy], {
                obs_t_ph: obs_batch,
                target: target_batch
            })

            print("epoch {0}: , loss: {1} , accuracy: {2}\n".format(epoch, loss, train_accuracy))
            sys.stdout.flush()

    # meta_graph_def = tf.train.export_meta_graph(filename=exp_dir + '/saved_model/graph.ckpt.meta')
    save_path2 = saver2.save(session, exp_dir + '/saved_model/graph.ckpt', write_meta_graph=False)


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
    # create a new folder for this
    os.chdir('../experiments/DQN&Options end-to-end/')
    dir_name = "experiment " + str(datetime.datetime.now())[:-10]
    createFolder(dir_name)
    os.chdir('../../DQN&Options end-to-end/')

    session = get_session()

    X_dataset = np.load('../experiments/DQN&Options end-to-end/experiment task2/obs_dataset.npy')
    y_datatset = np.load('../experiments/DQN&Options end-to-end/experiment task2/done_dataset.npy')

    y_train = []
    for i in y_datatset:
        if i == 1:
            y_train.append([0, 1])
        else:
            y_train.append([1, 0])

    train(conv_model,
          mlp_model,
          session=session,
          epochs=600,
          X_train=X_dataset,
          y_train=y_train,
          # X_test=X_test,
          # y_test=y_test,
          batch_size=16,
          exp_dir='../experiments/DQN&Options end-to-end/' + dir_name
          )


if __name__ == "__main__":
    main()
