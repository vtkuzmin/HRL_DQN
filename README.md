# HRL_DQN

Combination of the Options RL framework and DQNs.
The project contains of the following parts:

## DQN

Implementation of the plain DQN algorithm

## DQN with Options

Using DQN to represent every option and a separate DQN over options

## DQN&Options end-to-end

The united architecture

## environments

The testing environment used in the experiments

## utils

Utils needed for plotting and DQN implementation.

The file  dqn_utils.py contains the following:

	    huber_loss
	    Constant Piecewise Linear Schedules
	    compute_exponential_averages
	    minimize_and_clip
	    initialize_interdependent_variables
	    ReplayBuffer
	    ReplayBufferOptions

/environments contains the testing environment (arm_env_dqn) and the environments for training the options (arm_env_dqn_go_down and arm_env_dqn_lift_cube)

/utils contains the file  dqn_utils.py with needed utils to implement the DQN algorithm (with or without options)




To run the plain DQN go for ../DQN/run_dqn.py where the environment, network and learning parameters can be modified;
The learning process for the plain DQN is implemented in the file ../DQN/dqn.py

To run the DQN algorithms with options, firstly the options must be trained running the files ../DQN with Options/train_option1(go down).py and ../DQN with Options/train_option2(lift cube).py
After that to train the DQN over options run the file ../DQN with Options/train_over_options.py
The learning process for the DQN over options is implemented in the file ../DQN with Options/dqn_with_options.py and the file ../DQN with Options/option_class.py contains the implementation of the option work