# HRL_DQN
Combination of the Options RL framework and DQNs

/environments contains the testing environment (arm_env_dqn) and the environments for training the options (arm_env_dqn_go_down and arm_env_dqn_lift_cube)

/utils contains the file  dqn_utils.py with needed utils to implement the DQN algorithm (with or without options)

    dqn_utils.py

	    huber_loss
	    Constant Piecewise Linear Schedules
	    compute_exponential_averages
	    minimize_and_clip
	    initialize_interdependent_variables
	    ReplayBuffer
	    ReplayBufferOptions