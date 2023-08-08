# -*- coding: UTF-8 -*-
# This Notebook explains the Advantage Actor Critic implementation.
# The repository link
# Acknowledgements
# This implementation was inspired by 2 repositories:
# OpenAI Baselines A2C
# Alexandre Borghi retro_contest_agent

# @Thomas Simonini
# https://github.com/simoninithomas/Deep_reinforcement_learning_Course/tree/master/A2C%20with%20Sonic%20the%20Hedgehog

"""
How to use it?
First you need to follow Step 1 (Download Sonic the Hedgehog)

Watch our agent playing
Modify the environment: change env.make_train_3with the env you want in env= DummyVecEnv([env.make_train_3])) in play.py
See the agent playing: run python play.py

Continue to train ️
There is a big risk of overfitting
Run python agent.py
"""

# Step 1: Download Sonic the Hedgehog
"""
The first step is to download the game, to make it works on retro you need to buy it legally on Steam
Then follow the Quickstart part of this website
"""

# Step 2: Build all elements we need for our environement in sonic_env.py
"""
PreprocessFrame(gym.ObservationWrapper): in this class we will preprocess our environment
    Set frame to gray 
    Resize the frame to 96x96x1 
ActionsDiscretizer(gym.ActionWrapper): in this class we limit the possibles actions in our environment (make it discrete)

In fact you'll see that for each action in actions: Create an array of 12 False (12 = nb of buttons) For each button in
action: (for instance ['LEFT']) we need to make that left button index = True Then the button index = LEFT = True

--> In fact at the end we will have an array where each array is an action and each elements True of this array are the
buttons clicked. For instance LEFT action = [F, F, F, F, F, F, T, F, F, F, F, F] 
    RewardScaler(gym.RewardWrapper): We scale the rewards to reasonable scale (useful in PPO but also in A2C). 
    AllowBacktracking(gym.Wrapper): Allow the agent to go backward without being discourage (avoid our agent to be stuck
on a wall during the game). 
    make_env(env_idx) : Build an environement (and stack 4 frames together using FrameStack 

The idea is that we'll build multiple instances of the environment, different environments each times (different level)
to avoid overfitting and helping our agent to generalize better at playing sonic 
To handle these multiple environements we'll use SubprocVecEnv that creates a vector of n environments to run them simultaneously.
"""

# Step 3: Build the A2C architecture in architecture.py
"""
from baselines.common.distributions import make_pdtype: This function selects the probability distribution over actions 

First, we create two functions that will help us to avoid to call conv and fc each time.
    conv: function to create a convolutional layer.
    fc: function to create a fully connected layer. 

Then, we create A2CPolicy, the object that contains the architecture
3 CNN for spatial dependencies
Temporal dependencies is handle by stacking frames
(Something funny nobody use LSTM in OpenAI Retro contest)
1 common FC
1 FC for policy
1 FC for value 

self.pdtype = make_pdtype(action_space): Based on the action space, will select what probability distribution typewe will use to distribute action in our stochastic policy (in our case DiagGaussianPdType aka Diagonal Gaussian, multivariate normal distribution 

self.pdtype.pdfromlatent : return a returns a probability distribution over actions (self.pd) and our pi logits (self.pi). 

We create also 3 useful functions in A2CPolicy
    def step(state_in, *_args, **_kwargs): Function use to take a step returns action to take and V(s)
    def value(state_in, *_args, **_kwargs): Function that calculates only the V(s)
    def select_action(state_in, *_args, **_kwargs): Function that output only the action to take
"""

# Step 4: Build the Model in model.py
"""
We use Model object to: init:
    policy(sess, ob_space, action_space, nenvs, 1, reuse=False): Creates the step_model (used for sampling)
    policy(sess, ob_space, action_space, nenvs*nsteps, nsteps, reuse=True): Creates the train_model (used for training)
    def train(states_in, actions, returns, values, lr): Make the training part (calculate advantage and feedforward and retropropagation of gradients)

save/load():
    def save(save_path):Save the weights.
    def load(load_path):Load the weights
"""

# Step 5: Build the Runner in model.py
"""
Runner will be used to make a mini batch of experiences
    Each environement send 1 timestep (4 frames stacked) (self.obs)
    This goes through step_model
        Returns actions, values.
    Append mb_obs, mb_actions, mb_values, mb_dones.
    Take actions in environments and watch the consequences
        return obs, rewards, dones
    We need to calculate advantage to do that we use General Advantage Estimation
"""

# Step 6: Build the learn function in model.py
"""
The learn function can be seen as the gathering of all the logic of our A2C
    Instantiate the model object (that creates step_model and train_model)
    Instantiate the runner object
    Train always in two phases:
        Run to get a batch of experiences
        Train that batch of experiences 

We use explained_variance which is a really important parameter: 

ev = 1 - Variance[y - ypredicted] / Variance [y] 

In fact this calculates if value function is a good predictor of the returns or if it's just worse than predicting 
nothing. ev=0 => might as well have predicted zero ev worse than just predicting zero so you're overfitting (need to
tune some hyperparameters) ev=1 => perfect prediction
--> The goal is that ev goes closer and closer to 1.
"""

# Step 7: Build the play function in model.py
"""
This function will be use to play an environment using the trained model.
"""

# Step 8: Build the agent.py
"""
config.gpu_options.allow_growth = True : This creates a GPU session
model.learn(policy=policies.A2CPolicy,env=SubprocVecEnv([env.make_train_0,
                                                        env.make_train_1, 
                                                        env.make_train_2, 
                                                        env.make_train_3, 
                                                        env.make_train_4, 
                                                        env.make_train_5,
                                                        env.make_train_6,
                                                        env.make_train_7,
                                                        env.make_train_8,
                                                        env.make_train_9,
                                                        env.make_train_10,
                                                        env.make_train_11,
                                                        env.make_train_12 ]), nsteps=2048, # Steps per environment
            total_timesteps=10000000,gamma=0.99,lam = 0.95,vf_coef=0.5,ent_coef=0.01,lr = 2e-4,
            max_grad_norm = 0.5, log_interval = 10) : Here we just call the learn function that contains all the 
                                                        elements needed to train our A2C agent
"""
