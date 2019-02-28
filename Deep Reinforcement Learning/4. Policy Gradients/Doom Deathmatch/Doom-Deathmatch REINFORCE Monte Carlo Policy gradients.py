# -*- coding: UTF-8 -*-
# try to survive in Doom environment by killing all monster that are spawned in the map.
# 通过杀死地图中所有的怪物来尝试在毁灭环境中生存。
# Our agent playing after 206 epochs of 1000 batch_size (2h of training with CPU), we can
# see that our agent needs much more training but he begins to have rational actions.
# 我们的代理在经过206次1000 batch_size (2h的CPU训练)的时间后运行，我们可以看来我们的经纪人需要更多的训练，但他开始有理性的行动。

# @Thomas Simonini
# https://github.com/simoninithomas/Deep_reinforcement_learning_Course/tree/master/Policy%20Gradients/Doom%20Deathmatch

# Step 1: Import the libraries
import tensorflow as tf         # Deep Learning library
import numpy as np              # Handle matrices
from vizdoom import *           # Doom Environment
import random                   # Handling random number generation
import time                     # Handling time calculation
from skimage import transform   # Help us to preprocess the frames

from collections import deque   # Ordered collection with ends
import matplotlib.pyplot as plt # Display graphs

import warnings     # This ignore all the warning messages that are normally printed during the training because of skiimage
warnings.filterwarnings('ignore')

# Step 2: Create our environment
"""
Doom environment takes:
    A configuration file that handle all the options (size of the frame, possible actions...)
    A scenario file: that generates the correct scenario (in our case basic but you're invited to try other scenarios).
Note: We have 3 possible actions turn left, turn right, shoot (attack) [[0,0,1], [1,0,0], [0,1,0]] so we don't need to do one hot encoding

environment
The purpose of this scenario is to teach the agent that killing the monsters is GOOD and when monsters kill you is BAD.
这个场景的目的是告诉代理杀死怪物是好事，当怪物杀死你是坏事。 
In addition, wasting amunition is not very good either. 此外，浪费弹药也不是很好。
Agent is rewarded only for killing monsters so he has to figure out the rest for himself. 
特工只会因为杀死怪物而得到奖励，所以他必须自己解决剩下的问题。

Map is a large circle. Player is spawned in the exact center. 5 melee-only, monsters are spawned along the wall. 
地图是一个大圆。玩家在确切的中心被衍生。只有近战时，怪物才会沿着墙壁出现。
Monsters are killed after a single shot. 怪物一枪毙命。
After dying each monster is respawned after some time. 每一个怪物死后都会复活一段时间。
Episode ends when the player dies (it's inevitable becuse of limitted ammo). 当玩家死亡时，游戏结束(这是不可避免的，因为使用有限的弹药)。

REWARDS: 
    +1 for killing a monster
    3 available buttons: turn left, turn right, shoot (attack)
    death penalty = 1
"""
"""
Here we create our environment
"""
def create_environment():
    game = DoomGame()
    # Load the correct configuration
    game.load_config("defend_the_center.cfg")
    # Load the correct scenario (in our case defend_the_center scenario)
    game.set_doom_scenario_path("defend_the_center.wad")
    game.init()
    # Here our possible actions
    # [[1,0,0],[0,1,0],[0,0,1]]
    possible_actions = np.identity(3, dtype=int).tolist()
    return game, possible_actions

game, possible_actions = create_environment()

# Step 3: Define the preprocessing functions
"""
preprocess_frame
Preprocessing is an important step, because we want to reduce the complexity of our states to reduce the computation time needed for training. 

Our steps:
    Grayscale each of our frames (because color does not add important information ). But this is already done by the config file.
    每个帧的灰度值(因为颜色不添加重要的信息)。但这已经由配置文件完成了。
    Crop the screen (in our case we remove the roof because it contains no information)
    裁剪屏幕(在我们的示例中，我们删除屋顶，因为它不包含任何信息)
    We normalize pixel values
    Finally we resize the preprocessed frame
"""
"""
    preprocess_frame:
    Take a frame.
    Resize it.
        __________________
        |                 |
        |                 |
        |                 |
        |                 |
        |_________________|
        to
        _____________
        |            |
        |            |
        |            |
        |____________|
    Normalize it.
    return preprocessed_frame
    """
def preprocess_frame(frame):
    # Greyscale frame already done in our vizdoom config
    # x = np.mean(frame,-1)
    # Crop the screen (remove the roof because it contains no information)
    # [Up: Down, Left: right]
    cropped_frame = frame[40:, :]
    # Normalize Pixel Values
    normalized_frame = cropped_frame / 255.0
    # Resize
    preprocessed_frame = transform.resize(normalized_frame, [100, 160])
    return preprocessed_frame

"""
stack_frames

Stacking frames is really important because it helps us to give have a sense of motion to our Neural Network.
    First we preprocess frame
    Then we append the frame to the deque that automatically removes the oldest frame
    Finally we build the stacked state
This is how work stack:
    For the first frame, we feed 4 frames
    At each timestep, we add the new frame to deque and then we stack them to form a new stacked frame
    在每个时间步中，我们将新框架添加到deque中，然后将它们堆叠起来，形成一个新的堆叠框架
    And so on
If we're done, we create a new stack with 4 new frames (because we are in a new episode).
"""
stack_size = 4  # We stack 4 frames
# Initialize deque with zero-images one array for each image初始化deque与零图像每个图像一个数组
stacked_frames = deque([np.zeros((100, 160), dtype=np.int) for i in range(stack_size)], maxlen=4)

def stack_frames(stacked_frames, state, is_new_episode):
    # Preprocess frame
    frame = preprocess_frame(state)
    if is_new_episode:
        # Clear our stacked_frames
        stacked_frames = deque([np.zeros((100, 160), dtype=np.int) for i in range(stack_size)], maxlen=4)
        # Because we're in a new episode, copy the same frame 4x
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        # Stack the frames
        stacked_state = np.stack(stacked_frames, axis=2)
    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(frame)
        # Build the stacked state (first dimension specifies different frames)构建堆栈状态(第一个维度指定不同的帧)
        stacked_state = np.stack(stacked_frames, axis=2)
    return stacked_state, stacked_frames

"""
discount_and_normalize_rewards
This function is important, because we are in a Monte Carlo situation. 这个函数很重要，因为我们在蒙特卡罗的情况下。
We need to discount the rewards at the end of the episode. This function takes, the reward discount it, and
then normalize them (to avoid a big variability in rewards).
我们需要在这一集结束时打折。这个函数对它进行折现，然后对其进行归一化(以避免在奖励上有很大的变化)。
"""
def discount_and_normalize_rewards(episode_rewards):
    discounted_episode_rewards = np.zeros_like(episode_rewards)
    cumulative = 0.0
    # reversed 翻转
    for i in reversed(range(len(episode_rewards))):
        cumulative = cumulative * gamma + episode_rewards[i]
        discounted_episode_rewards[i] = cumulative
    mean = np.mean(discounted_episode_rewards)
    # np.std计算矩阵标准差, axis=0计算每一列的标准差,axis=1计算每一行的标准差
    std = np.std(discounted_episode_rewards)
    discounted_episode_rewards = (discounted_episode_rewards - mean) / (std)
    return discounted_episode_rewards

# Step 4: Set up our hyperparameters
"""
In this part we'll set up our different hyperparameters. But when you implement a Neural Network by yourself you
will not implement hyperparamaters at once but progressively.
    First, you begin by defining the neural networks hyperparameters when you implement the model.
    Then, you'll add the training hyperparameters when you implement the training algorithm.
"""
## ENVIRONMENT HYPERPARAMETERS
state_size = [100,160,4]    # Our input is a stack of 4 frames hence 100x160x4 (Width, height, channels)
action_size = game.get_available_buttons_size()     # 3 possible actions: turn left, turn right, shoot
stack_size = 4      # Defines how many frames are stacked together
## TRAINING HYPERPARAMETERS
learning_rate = 0.0001
num_epochs = 1000   # Total epochs for training
batch_size = 1000   # Each 1 is a timestep (NOT AN EPISODE) # YOU CAN CHANGE TO 5000 if you have GPU
gamma = 0.99        # Discounting rate
## MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT如果您只想查看训练过的代理，请将其修改为FALSE
training = False

"""
Quick note: Policy gradient methods like reinforce are on-policy method which can not be updated from experience replay.
快速说明:政策梯度方法，如加强是对政策的方法，不能更新的经验重演。
"""

# Step 5: Create our Policy Gradient Neural Network model
"""
This is our Policy Gradient model:
    We take a stack of 4 frames as input
    It passes through 3 convnets
    Then it is flatened
    Finally it passes through 2 FC layers
    It outputs a probability distribution over actions
"""
class PGNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='PGNetwork'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):
            with tf.name_scope("inputs"):
                # We create the placeholders
                # *state_size means that we take each elements of state_size in tuple hence is like if we wrote
                # [None, 84, 84, 4]
                self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs_")
                self.actions = tf.placeholder(tf.int32, [None, action_size], name="actions")
                self.discounted_episode_rewards_ = tf.placeholder(tf.float32, [None, ],
                                                                  name="discounted_episode_rewards_")
                # Add this placeholder for having this variable in tensorboard
                self.mean_reward_ = tf.placeholder(tf.float32, name="mean_reward")
            with tf.name_scope("conv1"):
                """
                First convnet:
                CNN
                BatchNormalization
                ELU
                """
                # Input is 84x84x4
                self.conv1 = tf.layers.conv2d(inputs=self.inputs_,
                                              filters=32,
                                              kernel_size=[8, 8],
                                              strides=[4, 4],
                                              padding="VALID",
                                              kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                              name="conv1")
                self.conv1_batchnorm = tf.layers.batch_normalization(self.conv1,
                                                                     training=True,
                                                                     epsilon=1e-5,
                                                                     name='batch_norm1')
                self.conv1_out = tf.nn.elu(self.conv1_batchnorm, name="conv1_out")
                ## --> [20, 20, 32]
            with tf.name_scope("conv2"):
                """
                Second convnet:
                CNN
                BatchNormalization
                ELU
                """
                self.conv2 = tf.layers.conv2d(inputs=self.conv1_out,
                                              filters=64,
                                              kernel_size=[4, 4],
                                              strides=[2, 2],
                                              padding="VALID",
                                              kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                              name="conv2")
                self.conv2_batchnorm = tf.layers.batch_normalization(self.conv2,
                                                                     training=True,
                                                                     epsilon=1e-5,
                                                                     name='batch_norm2')
                self.conv2_out = tf.nn.elu(self.conv2_batchnorm, name="conv2_out")
                ## --> [9, 9, 64]
            with tf.name_scope("conv3"):
                """
                Third convnet:
                CNN
                BatchNormalization
                ELU
                """
                self.conv3 = tf.layers.conv2d(inputs=self.conv2_out,
                                              filters=128,
                                              kernel_size=[4, 4],
                                              strides=[2, 2],
                                              padding="VALID",
                                              kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                              name="conv3")
                self.conv3_batchnorm = tf.layers.batch_normalization(self.conv3,
                                                                     training=True,
                                                                     epsilon=1e-5,
                                                                     name='batch_norm3')
                self.conv3_out = tf.nn.elu(self.conv3_batchnorm, name="conv3_out")
                ## --> [3, 3, 128]
            with tf.name_scope("flatten"):
                self.flatten = tf.layers.flatten(self.conv3_out)
                ## --> [1152]
            with tf.name_scope("fc1"):
                self.fc = tf.layers.dense(inputs=self.flatten,
                                          units=512,
                                          activation=tf.nn.elu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          name="fc1")
            with tf.name_scope("logits"):
                self.logits = tf.layers.dense(inputs=self.fc,
                                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                              units=3,
                                              activation=None)
            with tf.name_scope("softmax"):
                self.action_distribution = tf.nn.softmax(self.logits)
            with tf.name_scope("loss"):
                # tf.nn.softmax_cross_entropy_with_logits computes the cross entropy of the result after applying the softmax function
                # If you have single-class labels, where an object can only belong to one class, you might now consider using
                # tf.nn.sparse_softmax_cross_entropy_with_logits so that you don't have to convert your labels to a dense one-hot array.
                self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.actions)
                self.loss = tf.reduce_mean(self.neg_log_prob * self.discounted_episode_rewards_)
            with tf.name_scope("train"):
                self.train_opt = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

# Reset the graph
tf.reset_default_graph()
# Instantiate the PGNetwork
PGNetwork = PGNetwork(state_size, action_size, learning_rate)
# Initialize Session
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# Step 6: Set up Tensorboard
# Setup TensorBoard Writer
writer = tf.summary.FileWriter("/tensorboard/policy_gradients/1")
## Losses
tf.summary.scalar("Loss", PGNetwork.loss)
## Reward mean
tf.summary.scalar("Reward_mean", PGNetwork.mean_reward_ )
write_op = tf.summary.merge_all()

# Step 7: Train our Agent ️
"""
Here we'll create batches.
These batches contains episodes (their number depends on how many rewards we collect: for instance if we have
episodes with only 10 rewards we can put batch_size/10 episodes 
    Make a batch
        For each step:
            Choose action a
            Perform action a
            Store s, a, r
            If done:
                Calculate sum reward
                Calculate gamma Gt
"""
def make_batch(batch_size, stacked_frames):
    # Initialize lists: states, actions, rewards_of_episode, rewards_of_batch, discounted_rewards
    states, actions, rewards_of_episode, rewards_of_batch, discounted_rewards = [], [], [], [], []
    # Reward of batch is also a trick to keep track of how many timestep we made.
    # 批量奖励也是一个跟踪我们执行了多少时间步长的技巧。
    # We use to to verify at the end of each episode if > batch_size or not.
    # Keep track of how many episodes in our batch (useful when we'll need to calculate the average reward per episode)
    episode_num = 1
    # Launch a new episode
    game.new_episode()
    # Get a new state
    state = game.get_state().screen_buffer
    state, stacked_frames = stack_frames(stacked_frames, state, True)
    while True:
        # Run State Through Policy & Calculate Action
        action_probability_distribution = sess.run(PGNetwork.action_distribution,
                                                   feed_dict={PGNetwork.inputs_: state.reshape(1, *state_size)})
        # REMEMBER THAT WE ARE IN A STOCHASTIC POLICY SO WE DON'T ALWAYS TAKE THE ACTION WITH THE HIGHEST PROBABILITY
        # 记住，我们处于随机政策中所以我们并不总是采取概率最高的行动
        # (For instance if the action with the best probability for state S is a1 with 70% chances, there is
        # 30% chance that we take action a2)
        # (例如，如果状态S的最佳概率是a1，有70%的概率，那么我们采取行动a2的概率是30%)
        action = np.random.choice(range(action_probability_distribution.shape[1]),
                                  p=action_probability_distribution.ravel())    # select action w.r.t the actions prob
        action = possible_actions[action]
        # Perform action
        reward = game.make_action(action)
        done = game.is_episode_finished()
        # Store results
        states.append(state)
        actions.append(action)
        rewards_of_episode.append(reward)
        if done:
            # The episode ends so no next state
            next_state = np.zeros((100, 160), dtype=np.int)
            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
            # Append the rewards_of_batch to reward_of_episode
            rewards_of_batch.append(rewards_of_episode)
            # Calculate gamma Gt
            discounted_rewards.append(discount_and_normalize_rewards(rewards_of_episode))
            # If the number of rewards_of_batch > batch_size stop the minibatch creation
            # (Because we have sufficient number of episode mb)
            # Remember that we put this condition here, because we want entire episode (Monte Carlo)
            # so we can't check that condition for each step but only if an episode is finished
            if len(np.concatenate(rewards_of_batch)) > batch_size:
                break
            # Reset the transition stores
            rewards_of_episode = []
            # Add episode
            episode_num += 1
            # Start a new episode
            game.new_episode()
            # First we need a state
            state = game.get_state().screen_buffer
            # Stack the frames
            state, stacked_frames = stack_frames(stacked_frames, state, True)
        else:
            # If not done, the next_state become the current state
            next_state = game.get_state().screen_buffer
            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
            state = next_state

    return np.stack(np.array(states)), np.stack(np.array(actions)), np.concatenate(rewards_of_batch), np.concatenate(
        discounted_rewards), episode_num

"""
Create the Neural Network
Initialize the weights
Init the environment
maxReward = 0 # Keep track of maximum reward
For epochs in range(num_epochs):
    Get batches
    Optimize
"""
# Keep track of all rewards total for each batch
allRewards = []
total_rewards = 0
maximumRewardRecorded = 0
mean_reward_total = []
epoch = 1
average_reward = []

# Saver
saver = tf.train.Saver()
if training:
    # Load the model
    #saver.restore(sess, "./models/model.ckpt")
    while epoch < num_epochs + 1:
        # Gather training data
        states_mb, actions_mb, rewards_of_batch, discounted_rewards_mb, nb_episodes_mb = make_batch(batch_size, stacked_frames)

        ### These part is used for analytics
        # Calculate the total reward ot the batch
        total_reward_of_that_batch = np.sum(rewards_of_batch)
        allRewards.append(total_reward_of_that_batch)
        # Calculate the mean reward of the batch
        # Total rewards of batch / nb episodes in that batch
        mean_reward_of_that_batch = np.divide(total_reward_of_that_batch, nb_episodes_mb)
        mean_reward_total.append(mean_reward_of_that_batch)
        # Calculate the average reward of all training
        # mean_reward_of_that_batch / epoch
        average_reward_of_all_training = np.divide(np.sum(mean_reward_total), epoch)
        # Calculate maximum reward recorded
        maximumRewardRecorded = np.amax(allRewards)
        print("==========================================")
        print("Epoch: ", epoch, "/", num_epochs)
        print("-----------")
        print("Number of training episodes: {}".format(nb_episodes_mb))
        print("Total reward: {}".format(total_reward_of_that_batch, nb_episodes_mb))
        print("Mean Reward of that batch {}".format(mean_reward_of_that_batch))
        print("Average Reward of all training: {}".format(average_reward_of_all_training))
        print("Max reward for a batch so far: {}".format(maximumRewardRecorded))

        # Feedforward, gradient and backpropagation
        loss_, _ = sess.run([PGNetwork.loss, PGNetwork.train_opt], feed_dict={PGNetwork.inputs_: states_mb.reshape((len(states_mb), 100,160,4)),
                                                                              PGNetwork.actions: actions_mb,
                                                                              PGNetwork.discounted_episode_rewards_: discounted_rewards_mb
                                                                             })
        print("Training Loss: {}".format(loss_))
        # Write TF Summaries
        summary = sess.run(write_op, feed_dict={PGNetwork.inputs_: states_mb.reshape((len(states_mb), 100,160,4)),
                                                PGNetwork.actions: actions_mb,
                                                PGNetwork.discounted_episode_rewards_: discounted_rewards_mb,
                                                PGNetwork.mean_reward_: mean_reward_of_that_batch
                                                })
        #summary = sess.run(write_op, feed_dict={x: s_.reshape(len(s_),84,84,1), y:a_, d_r: d_r_, r: r_, n: n_})
        writer.add_summary(summary, epoch)
        writer.flush()

        # Save Model
        if epoch % 10 == 0:
            saver.save(sess, "./models/model.ckpt")
            print("Model saved")
        epoch += 1

# Step 8: Watch our Agent play
# Saver
saver = tf.train.Saver()
with tf.Session() as sess:
    game = DoomGame()
    # Load the correct configuration
    game.load_config("defend_the_center.cfg")
    # Load the correct scenario (in our case basic scenario)
    game.set_doom_scenario_path("defend_the_center.wad")
    # Load the model
    saver.restore(sess, "./models/model.ckpt")
    game.init()
    for i in range(10):
        # Launch a new episode
        game.new_episode()
        # Get a new state
        state = game.get_state().screen_buffer
        state, stacked_frames = stack_frames(stacked_frames, state, True)
        while not game.is_episode_finished():
            # Run State Through Policy & Calculate Action
            action_probability_distribution = sess.run(PGNetwork.action_distribution,
                                                       feed_dict={PGNetwork.inputs_: state.reshape(1, *state_size)})
            # REMEMBER THAT WE ARE IN A STOCHASTIC POLICY SO WE DON'T ALWAYS TAKE THE ACTION WITH THE HIGHEST PROBABILITY
            # (For instance if the action with the best probability for state S is a1 with 70% chances, there is
            # 30% chance that we take action a2)
            action = np.random.choice(range(action_probability_distribution.shape[1]),
                                      p=action_probability_distribution.ravel())  # select action w.r.t the actions prob
            action = possible_actions[action]
            # Perform action
            reward = game.make_action(action)
            done = game.is_episode_finished()
            if done:
                break
            else:
                # If not done, the next_state become the current state
                next_state = game.get_state().screen_buffer
                next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                state = next_state
        print("Score for episode ", i, " :", game.get_total_reward())
    game.close()
