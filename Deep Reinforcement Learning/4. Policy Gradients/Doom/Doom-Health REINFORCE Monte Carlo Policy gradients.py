# -*- coding: UTF-8 -*-
# try to survive in Doom environment by using a Policy Gradient architecture.

# @Thomas Simonini
# https://github.com/simoninithomas/Deep_reinforcement_learning_Course/tree/master/Policy%20Gradients/Doom

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
一个处理所有选项的配置文件(框架的大小，可能的操作…)
A scenario file: that generates the correct scenario (in our case basic but you're invited to try other scenarios).
场景文件:生成正确的场景(在我们的示例中是basic，但我们邀请您尝试其他场景)。
Note: We have 3 possible actions [[0,0,1], [1,0,0], [0,1,0]] so we don't need to do one hot encoding

environment
The purpose of this scenario is to teach the agent how to survive without knowing what makes him survive. Agent know
only that life is precious and death is bad so he must learn what prolongs his existence and that his health is connected with it.
这个场景的目的是教会代理如何生存，而不知道是什么让他生存下来。代理人只知道生命是宝贵的，死亡是坏的，所以他必须知道是什么延长了他的存在，他的健康是与之相关的。

Map is a rectangle with green, acidic floor which hurts the player periodically. Initially there are some medkits spread
uniformly over the map. A new medkit falls from the skies every now and then. Medkits heal some portions of player's 
health - to survive agent needs to pick them up. Episode finishes after player's death or on timeout.
地图是一个带有绿色、酸性地板的矩形，会周期性地伤害玩家。最初有一些医疗包均匀地分布在地图上。一个新的医疗设备不时地从天
而降。医疗包可以治疗一部分玩家的健康——为了生存，特工需要把它们捡起来。在玩家死亡或暂停后，游戏结束。

Further configuration:
    living_reward = 1
    3 available buttons: turn left, turn right, move forward 3个可用按钮:左转，右转，向前移动
    1 available game variable: HEALTH可用的游戏变量:健康
    death penalty = 100 
"""
"""
Here we create our environment
"""
def create_environment():
    game = DoomGame()
    # Load the correct configuration
    game.load_config("health_gathering.cfg")
    # Load the correct scenario (in our case defend_the_center scenario)
    game.set_doom_scenario_path("health_gathering.wad")
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
预处理是一个重要的步骤，因为我们想要降低状态的复杂性，以减少训练所需的计算时间。

Our steps:
    Grayscale each of our frames (because color does not add important information ). But this is already done by the config file.
    每个帧的灰度值(因为颜色不添加重要的信息)。但这已经由配置文件完成了。
    Crop the screen (in our case we remove the roof because it contains no information)
    裁剪屏幕(在我们的示例中，我们删除屋顶，因为它不包含任何信息)
    We normalize pixel values对像素值进行归一化
    Finally we resize the preprocessed frame调整预处理帧的大小
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
    # Greyscale frame already done in our vizdoom config灰度框架已经完成在我们的vizdoom配置
    # x = np.mean(frame,-1)
    # Crop the screen (remove the roof because it contains no information)裁剪屏幕(删除屋顶，因为它不包含任何信息)
    # [Up: Down, Left: right]
    cropped_frame = frame[80:, :]
    # Normalize Pixel Values
    normalized_frame = cropped_frame / 255.0
    # Resize
    preprocessed_frame = transform.resize(normalized_frame, [84, 84])
    return preprocessed_frame

"""
stack_frames
Stacking frames is really important because it helps us to give have a sense of motion to our Neural Network.
    First we preprocess frame首先对框架进行预处理
    Then we append the frame to the deque that automatically removes the oldest frame
    然后我们将这个框架附加到deque，它会自动删除最老的框架
    Finally we build the stacked state最后我们构建堆栈状态
This is how work stack:
    For the first frame, we feed 4 frames对于第一帧，我们输入4帧
    At each timestep, we add the new frame to deque and then we stack them to form a new stacked frame
    在每个时间步中，我们将新框架添加到deque中，然后将它们堆叠起来，形成一个新的堆叠框架
    And so on 
If we're done, we create a new stack with 4 new frames (because we are in a new episode).
"""
stack_size = 4  # We stack 4 frames
# Initialize deque with zero-images one array for each image初始化deque与零图像每个图像一个数组
stacked_frames = deque([np.zeros((84, 84), dtype=np.int) for i in range(stack_size)], maxlen=4)

def stack_frames(stacked_frames, state, is_new_episode):
    # Preprocess frame
    frame = preprocess_frame(state)
    if is_new_episode:
        # Clear our stacked_frames
        stacked_frames = deque([np.zeros((84, 84), dtype=np.int) for i in range(stack_size)], maxlen=4)
        # Because we're in a new episode, copy the same frame 4x 4x因为我们在新一集，复制相同的坐标系4x
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        # Stack the frames叠加帧
        # axis=0对指定axis增加维度
        # axis=1时，对二维平面的行进行增加
        stacked_state = np.stack(stacked_frames, axis=2)
    else:
        # Append frame to deque, automatically removes the oldest frame添加帧到deque，自动删除最老的帧
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
In this part we'll set up our different hyperparameters. But when you implement a Neural Network by yourself
you will not implement hyperparamaters at once but progressively.
    First, you begin by defining the neural networks hyperparameters when you implement the model.
    Then, you'll add the training hyperparameters when you implement the training algorithm.
"""
### ENVIRONMENT HYPERPARAMETERS环境超参数
state_size = [84,84,4]      # Our input is a stack of 4 frames hence 84x84x4 (Width, height, channels)我们的输入是4帧的堆栈，因此是84x84x4(宽度，高度，通道)
action_size = game.get_available_buttons_size()     # 3 possible actions: turn left, turn right, move forward3个可能的动作:向左拐，向右拐，向前走
stack_size = 4      # Defines how many frames are stacked together定义有多少帧被堆叠在一起
## TRAINING HYPERPARAMETERS
learning_rate = 0.002
num_epochs = 500    # Total epochs for training
batch_size = 1000   # Each 1 is a timestep (NOT AN EPISODE) # YOU CAN CHANGE TO 5000 if you have GPU每个1是一个时间步长(不是一集)。如果你有GPU，你可以把它改成5000
gamma = 0.95    # Discounting rate
### MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT如果您只想查看训练过的代理，请将其修改为FALSE
training = True

"""
Policy gradient methods like reinforce are on-policy method which can not be updated from experience replay.
强化策略梯度法是一种基于策略的方法，不能通过经验回放进行更新。
"""
# Step 5: Create our Policy Gradient Neural Network model
class PGNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='PGNetwork'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):
            with tf.name_scope("inputs"):
                # We create the placeholders
                # *state_size means that we take each elements of state_size in tuple hence is like if we wrote[None, 84, 84, 4]
                # *state_size的意思是我们把state_size的每个元素放在一个元组中，就像我们写了[None, 84, 84, 4]
                self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs_")
                self.actions = tf.placeholder(tf.int32, [None, action_size], name="actions")
                self.discounted_episode_rewards_ = tf.placeholder(tf.float32, [None, ], name="discounted_episode_rewards_")
                # Add this placeholder for having this variable in tensorboard在tensorboard中添加这个变量的占位符
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
                # tf.nn.elu指数线性单元
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
                self.fc1 = tf.layers.dense(inputs=self.flatten,
                                          units=512,
                                          activation=tf.nn.elu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          name="fc1")
            with tf.name_scope("logits"):
                self.logits = tf.layers.dense(inputs=self.fc1,
                                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                              units=3,
                                              activation=None)
            with tf.name_scope("softmax"):
                self.action_distribution = tf.nn.softmax(self.logits)
            with tf.name_scope("loss"):
                # tf.nn.softmax_cross_entropy_with_logits computes the cross entropy of the result after applying the softmax function
                # tf.nn。softmax_cross_entropy_with_logits计算 softmax(logits) 和 labels 之间的交叉熵
                # If you have single-class labels, where an object can only belong to one class, you might now consider using
                # 如果您有单类标签，其中一个对象只能属于一个类，那么现在可以考虑使用
                # tf.nn.sparse_softmax_cross_entropy_with_logits so that you don't have to convert your labels to a dense one-hot array.
                # tf.nn。sparse_softmax_cross_entropy_with_logits，这样您就不必将标签转换为密集的单热数组。
                self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.actions)
                self.loss = tf.reduce_mean(self.neg_log_prob * self.discounted_episode_rewards_)
            with tf.name_scope("train"):
                self.train_opt = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

# Reset the graph
tf.reset_default_graph()
# Instantiate the PGNetwork实例化PGNetwork
PGNetwork = PGNetwork(state_size, action_size, learning_rate)
# Initialize Session
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# Step 6: Set up Tensorboard
# Setup TensorBoard Writer
writer = tf.summary.FileWriter("/tensorboard/pg/test")
## Losses
tf.summary.scalar("Loss", PGNetwork.loss)
## Reward mean
tf.summary.scalar("Reward_mean", PGNetwork.mean_reward_ )
write_op = tf.summary.merge_all()

# Step 7: Train our Agent
"""
Here we'll create batches.这里我们将创建批处理。
These batches contains episodes (their number depends on how many rewards we collect: for instance if we have episodes
with only 10 rewards we can put batch_size/10 episodes 
这些批量包含剧集(它们的数量取决于我们收集了多少奖励:例如，如果我们的剧集只有10个奖励，那么我们可以把batch_size/10集放在一起
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
    # Reward of batch is also a trick to keep track of how many timestep we made.批量奖励也是一个跟踪我们执行了多少时间步长的技巧。
    # We use to to verify at the end of each episode if > batch_size or not.我们习惯于在每一集结束时验证> batch_size是否正确。
    # Keep track of how many episodes in our batch (useful when we'll need to calculate the average reward per episode)
    # 记录我们一季有多少集(当我们需要计算每集的平均奖励时很有用)
    episode_num = 1
    # Launch a new episode
    game.new_episode()
    # Get a new state
    state = game.get_state().screen_buffer
    state, stacked_frames = stack_frames(stacked_frames, state, True)
    while True:
        # Run State Through Policy & Calculate Action通过策略和计算操作运行状态
        action_probability_distribution = sess.run(PGNetwork.action_distribution,
                                                   feed_dict={PGNetwork.inputs_: state.reshape(1, *state_size)})
        # REMEMBER THAT WE ARE IN A STOCHASTIC POLICY SO WE DON'T ALWAYS TAKE THE ACTION WITH THE HIGHEST PROBABILITY
        # 记住，我们处于随机政策中所以我们并不总是采取概率最高的行动
        # (For instance if the action with the best probability for state S is a1 with 70% chances, there is 30% chance that we take action a2)
        # (例如，如果状态S的最佳概率是a1，有70%的概率，那么我们采取行动a2的概率是30%)
        action = np.random.choice(range(action_probability_distribution.shape[1]),
                                  p=action_probability_distribution.ravel())  # select action w.r.t the actions prob
        action = possible_actions[action]
        # Perform action
        reward = game.make_action(action)
        done = game.is_episode_finished()
        # Store results
        states.append(state)
        actions.append(action)
        rewards_of_episode.append(reward)
        if done:
            # The episode ends so no next state这一集结束了，所以没有下一个state
            next_state = np.zeros((84, 84), dtype=np.int)
            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
            # Append the rewards_of_batch to reward_of_episode
            rewards_of_batch.append(rewards_of_episode)
            # Calculate gamma Gt计算γGt
            discounted_rewards.append(discount_and_normalize_rewards(rewards_of_episode))
            # If the number of rewards_of_batch > batch_size stop the minibatch creation如果奖励数为_of_batch > batch_size，则停止创建小批处理
            # (Because we have sufficient number of episode mb)(因为我们有足够的剧集mb)
            # Remember that we put this condition here, because we want entire episode (Monte Carlo)
            # 记住，我们把这个条件放在这里，因为我们想要整个事件(蒙特卡洛)
            # so we can't check that condition for each step but only if an episode is finished
            # 所以我们不能检查每一步的情况，只有在一集结束的时候
            if len(np.concatenate(rewards_of_batch)) > batch_size:      # np.concatenate数组拼接
                break
            # Reset the transition stores重置转换存储
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
            # If not done, the next_state become the current state如果没有完成，next_state将成为当前状态
            next_state = game.get_state().screen_buffer
            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
            state = next_state
    return np.stack(np.array(states)), np.stack(np.array(actions)), np.concatenate(rewards_of_batch), np.concatenate(
        discounted_rewards), episode_num

"""
Create the Neural Network
Initialize the weights
Init the environment
maxReward = 0       # Keep track of maximum reward
For epochs in range(num_epochs):
    Get batches
    Optimize
"""
# Keep track of all rewards total for each batch跟踪每批奖励的总数
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
        # Gather training data收集培训数据
        states_mb, actions_mb, rewards_of_batch, discounted_rewards_mb, nb_episodes_mb = make_batch(batch_size, stacked_frames)
        ### These part is used for analytics这些部分用于分析
        # Calculate the total reward ot the batch计算该批次的总奖励
        total_reward_of_that_batch = np.sum(rewards_of_batch)
        allRewards.append(total_reward_of_that_batch)
        # Calculate the mean reward of the batch计算每批的平均奖励
        # Total rewards of batch / nb episodes in that batch
        mean_reward_of_that_batch = np.divide(total_reward_of_that_batch, nb_episodes_mb)
        mean_reward_total.append(mean_reward_of_that_batch)
        # Calculate the average reward of all training计算所有培训的平均回报
        # mean_reward_of_that_batch / epoch
        average_reward_of_all_training = np.divide(np.sum(mean_reward_total), epoch)
        # Calculate maximum reward recorded计算已录得的最高奖赏
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
        loss_, _ = sess.run([PGNetwork.loss, PGNetwork.train_opt], feed_dict={PGNetwork.inputs_: states_mb.reshape((len(states_mb), 84,84,4)),
                                                                              PGNetwork.actions: actions_mb,
                                                                              PGNetwork.discounted_episode_rewards_: discounted_rewards_mb
                                                                             })
        print("Training Loss: {}".format(loss_))
        # Write TF Summaries
        summary = sess.run(write_op, feed_dict={PGNetwork.inputs_: states_mb.reshape((len(states_mb), 84,84,4)),
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
    game.load_config("health_gathering.cfg")
    # Load the correct scenario (in our case basic scenario)
    game.set_doom_scenario_path("health_gathering.wad")
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
            # (For instance if the action with the best probability for state S is a1 with 70% chances, there is 30% chance that we take action a2)
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
