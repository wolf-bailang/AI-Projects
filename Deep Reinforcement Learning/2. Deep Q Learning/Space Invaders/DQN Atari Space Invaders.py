# Deep Q Learning with Atari Space Invaders

# @Thomas Simonini
# https://github.com/simoninithomas/Deep_reinforcement_learning_Course/tree/master/Deep%20Q%20Learning/Space%20Invaders

import tensorflow as tf      # Deep Learning library
import numpy as np           # Handle matrices处理矩阵
import retro                 # Retro Environment

from skimage import transform # Help us to preprocess the frames帮助我们对框架进行预处理
from skimage.color import rgb2gray # Help us to gray our frames帮我们把框架弄灰
import matplotlib.pyplot as plt # Display graphs
from collections import deque# Ordered collection with ends有序集合
import random
# This ignore all the warning messages that are normally printed during the training because of skiimage
# 这将忽略由于skiimage而在训练期间通常打印的所有警告消息
import warnings
warnings.filterwarnings('ignore')

# Step 2: Create our environment
env = retro.make(game='SpaceInvaders-Atari2600')
print("The size of our frame is: ", env.observation_space) # 框架的大小
print("The action size is : ", env.action_space.n) # 动作的大小
# Here we create an hot encoded version of our actions在这里，我们创建了动作的热编码版本
# possible_actions = [[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0]...]
# np.identity只能创建方形矩阵，返回的是nxn的主对角线为1，其余地方为0的数组
possible_actions = np.array(np.identity(env.action_space.n,dtype=int).tolist())

# Step 3 : Define the preprocessing functions
"""
Grayscale each of our frames (because color does not add important information ).
Crop the screen (in our case we remove the part below the player because it does not add any useful information)
We normalize pixel values
Finally we resize the preprocessed frame
"""
"""
    preprocess_frame:
    Take a frame.做一个框架
    Grayscale it灰度它
    Resize it.调整它
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
    Normalize it.正常化
    return preprocessed_frame
    """
def preprocess_frame(frame):
    # Greyscale frame 灰度框架
    gray = rgb2gray(frame)
    # Crop the screen (remove the part below the player)裁剪屏幕(删除播放器下方的部分)
    # [Up: Down, Left: right]
    cropped_frame = gray[8:-12, 4:-12]
    # Normalize Pixel Values规范化的像素值
    normalized_frame = cropped_frame / 255.0
    # Resize调整大小
    # Thanks to Mikołaj Walkowiak
    preprocessed_frame = transform.resize(normalized_frame, [110, 84])
    return preprocessed_frame  # 110x84x1 frame

# stack_frames
"""
The frame skipping method is already implemented in the library.
    First we preprocess frame
    Then we append the frame to the deque that automatically removes the oldest frame
    Finally we build the stacked state

This is how work stack:
    For the first frame, we feed 4 frames
    At each timestep, we add the new frame to deque and then we stack them to form a new stacked frame
    And so on 
"""
stack_size = 4  # We stack 4 frames堆叠4帧
# Initialize deque with zero-images one array for each image初始化deque与零图像每个图像一个数组
stacked_frames = deque([np.zeros((110, 84), dtype=np.int) for i in range(stack_size)], maxlen=4)

def stack_frames(stacked_frames, state, is_new_episode):
    # Preprocess frame预处理框架
    frame = preprocess_frame(state)
    if is_new_episode:
        # Clear our stacked_frames明确我们stacked_frames
        stacked_frames = deque([np.zeros((110, 84), dtype=np.int) for i in range(stack_size)], maxlen=4)
        # Because we're in a new episode, copy the same frame 4x因为我们在新一集，复制相同的框架4x
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        # Stack the frames堆叠帧
        stacked_state = np.stack(stacked_frames, axis=2)
    else:
        # Append frame to deque, automatically removes the oldest frame添加帧到deque，自动删除最老的帧
        stacked_frames.append(frame)
        # Build the stacked state (first dimension specifies different frames)构建堆栈状态(第一个维度指定不同的帧)
        stacked_state = np.stack(stacked_frames, axis=2)
    return stacked_state, stacked_frames

# Step 4: Set up our hyperparameters
"""
First, you begin by defining the neural networks hyperparameters when you implement the model.
Then, you'll add the training hyperparameters when you implement the training algorithm.
"""
### MODEL HYPERPARAMETERS
state_size = [110, 84, 4]      # Our input is a stack of 4 frames hence 110x84x4 (Width, height, channels)
action_size = env.action_space.n # 8 possible actions
learning_rate =  0.00025      # Alpha (aka learning rate)
### TRAINING HYPERPARAMETERS
total_episodes = 50            # Total episodes for training
max_steps = 50000              # Max possible steps in an episode在一集里尽可能多的步骤
batch_size = 64                # Batch size
# Exploration parameters for epsilon greedy strategy贪心策略的探索参数
explore_start = 1.0            # exploration probability at start开始勘探概率
explore_stop = 0.01            # minimum exploration probability 最小探测概率
decay_rate = 0.00001           # exponential decay rate for exploration prob勘探过程的指数衰减率
# Q learning hyperparameters
gamma = 0.9                    # Discounting rate折扣率
### MEMORY HYPERPARAMETERS
pretrain_length = batch_size   # Number of experiences stored in the Memory when initialized for the first time第一次初始化时存储在内存中的经验数
memory_size = 1000000          # Number of experiences the Memory can keep记忆可以保留的经验的数量
### PREPROCESSING HYPERPARAMETERS
stack_size = 4                 # Number of frames stacked叠加帧数
### MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT如果您只想查看训练过的代理，请将其修改为FALSE
training = False
## TURN THIS TO TRUE IF YOU WANT TO RENDER THE ENVIRONMENT如果您想渲染环境，请将其设置为TRUE
episode_render = False

# Step 5: Create our Deep Q-learning Neural Network model
"""
Deep Q-learning model:
    We take a stack of 4 frames as input
    It passes through 3 convnets
    Then it is flatened
    Finally it passes through 2 FC layers
    It outputs a Q value for each actions
"""
class DQNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='DQNetwork'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):
            # We create the placeholders我们创建占位符
            # *state_size means that we take each elements of state_size in tuple hence is like if we wrote
            # *state_size的意思是我们把state_size的每个元素都放到tuple中，就像我们写的那样
            # [None, 84, 84, 4]
            self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs")
            self.actions_ = tf.placeholder(tf.float32, [None, self.action_size], name="actions_")

            # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
            # 记住target_Q是R(s,a) + ymax Qhat(s'， a')
            self.target_Q = tf.placeholder(tf.float32, [None], name="target")

            """
            First convnet:
            CNN
            RELU
            """
            # Input is 110x84x4
            self.conv1 = tf.layers.conv2d(inputs=self.inputs_, filters=32, kernel_size=[8, 8], strides=[4, 4],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv1")
            self.conv1_out = tf.nn.relu(self.conv1, name="conv1_out")
            """
            Second convnet:
            CNN
            RELU
            """
            self.conv2 = tf.layers.conv2d(inputs=self.conv1_out,
                                          filters=64,
                                          kernel_size=[4, 4],
                                          strides=[2, 2],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv2")
            self.conv2_out = tf.nn.relu(self.conv2, name="conv2_out")
            """
            Third convnet:
            CNN
            RELU
            """
            self.conv3 = tf.layers.conv2d(inputs=self.conv2_out,
                                          filters=64,
                                          kernel_size=[3, 3],
                                          strides=[2, 2],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv3")
            self.conv3_out = tf.nn.relu(self.conv3, name="conv3_out")
            self.flatten = tf.contrib.layers.flatten(self.conv3_out)
            self.fc = tf.layers.dense(inputs=self.flatten,
                                      units=512,
                                      activation=tf.nn.elu,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                      name="fc1")
            self.output = tf.layers.dense(inputs=self.fc,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          units=self.action_size,
                                          activation=None)

            # Q is our predicted Q value.Q是我们预测的Q值
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_))
            # The loss is the difference between our predicted Q_values and the Q_target损失是我们预测的q_值与Q_target之间的差值
            # Sum(Qtarget - Q)^2
            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

# Reset the graph重置图
tf.reset_default_graph()
# Instantiate the DQNetwork实例化DQNetwork
DQNetwork = DQNetwork(state_size, action_size, learning_rate)

# Step 6: Experience Replay
"""
Here we'll create the Memory object that creates a deque.A deque (double ended queue) is a data type
that removes the oldest element each time that you add a new element.
"""
class Memory():
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size),
                                 size=batch_size,
                                 replace=False)
        return [self.buffer[i] for i in index]
"""
Here we'll deal with the empty memory problem: we pre-populate our memory by taking random actions and
storing the experience (state, action, reward, next_state).
"""
# Instantiate memory初始化内存
memory = Memory(max_size=memory_size)
for i in range(pretrain_length):
    # If it's the first step如果这是第一步
    if i == 0:
        state = env.reset()
        state, stacked_frames = stack_frames(stacked_frames, state, True)
    # Get the next_state, the rewards, done by taking a random action通过随机操作来获得next_state，即奖励
    choice = random.randint(1, len(possible_actions)) - 1
    action = possible_actions[choice]
    next_state, reward, done, _ = env.step(action)
    # env.render()
    # Stack the frames
    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
    # If the episode is finished (we're dead 3x)如果这一集结束了(我们死了3x)
    if done:
        # We finished the episode我们结束了这一集
        next_state = np.zeros(state.shape)
        # Add experience to memory给记忆添加经验
        memory.add((state, action, reward, next_state, done))
        # Start a new episode开始新的一集
        state = env.reset()
        # Stack the frames
        state, stacked_frames = stack_frames(stacked_frames, state, True)
    else:
        # Add experience to memory给记忆添加经验
        memory.add((state, action, reward, next_state, done))
        # Our new state is now the next_state我们的新状态现在是next_state
        state = next_state

# Step 7: Set up Tensorboard
# Setup TensorBoard Writer
writer = tf.summary.FileWriter("/tensorboard/dqn/1")
## Losses
tf.summary.scalar("Loss", DQNetwork.loss)
write_op = tf.summary.merge_all()

# Step 8: Train our Agent
"""
Our algorithm: 
Initialize the weights
Init the environment
Initialize the decay rate (that will use to reduce epsilon) 

For episode to max_episode do 
    Make new episode
    Set step to 0
    Observe the first state s0

While step < max_steps do:
    Increase decay_rate
    With ϵ select a random action at, otherwise select at=argmaxaQ(st,a)
    Execute action at in simulator and observe reward rt+1 and new state st+1
    Store transition <st,at,rt+1,st+1> in memory D
    Sample random mini-batch from D: <s,a,r,s′>
    Set Q^=r if the episode ends at +1, otherwise set Q^=r+γmaxa′Q(s′,a′)
    Make a gradient descent step with loss (Q^−Q(s,a))2
    endfor 
endfor
"""
"""
This function will do the part
With ϵϵ select a random action atat, otherwise select at=argmaxaQ(st,a)
"""
def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, actions):
    ## EPSILON GREEDY STRATEGYε贪婪策略
    # Choose action a from state s using epsilon greedy.使用贪心从状态s中选择动作a。
    ## First we randomize a number首先我们随机选择一个数字
    exp_exp_tradeoff = np.random.rand()
    # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
    # 这里我们将使用Q-learning笔记本中使用的epsilon贪心策略的改进版本
    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
    if (explore_probability > exp_exp_tradeoff):
        # Make a random action (exploration) 随机行动(探索)
        choice = random.randint(1, len(possible_actions)) - 1
        action = possible_actions[choice]
    else:
        # Get action from Q-network (exploitation)从q网络获取行动(利用)
        # Estimate the Qs values state估计Qs值的状态
        Qs = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs_: state.reshape((1, *state.shape))})
        # Take the biggest Q value (= the best action)取最大Q值(=最佳动作)
        choice = np.argmax(Qs)
        action = possible_actions[choice]
    return action, explore_probability

# Saver will help us to save our model保存我们的模型
saver = tf.train.Saver()
if training == True:
    with tf.Session() as sess:
        # Initialize the variables初始化变量
        sess.run(tf.global_variables_initializer())
        # Initialize the decay rate (that will use to reduce epsilon) 初始化衰减率(用来减小)
        decay_step = 0
        for episode in range(total_episodes):
            # Set step to 0
            step = 0
            # Initialize the rewards of the episode初始化事件的奖励
            episode_rewards = []
            # Make a new episode and observe the first state制作一个新片段，观察第一种状态
            state = env.reset()
            # Remember that stack frame function also call our preprocess function.记住堆栈帧函数也调用预处理函数
            state, stacked_frames = stack_frames(stacked_frames, state, True)
            while step < max_steps:
                step += 1
                # Increase decay_step增加decay_step
                decay_step += 1
                # Predict the action to take and take it预测要采取的行动，然后采取行动
                action, explore_probability = predict_action(explore_start, explore_stop, decay_rate, decay_step, state,
                                                             possible_actions)
                # Perform the action and get the next_state, reward, and done information执行该操作并获得next_state、reward和done信息
                next_state, reward, done, _ = env.step(action)
                if episode_render:
                    env.render()
                # Add the reward to total reward将奖励加到总奖励中
                episode_rewards.append(reward)
                # If the game is finished如果游戏结束了
                if done:
                    # The episode ends so no next state这一集结束了，所以没有下一个state
                    next_state = np.zeros((110, 84), dtype=np.int)
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                    # Set step = max_steps to end the episode设置step = max_steps结束本集
                    step = max_steps
                    # Get the total reward of the episode获得该集的总奖励
                    total_reward = np.sum(episode_rewards)
                    print('Episode: {}'.format(episode),
                          'Total reward: {}'.format(total_reward),
                          'Explore P: {:.4f}'.format(explore_probability),
                          'Training Loss {:.4f}'.format(loss))
                    rewards_list.append((episode, total_reward))
                    # Store transition <st,at,rt+1,st+1> in memory D存储转换<st,at,rt+1,st+1>在内存D中
                    memory.add((state, action, reward, next_state, done))
                else:
                    # Stack the frame of the next_state堆栈next_state的帧
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                    # Add experience to memory给记忆添加经验
                    memory.add((state, action, reward, next_state, done))
                    # st+1 is now our current state st+1现在是我们的状态
                    state = next_state
                ### LEARNING PART
                # Obtain random mini-batch from memory从内存中获取随机小批处理
                batch = memory.sample(batch_size)
                states_mb = np.array([each[0] for each in batch], ndmin=3)
                actions_mb = np.array([each[1] for each in batch])
                rewards_mb = np.array([each[2] for each in batch])
                next_states_mb = np.array([each[3] for each in batch], ndmin=3)
                dones_mb = np.array([each[4] for each in batch])
                target_Qs_batch = []
                # Get Q values for next_state 获取next_state的Q值
                Qs_next_state = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs_: next_states_mb})
                # Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gamma*maxQ(s', a')
                # 设置Q_target = r如果事件在s+1结束，则设置Q_target = r + gamma * maxQ(s'， a')
                for i in range(0, len(batch)):
                    terminal = dones_mb[i]
                    # If we are in a terminal state, only equals reward如果我们处于终极状态，就只等于奖励
                    if terminal:
                        target_Qs_batch.append(rewards_mb[i])
                    else:
                        target = rewards_mb[i] + gamma * np.max(Qs_next_state[i])
                        target_Qs_batch.append(target)
                targets_mb = np.array([each for each in target_Qs_batch])
                loss, _ = sess.run([DQNetwork.loss, DQNetwork.optimizer],
                                   feed_dict={DQNetwork.inputs_: states_mb,
                                              DQNetwork.target_Q: targets_mb,
                                              DQNetwork.actions_: actions_mb})
                # Write TF Summaries
                summary = sess.run(write_op, feed_dict={DQNetwork.inputs_: states_mb,
                                                        DQNetwork.target_Q: targets_mb,
                                                        DQNetwork.actions_: actions_mb})
                writer.add_summary(summary, episode)
                writer.flush()
            # Save model every 5 episodes每5集保存模型
            if episode % 5 == 0:
                save_path = saver.save(sess, "./models/model.ckpt")
                print("Model Saved")

# Step 9: Test and Watch our Agent play
with tf.Session() as sess:
    total_test_rewards = []
    # Load the model
    saver.restore(sess, "./models/model.ckpt")
    for episode in range(1):
        total_rewards = 0
        state = env.reset()
        state, stacked_frames = stack_frames(stacked_frames, state, True)
        print("****************************************************")
        print("EPISODE ", episode)
        while True:
            # Reshape the state
            state = state.reshape((1, *state_size))
            # Get action from Q-network 从Q-network获得动作
            # Estimate the Qs values state估计Qs值的状态
            Qs = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs_: state})
            # Take the biggest Q value (= the best action)取最大Q值(=最佳动作)
            choice = np.argmax(Qs)
            action = possible_actions[choice]
            # Perform the action and get the next_state, reward, and done information执行该操作并获得next_state、reward和done信息
            next_state, reward, done, _ = env.step(action)
            env.render()
            total_rewards += reward
            if done:
                print("Score", total_rewards)
                total_test_rewards.append(total_rewards)
                break
            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
            state = next_state
    env.close()
