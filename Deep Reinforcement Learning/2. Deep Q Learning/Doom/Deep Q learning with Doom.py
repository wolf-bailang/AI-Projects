# Deep Q learning with Doom

# @Thomas Simonini
# https://github.com/wolf-bailang/Deep_reinforcement_learning_Course/tree/master/Deep%20Q%20Learning/Doom

import tensorflow as tf      # Deep Learning library
import numpy as np           # Handle matrices
from vizdoom import *        # Doom Environment

import random                # Handling random number generation
import time                  # Handling time calculation
# 图像的形变与缩放，图像批量转换为灰度图
from skimage import transform# Help us to preprocess the frames
# deque是为了高效实现插入和删除操作的双向列表,适合用于队列和栈
from collections import deque# Ordered collection with ends
import matplotlib.pyplot as plt # Display graphs

import warnings # This ignore all the warning messages that are normally printed during the training because of skiimage
# 这将忽略由于skiimage而在训练期间通常打印的所有警告消息
warnings.filterwarnings('ignore')

# Create our environment
"""
Now that we imported the libraries/dependencies, we will create our environment.
Doom environment takes:
A configuration file that handle all the options (size of the frame, possible actions...)
A scenario file: that generates the correct scenario (in our case basic but you're invited to try other scenarios).
Note: We have 3 possible actions [[0,0,1], [1,0,0], [0,1,0]] so we don't need to do one hot encoding

A monster is spawned randomly somewhere along the opposite wall. 
Player can only go left/right and shoot. 
1 hit is enough to kill the monster. 
Episode finishes when monster is killed or on timeout (300). 

REWARDS:
+101 for killing the monster 
-5 for missing 
Episode ends after killing the monster or on timeout.
living reward = -1
"""
# Here we create our environment
def create_environment():
    game = DoomGame()
    # Load the correct configuration加载正确的配置
    game.load_config("basic.cfg")
    # Load the correct scenario (in our case basic scenario)加载正确的场景(在我们的示例中是基本场景)
    game.set_doom_scenario_path("basic.wad")
    game.init()
    # Here our possible actions这里是我们可能的行动
    left = [1, 0, 0]
    right = [0, 1, 0]
    shoot = [0, 0, 1]
    possible_actions = [left, right, shoot]
    return game, possible_actions

# Here we performing random action to test the environment在这里，我们执行随机操作来测试环境
def test_environment():
    game = DoomGame()
    game.load_config("basic.cfg")
    game.set_doom_scenario_path("basic.wad")
    game.init()
    shoot = [0, 0, 1]
    left = [1, 0, 0]
    right = [0, 1, 0]
    actions = [shoot, left, right]

    episodes = 10
    for i in range(episodes):
        game.new_episode()
        while not game.is_episode_finished():
            state = game.get_state()
            img = state.screen_buffer
            misc = state.game_variables
            action = random.choice(actions)
            print(action)
            reward = game.make_action(action)
            print("\treward:", reward)
            time.sleep(0.02)
        print("Result:", game.get_total_reward())
        time.sleep(2)
    game.close()

game, possible_actions = create_environment()

# Define the preprocessing functions
"""
Grayscale each of our frames (because color does not add important information ). But this is already done by the config file.
Crop the screen (in our case we remove the roof because it contains no information)
We normalize pixel values
Finally we resize the preprocessed frame
"""
"""
    预处理框架
    preprocess_frame:
    Take a frame.做一个框架
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
    Normalize it.归一化
    return preprocessed_frame返回preprocessed_frame
    """
def preprocess_frame(frame):
    # Greyscale frame already done in our vizdoom config灰度框架已经完成在我们的vizdoom配置
    # x = np.mean(frame,-1)
    # Crop the screen (remove the roof because it contains no information)裁剪屏幕(删除屋顶，因为它不包含任何信息)
    cropped_frame = frame[30:-10, 30:-30]
    # Normalize Pixel Values规范化的像素值
    normalized_frame = cropped_frame / 255.0
    # Resize调整尺寸
    preprocessed_frame = transform.resize(normalized_frame, [84, 84])
    return preprocessed_frame

# stack_frames
"""
Stacking frames is really important because it helps us to give have a sense of motion to our Neural Network.
First we preprocess frame
Then we append the frame to the deque that automatically removes the oldest frame
Finally we build the stacked state
This is how work stack:
For the first frame, we feed 4 frames
At each timestep, we add the new frame to deque and then we stack them to form a new stacked frame
And so on 
"""
# 叠加帧
stack_size = 4  # We stack 4 frames我们堆叠4帧
# Initialize deque with zero-images one array for each image初始化deque与zero-images每个图像一个数组
stacked_frames = deque([np.zeros((84, 84), dtype=np.int) for i in range(stack_size)], maxlen=4)

def stack_frames(stacked_frames, state, is_new_episode):
    # Preprocess frame
    frame = preprocess_frame(state)
    if is_new_episode:
        # Clear our stacked_frames
        stacked_frames = deque([np.zeros((84, 84), dtype=np.int) for i in range(stack_size)], maxlen=4)
        # Because we're in a new episode, copy the same frame 4x因为我们在新一集，复制相同的坐标系4x
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

# Set up our hyperparameters
"""
First, you begin by defining the neural networks hyperparameters when you implement the model.
Then, you'll add the training hyperparameters when you implement the training algorithm.
"""
### MODEL HYPERPARAMETERS模型超参数
state_size = [84,84,4]      # Our input is a stack of 4 frames hence 84x84x4 (Width, height, channels)
action_size = game.get_available_buttons_size()              # 3 possible actions: left, right, shoot
learning_rate =  0.0002     # Alpha (aka learning rate)
### TRAINING HYPERPARAMETERS训练超参数
total_episodes = 500        # Total episodes for training
max_steps = 100             # Max possible steps in an episode
batch_size = 64
# Exploration parameters for epsilon greedy strategy贪心策略的探索参数
explore_start = 1.0         # exploration probability at start开始勘探概率
explore_stop = 0.01         # minimum exploration probability 最小探测概率
decay_rate = 0.0001         # exponential decay rate for exploration prob勘探过程的指数衰减率
# Q learning hyperparameters
gamma = 0.95                # Discounting rate折扣率
### MEMORY HYPERPARAMETERS记忆单元超参数
pretrain_length = batch_size   # Number of experiences stored in the Memory when initialized for the first time第一次初始化时存储在内存中的经验数
memory_size = 1000000          # Number of experiences the Memory can keep记忆可以保留的经验的数量
### MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT如果您只想查看训练过的代理，请将其修改为FALSE
training = True
## TURN THIS TO TRUE IF YOU WANT TO RENDER THE ENVIRONMENT如果您想渲染环境，请将其设置为TRUE
episode_render = False

# Create our Deep Q-learning Neural Network model
"""
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
            # We create the placeholders创建占位符
            # *state_size means that we take each elements of state_size in tuple hence is like if we wrote[None, 84, 84, 4]
            # *state_size的意思是我们把state_size的每个元素都放到tuple中，就像我们写的那样[None, 84, 84, 4]
            self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs")
            self.actions_ = tf.placeholder(tf.float32, [None, 3], name="actions_")
            # Remember that target_Q is the R(s,a) + ymax Qhat(s', a') 记住target_Q是R(s,a) + ymax Qhat(s'， a')
            self.target_Q = tf.placeholder(tf.float32, [None], name="target")
            """
            First convnet:
            CNN
            BatchNormalization
            ELU
            """
            # Input is 84x84x4
            self.conv1 = tf.layers.conv2d(inputs=self.inputs_, filters=32, kernel_size=[8, 8], strides=[4, 4],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv1")
            self.conv1_batchnorm = tf.layers.batch_normalization(self.conv1, training=True, epsilon=1e-5,
                                                                 name='batch_norm1')
            self.conv1_out = tf.nn.relu(self.conv1_batchnorm, name="conv1_out")
            ## --> [20, 20, 32]
            """
            Second convnet:
            CNN
            BatchNormalization
            ELU
            """
            self.conv2 = tf.layers.conv2d(inputs=self.conv1_out, filters=64, kernel_size=[4, 4], strides=[2, 2],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv2")
            self.conv2_batchnorm = tf.layers.batch_normalization(self.conv2, training=True, epsilon=1e-5,
                                                                 name='batch_norm2')
            self.conv2_out = tf.nn.relu(self.conv2_batchnorm, name="conv2_out")
            ## --> [9, 9, 64]
            """
            Third convnet:
            CNN
            BatchNormalization
            ELU
            """
            self.conv3 = tf.layers.conv2d(inputs=self.conv2_out, filters=128, kernel_size=[4, 4], strides=[2, 2],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv3")
            self.conv3_batchnorm = tf.layers.batch_normalization(self.conv3, training=True, epsilon=1e-5,
                                                                 name='batch_norm3')
            self.conv3_out = tf.nn.relu(self.conv3_batchnorm, name="conv3_out")
            ## --> [3, 3, 128]
            self.flatten = tf.layers.flatten(self.conv3_out)
            ## --> [1152]
            self.fc = tf.layers.dense(inputs=self.flatten, units=512, activation=tf.nn.relu,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(), name="fc1")
            self.output = tf.layers.dense(inputs=self.fc,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(), units=3,
                                          activation=None)
            # Q is our predicted Q value.
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)
            # The loss is the difference between our predicted Q_values and the Q_target损失是我们预测的q_值与Q_target之间的差值
            # Sum(Qtarget - Q)^2
            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))  # self.target_Q - self.Q时序差分误差（或TD误差）
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

# Reset the graph
tf.reset_default_graph()
# Instantiate the DQNetwork实例化DQNetwork
DQNetwork = DQNetwork(state_size, action_size, learning_rate)

# Experience Replay
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
Here we'll deal with the empty memory problem: we pre-populate our memory by taking random actions and storing
the experience (state, action, reward, new_state).
"""
# Instantiate memory初始化内存
memory = Memory(max_size=memory_size)
# Render the environment渲染环境
game.new_episode()

for i in range(pretrain_length):
    # If it's the first step如果这是第一步
    if i == 0:
        # First we need a state首先我们需要一个状态
        state = game.get_state().screen_buffer
        state, stacked_frames = stack_frames(stacked_frames, state, True)
    # Random action随机动作
    action = random.choice(possible_actions)
    # Get the rewards得到的回报
    reward = game.make_action(action)
    # Look if the episode is finished看看这一集是否结束了
    done = game.is_episode_finished()
    # If we're dead
    if done:
        # We finished the episode我们结束了这一集
        next_state = np.zeros(state.shape)
        # Add experience to memory给记忆添加经验
        memory.add((state, action, reward, next_state, done))
        # Start a new episode开始新的一集
        game.new_episode()
        # First we need a state
        state = game.get_state().screen_buffer
        # Stack the frames
        state, stacked_frames = stack_frames(stacked_frames, state, True)
    else:
        # Get the next state获得下一个状态
        next_state = game.get_state().screen_buffer
        next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
        # Add experience to memory给记忆添加经验
        memory.add((state, action, reward, next_state, done))
        # Our state is now the next_state我们的状态现在是next_state
        state = next_state

# Set up Tensorboard
# Setup TensorBoard Writer设置TensorBoard写
writer = tf.summary.FileWriter("/tensorboard/dqn/1")
## Losses损失
tf.summary.scalar("Loss", DQNetwork.loss)
write_op = tf.summary.merge_all()

# Train our Agent
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
With ϵ select a random action atat, otherwise select at=argmaxaQ(st,a) ϵ选择一个随机行动at,否则选择在at= argmaxaQ(st,a)
"""
def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, actions):
    ## EPSILON GREEDY STRATEGY ε贪婪策略
    # Choose action a from state s using epsilon greedy.使用贪心从状态s中选择动作a。
    ## First we randomize a number首先我们随机选择一个数字
    exp_exp_tradeoff = np.random.rand()
    # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
    # 这里我们将使用Q-learning笔记本中使用的epsilon贪心策略的改进版本
    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
    if (explore_probability > exp_exp_tradeoff):
        # Make a random action (exploration)
        action = random.choice(possible_actions)
    else:
        # Get action from Q-network (exploitation)从q网络获取行动(利用)
        # Estimate the Qs values state估计Qs值的状态
        Qs = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs_: state.reshape((1, *state.shape))})
        # Take the biggest Q value (= the best action)取最大Q值(=最佳动作)
        choice = np.argmax(Qs)
        action = possible_actions[int(choice)]
    return action, explore_probability

# Saver will help us to save our model Saver将帮助我们保存我们的模型
saver = tf.train.Saver()
if training == True:
    with tf.Session() as sess:
        # Initialize the variables
        sess.run(tf.global_variables_initializer())
        # Initialize the decay rate (that will use to reduce epsilon) 初始化衰减率(用来减小epsilon)
        decay_step = 0
        # Init the game
        game.init()
        for episode in range(total_episodes):
            # Set step to 0
            step = 0
            # Initialize the rewards of the episode
            episode_rewards = []
            # Make a new episode and observe the first state
            game.new_episode()
            state = game.get_state().screen_buffer
            # Remember that stack frame function also call our preprocess function.记住堆栈帧函数也调用预处理函数。
            state, stacked_frames = stack_frames(stacked_frames, state, True)
            while step < max_steps:
                step += 1
                # Increase decay_step
                decay_step += 1
                # Predict the action to take and take it预测要采取的行动，然后采取行动
                action, explore_probability = predict_action(explore_start, explore_stop, decay_rate, decay_step, state,
                                                             possible_actions)
                # Do the action
                reward = game.make_action(action)
                # Look if the episode is finished
                done = game.is_episode_finished()
                # Add the reward to total reward
                episode_rewards.append(reward)
                # If the game is finished
                if done:
                    # the episode ends so no next state
                    next_state = np.zeros((84, 84), dtype=np.int)
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                    # Set step = max_steps to end the episode
                    step = max_steps
                    # Get the total reward of the episode
                    total_reward = np.sum(episode_rewards)
                    print('Episode: {}'.format(episode),
                          'Total reward: {}'.format(total_reward),
                          'Training loss: {:.4f}'.format(loss),
                          'Explore P: {:.4f}'.format(explore_probability))
                    memory.add((state, action, reward, next_state, done))
                else:
                    # Get the next state
                    next_state = game.get_state().screen_buffer
                    # Stack the frame of the next_state
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                    # Add experience to memory
                    memory.add((state, action, reward, next_state, done))
                    # st+1 is now our current state
                    state = next_state
                ### LEARNING PART     学习部分
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
                # 设置Q_target = r如果事件在s+1结束，则设置Q_target = r + gamma*maxQ(s'， a')
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
                writer.flush()  # 强制将缓冲区的数据输出

            # Save model every 5 episodes
            if episode % 5 == 0:
                save_path = saver.save(sess, "./models/model.ckpt")
                print("Model Saved")

# Watch our Agent play
with tf.Session() as sess:
    game, possible_actions = create_environment()
    totalScore = 0
    # Load the model
    saver.restore(sess, "./models/model.ckpt")
    game.init()
    for i in range(1):
        done = False
        game.new_episode()
        state = game.get_state().screen_buffer
        state, stacked_frames = stack_frames(stacked_frames, state, True)
        while not game.is_episode_finished():
            # Take the biggest Q value (= the best action)取最大Q值(=最佳动作)
            Qs = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs_: state.reshape((1, *state.shape))})
            # Take the biggest Q value (= the best action)取最大Q值(=最佳动作)
            choice = np.argmax(Qs)
            action = possible_actions[int(choice)]
            game.make_action(action)
            done = game.is_episode_finished()
            score = game.get_total_reward()
            if done:
                break
            else:
                print("else")
                next_state = game.get_state().screen_buffer
                next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                state = next_state
        score = game.get_total_reward()
        print("Score: ", score)
    game.close()
    
