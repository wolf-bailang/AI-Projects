# Doom by using a Dueling Double Deep Q learning architecture with Prioritized Experience Replay.
"""
Our agent playing Doom after 3 hours of training of CPU, remember that our agent needs about 2 days of GPU
to have optimal score, we'll train from beginning to end the most important architectures (PPO with transfer):
"""
# @Thomas Simonini
# https://github.com/simoninithomas/Deep_reinforcement_learning_Course/tree/master/Dueling%20Double%20DQN%20with%20PER%20and%20fixed-q%20targets

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
Note: We have 7 possible actions: turn left, turn right, move left, move right, shoot (attack)...[[0,0,0,0,1]...] 
so we don't need to do one hot encoding 
注意:我们有7种可能的动作:左转，右转，左移，右移，射击(攻击)……所以我们不需要做一个热编码

The purpose of this scenario is to teach the agent to navigate towards his fundamental goal (the vest) and make sure he survives at the same time.
这个场景的目的是教代理导航到他的基本目标(背心)，并确保他在同一时间生存。
Map is a corridor with shooting monsters on both sides (6 monsters in total). 
地图是一个两边都有射击怪物的走廊(一共6个怪物)。
    A green vest is placed at the oposite end of the corridor. 
    一件绿色的背心放在走廊的尽头。
    Reward is proportional (negative or positive) to change of the distance between the player and the vest. 
    奖励与玩家与背心之间的距离成正比(负或正)。
    If player ignores monsters on the sides and runs straight for the vest he will be killed somewhere along the way. 
    如果玩家忽略旁边的怪物，直接跑向背心，他将在途中某处被杀。
    To ensure this behavior doom_skill = 5 (config) is needed.
    要确保这种行为doom_skill = 5 (config)是必需的。

REWARDS:
    +dX for getting closer to the vest. -dX for getting further from the vest.
    +dX表示离背心更近。-dX表示离背心的距离。
    death penalty = 100
    死刑= 100
"""
"""
Here we create our environment
"""
def create_environment():
    game = DoomGame()
    # Load the correct configuration加载正确的配置
    game.load_config("deadly_corridor.cfg")
    # Load the correct scenario (in our case deadly_corridor scenario)加载正确的场景(在我们的例子中是deadly_corridor场景)
    game.set_doom_scenario_path("deadly_corridor.wad")
    game.init()
    # Here we create an hot encoded version of our actions (5 possible actions)在这里，我们创建操作的热编码版本(5个可能的操作)
    # possible_actions = [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]...]
    # np.identity只能创建方形矩阵，返回的是nxn的主对角线为1，其余地方为0的数组
    possible_actions = np.identity(7, dtype=int).tolist()
    return game, possible_actions

game, possible_actions = create_environment()

# Step 3: Define the preprocessing functions
"""
preprocess_frame
Preprocessing is an important step, because we want to reduce the complexity of our states to reduce the
computation time needed for training. 

Our steps:
    Grayscale each of our frames (because color does not add important information ). 
        But this is already done by the config file.
    Crop the screen (in our case we remove the roof because it contains no information)
    We normalize pixel values
    对像素值进行归一化
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
    # Crop the screen (remove part that contains no information)裁剪屏幕(删除不包含任何信息的部分)
    # [Up: Down, Left: right]
    cropped_frame = frame[15:-5, 20:-20]
    # Normalize Pixel Values规范化的像素值
    normalized_frame = cropped_frame / 255.0
    # Resize调整大小
    preprocessed_frame = transform.resize(cropped_frame, [100, 120])
    return preprocessed_frame  # 100x120x1 frame

# stack_frames
"""
Stacking frames is really important because it helps us to give have a sense of motion to our Neural Network.
加帧非常重要，因为它帮助我们给神经网络一个运动的感觉。
    First we preprocess frame
    Then we append the frame to the deque that automatically removes the oldest frame
    然后我们将这个框架附加到deque，它会自动删除最老的框架
    Finally we build the stacked state
    最后我们构建堆栈状态
This is how work stack:
    For the first frame, we feed 4 frames
    对于第一帧，我们输入4帧
    At each timestep, we add the new frame to deque and then we stack them to form a new stacked frame
    在每个时间步中，我们将新框架添加到deque中，然后将它们堆叠起来，形成一个新的堆叠框架
    And so on 
If we're done, we create a new stack with 4 new frames (because we are in a new episode).
"""
stack_size = 4  # We stack 4 frames
# Initialize deque with zero-images one array for each image初始化deque与零图像每个图像一个数组
stacked_frames = deque([np.zeros((100, 120), dtype=np.int) for i in range(stack_size)], maxlen=4)

def stack_frames(stacked_frames, state, is_new_episode):
    # Preprocess frame预处理框架
    frame = preprocess_frame(state)
    if is_new_episode:
        # Clear our stacked_frames明确我们stacked_frames
        stacked_frames = deque([np.zeros((100, 120), dtype=np.int) for i in range(stack_size)], maxlen=4)
        # Because we're in a new episode, copy the same frame 4x因为我们在新一集，复制相同的坐标系4x
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        # Stack the frames
        stacked_state = np.stack(stacked_frames, axis=2)
    else:
        # Append frame to deque, automatically removes the oldest frame添加帧到deque，自动删除最老的帧
        stacked_frames.append(frame)
        # Build the stacked state (first dimension specifies different frames)构建堆栈状态(第一个维度指定不同的帧)
        stacked_state = np.stack(stacked_frames, axis=2)
    return stacked_state, stacked_frames

# Step 4: Set up our hyperparameters
### MODEL HYPERPARAMETERS
state_size = [100,120,4]      # Our input is a stack of 4 frames hence 100x120x4 (Width, height, channels)
action_size = game.get_available_buttons_size()       # 7 possible actions
learning_rate =  0.00025      # Alpha (aka learning rate)

### TRAINING HYPERPARAMETERS
total_episodes = 5000         # Total episodes for training
max_steps = 5000              # Max possible steps in an episode
batch_size = 64

# FIXED Q TARGETS HYPERPARAMETERS 固定的Q目标超参数
max_tau = 10000   # Tau is the C step where we update our target network  Tau是我们更新目标网络的C步骤

# EXPLORATION HYPERPARAMETERS for epsilon greedy strategy探索epsilon贪心策略的超参数
explore_start = 1.0            # exploration probability at start开始勘探概率
explore_stop = 0.01            # minimum exploration probability 最小探测概率
decay_rate = 0.00005           # exponential decay rate for exploration prob勘探过程的指数衰减率

# Q LEARNING hyperparameters
gamma = 0.95               # Discounting rate

### MEMORY HYPERPARAMETERS
## If you have GPU change to 1million如果你有GPU换成100万
pretrain_length = 100000   # Number of experiences stored in the Memory when initialized for the first time第一次初始化时存储在内存中的经验数
memory_size = 100000       # Number of experiences the Memory can keep记忆可以保留的经验的数量

### MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT如果您只想查看训练过的代理，请将其修改为FALSE
training = False

## TURN THIS TO TRUE IF YOU WANT TO RENDER THE ENVIRONMENT如果您想渲染环境，请将其设置为TRUE
episode_render = False

# Step 5: Create our Dueling Double Deep Q-learning Neural Network model (aka DDDQN)
"""
This is our Dueling Double Deep Q-learning model:
    We take a stack of 4 frames as input
    我们取一个4帧的堆栈作为输入
    It passes through 3 convnets
    它通过3个对流网络
    Then it is flatened
    然后变平
    Then it is passed through 2 streams
    然后它通过两个流
        One that calculates V(s)计算V(s)
        The other that calculates A(s,a)另一个计算A(s, A)
    Finally an agregating layer
    最后是agregating层
    It outputs a Q value for each actions
    它为每个操作输出一个Q值
"""
class DDDQNNet:
    def __init__(self, state_size, action_size, learning_rate, name):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.name = name

        # We use tf.variable_scope here to know which network we're using (DQN or target_net)
        # 我们使用tf.variable_scope可以知道我们使用的是哪个网络(DQN还是target_net)
        # it will be useful when we will update our w- parameters (by copy the DQN parameters)
        # 当我们更新w-参数(通过复制DQN参数)时，它将非常有用。
        with tf.variable_scope(self.name):
            # We create the placeholders创建占位符
            # *state_size means that we take each elements of state_size in tuple hence is like if we wrote
            # *state_size的意思是我们把state_size的每个元素都放到tuple中，就像我们写的那样
            # [None, 100, 120, 4]
            self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs")
            self.ISWeights_ = tf.placeholder(tf.float32, [None, 1], name='IS_weights')      # 重要度采样权重
            self.actions_ = tf.placeholder(tf.float32, [None, action_size], name="actions_")

            # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')记住target_Q是R(s,a) + ymax Qhat(s'， a')
            self.target_Q = tf.placeholder(tf.float32, [None], name="target")
            """
            First convnet:
            CNN
            ELU
            """
            # Input is 100x120x4
            self.conv1 = tf.layers.conv2d(inputs=self.inputs_,
                                          filters=32,
                                          kernel_size=[8, 8],
                                          strides=[4, 4],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv1")
            self.conv1_out = tf.nn.relu(self.conv1, name="conv1_out")
            """
            Second convnet:
            CNN
            ELU
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
            ELU
            """
            self.conv3 = tf.layers.conv2d(inputs=self.conv2_out,
                                          filters=128,
                                          kernel_size=[4, 4],
                                          strides=[2, 2],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv3")
            self.conv3_out = tf.nn.relu(self.conv3, name="conv3_out")
            self.flatten = tf.layers.flatten(self.conv3_out)
            ## Here we separate into two streams在这里我们分成两条数据流
            # The one that calculate V(s)计算V(s)的那个
            self.value_fc = tf.layers.dense(inputs=self.flatten,
                                            units=512,
                                            activation=tf.nn.relu,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                            name="value_fc")
            self.value = tf.layers.dense(inputs=self.value_fc,
                                         units=1,
                                         activation=None,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name="value")
            # The one that calculate A(s,a)计算A(s, A)
            self.advantage_fc = tf.layers.dense(inputs=self.flatten,
                                                units=512,
                                                activation=tf.nn.elu,
                                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                name="advantage_fc")
            self.advantage = tf.layers.dense(inputs=self.advantage_fc,
                                             units=self.action_size,
                                             activation=None,
                                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                             name="advantages")
            # Agregating layer  Agregating层
            # Q(s,a) = V(s) + (A(s,a) - 1/|A| * sum A(s,a'))
            # tf.reduce_mean用于计算张量tensor沿着指定的数轴(tensor的某一维度)上的的平均值,主要用作降维或者计算tensor(图像)的平均值
            self.output = self.value + tf.subtract(self.advantage,
                                                   tf.reduce_mean(self.advantage, axis=1, keepdims=True))

            # Q is our predicted Q value. Q是我们预测的Q值。
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)

            # The loss is modified because of PER 这个损失因为PER而被修正了
            self.absolute_errors = tf.abs(self.target_Q - self.Q)  # for updating Sumtree更新Sumtree
            # tf.squared_difference指定的维度进行均方误差的计算
            self.loss = tf.reduce_mean(self.ISWeights_ * tf.squared_difference(self.target_Q, self.Q))
            # RMSprop优化器，最小loss函数
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

# Reset the graph
tf.reset_default_graph()
# Instantiate the DQNetwork实例化DQNetwork
DQNetwork = DDDQNNet(state_size, action_size, learning_rate, name="DQNetwork")
# Instantiate the target network实例化目标网络
TargetNetwork = DDDQNNet(state_size, action_size, learning_rate, name="TargetNetwork")

# Step 6: Prioritized Experience Replay
"""
we can't use a simple array to do that because sampling from it will be not efficient, so we use a binary
tree data type (in a binary tree each node has no + than 2 children).
我们不能使用一个简单的数组来实现这一点，因为从它进行采样效率不高，所以我们使用一个二叉树数据类型(在二叉树中，每个节点的子节点不超过2个子节点)。

Step 1: We construct a SumTree, which is a Binary Sum tree where leaves contains the priorities and a data
array where index points to the index of leaves.  
构造一个SumTree，它是一个二进制和树，其中叶子包含优先级，一个数据数组，其中索引指向叶子的索引。
    def init: Initialize our SumTree data object with all nodes = 0 and data (data array) with all = 0.
                用所有节点= 0初始化SumTree数据对象，用所有节点= 0初始化data (data array)。
    def add: add our priority score in the sumtree leaf and experience (S, A, R, S', Done) in data.
                在sumtree叶子中添加我们的优先级得分，并在数据中添加经验(S, A, R, S'， Done)。
    def update: we update the leaf priority score and propagate through tree.
                更新叶优先级得分并在树中传播。
    def get_leaf: retrieve priority score, index and experience associated with a leaf.
                检索与叶关联的优先级得分、索引和经验。
    def total_priority: get the root node value to calculate the total priority score of our replay buffer.
                获取根节点值，以计算重放缓冲区的总优先级得分。
Step 2: We create a Memory object that will contain our sumtree and data.
        创建一个内存对象，它将包含我们的sumtree和数据。
    def init: generates our sumtree and data by instantiating the SumTree object.
                通过实例化sumtree对象生成我们的sumtree和数据。
    def store: we store a new experience in our tree. Each new experience will have priority = max_priority and
    then this priority will be corrected during the training (when we'll calculating the TD error hence the priority score).
    我们把新的体验储存在我们的树上。每个新体验都有priority = max_priority和然后这个优先级将在训练期间被纠正(当我们计算TD错误时，因此优先级得分)。
    def sample:
        First, to sample a minibatch of k size, the range [0, priority_total] is / into k ranges.
        首先，要对k个大小的小批进行采样，范围[0,priority_total]是/到k个范围。
        Then a value is uniformly sampled from each range
        然后从每个范围均匀采样一个值
        We search in the sumtree, the experience where priority score correspond to sample values are retrieved from.
        我们在sumtree中搜索，检索优先级得分对应于样本值的体验。
        Then, we calculate IS weights for each minibatch element
        然后，我们计算每个小批元素的权重
    def update_batch: update the priorities on the tree  
                更新树上的优先级
"""
class SumTree(object):
    """
    This SumTree code is modified version of Morvan Zhou:
    https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5.2_Prioritized_Replay_DQN/RL_brain.py
    """
    data_pointer = 0

    """
    Here we initialize the tree with all nodes = 0, and initialize the data with all values = 0
    在这里，我们使用所有节点= 0初始化树，并使用所有值= 0初始化数据
    """
    def __init__(self, capacity):
        self.capacity = capacity  # Number of leaf nodes (final nodes) that contains experiences包含经验的叶节点(最终节点)的数量
        # Generate the tree with all nodes values = 0生成所有节点值= 0的树
        # To understand this calculation (2 * capacity - 1) look at the schema above要理解这个计算(2 *容量- 1)，请查看上面的模式
        # Remember we are in a binary node (each node has max 2 children) so 2x size of leaf (capacity) - 1 (root node)
        # 记住，我们是在二进制节点中(每个节点最多有2个子节点)所以叶节点的大小(容量)是2x - 1(根节点)
        # Parent nodes = capacity - 1   父节点 = 容量 - 1
        # Leaf nodes = capacityity    叶节点 = 容量
        self.tree = np.zeros(2 * capacity - 1)
        """ tree:
            0
           / \
          0   0
         / \ / \
        0  0 0  0  [Size: capacity] it's at this line that there is the priorities score (aka pi)
        [Size: capacity]在这条线上有优先级得分(即pi)
        """
        # Contains the experiences (so the size of data is capacity)包含经验(因此数据的大小就是容量)
        self.data = np.zeros(capacity, dtype=object)

    """
    Here we add our priority score in the sumtree leaf and add the experience in data
    在这里，我们在sumtree叶子中添加优先级得分，并在数据中添加体验
    """
    def add(self, priority, data):
        # Look at what index we want to put the experience看看我们想把体验放在什么索引里
        tree_index = self.data_pointer + self.capacity - 1
        """ tree:
            0
           / \
          0   0
         / \ / \
tree_index  0 0  0  We fill the leaves from left to right我们把叶子从左到右填满
        """
        # Update data frame更新数据帧
        self.data[self.data_pointer] = data
        # Update the leaf更新叶子
        self.update(tree_index, priority)
        # Add 1 to data_pointer
        self.data_pointer += 1
        # If we're above the capacity, you go back to first index (we overwrite)如果超出容量，就回到第一个索引(我们覆盖)
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    """
    Update the leaf priority score and propagate the change through tree更新叶优先级得分并在树中传播更改
    """
    def update(self, tree_index, priority):
        # Change = new priority score - former priority score更改 = 新的优先级分数 - 以前的优先级分数
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        # then propagate the change through tree然后通过树传播更改
        while tree_index != 0:  # this method is faster than the recursive loop in the reference code此方法比引用代码中的递归循环快
            """
            Here we want to access the line above这里我们要访问上面的这条线
            THE NUMBERS IN THIS TREE ARE THE INDEXES NOT THE PRIORITY VALUES这个树中的数字是索引而不是优先级值
                0
               / \
              1   2
             / \ / \
            3  4 5  [6] 
            If we are in leaf at index 6, we updated the priority score如果我们在索引6的leaf中，我们更新了优先级得分
            We need then to update index 2 node然后我们需要更新索引2节点
            So tree_index = (tree_index - 1) // 2所以tree_index = (tree_index - 1) // 2
            tree_index = (6-1)//2
            tree_index = 2 (because // round the result)  tree_index = 2(因为//将结果四舍五入)
            """
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    """
    Here we get the leaf_index, priority value of that leaf and experience associated with that index
    这里我们得到了leaf的索引，它的优先级值以及与该索引相关联的体验
    """
    def get_leaf(self, v):
        """
        Tree structure and array storage:树结构和数组存储
        Tree index:
             0         -> storing priority sum存储优先级和
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for experiences存储经验优先级
        Array type for storing:存储数组类型
        [0,1,2,3,4,5,6]
        """
        parent_index = 0
        while True:  # the while loop is faster than the method in the reference code  while循环比引用代码中的方法快
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1
            # If we reach bottom, end the search如果我们到达底部，结束搜索
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            else:  # downward search, always search for a higher priority node向下搜索，始终搜索优先级更高的节点
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index
        data_index = leaf_index - self.capacity + 1
        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @property   # 装饰器就是负责把一个方法变成属性调用的
    def total_priority(self):
        return self.tree[0]  # Returns the root node返回根节点

# Here we don't use deque anymore
# stored as ( s, a, r, s_ ) in SumTree 以(s, a, r, s_)的形式存储在SumTree中
class Memory(object):
    """
    This SumTree code is modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    # Hyperparameter that we use to avoid some experiences to have 0 probability of being taken
    # 超参数，我们用来避免一些经验有0的概率被采取
    PER_e = 0.01
    # Hyperparameter that we use to make a tradeoff between taking only exp with high priority and sampling randomly
    # 超参数，我们用来权衡只取高优先级的exp和随机抽样
    PER_a = 0.6
    PER_b = 0.4  # importance-sampling, from initial value increasing to 1重要性抽样，从初值增加到1
    PER_b_increment_per_sampling = 0.001
    absolute_error_upper = 1.  # clipped abs error剪除abs错误

    def __init__(self, capacity):
        # Making the tree
        """
        Remember that our tree is composed of a sum tree that contains the priority scores at his leaf
        请记住，我们的树由一个sum树组成，其中包含其叶子上的优先级得分
        And also a data array还有一个数据数组
        We don't use deque because it means that at each timestep our experiences change index by one.
        我们不使用deque，因为它意味着每一步我们的体验都会改变一个指标。
        We prefer to use a simple array and to overwrite when the memory is full.
        我们更喜欢使用简单的数组，并在内存已满时重写。
        """
        self.tree = SumTree(capacity)

    """
    Store a new experience in our tree在我们的树上储存新的体验
    Each new experience have a score of max_prority (it will be then improved when we use this exp to train our DDQN)
    每个新体验都有一个max_prority(当我们使用这个exp来培训我们的DDQN时，它将得到改善)
    """
    def store(self, experience):
        # Find the max priority找到最大优先级
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])
        # If the max priority = 0 we can't put priority = 0 since this exp will never have a chance to be selected
        # 如果最大优先级= 0，我们不能让优先级= 0，因为这个exp将永远没有机会被选中
        # So we use a minimum priority所以我们使用最小优先级
        if max_priority == 0:
            max_priority = self.absolute_error_upper
        self.tree.add(max_priority, experience)  # set the max p for new p设置新p的最大值p

    """
    - First, to sample a minibatch of k size, the range [0, priority_total] is / into k ranges.
    首先，要对k个大小的小批进行采样，范围[0,priority_total]是/到k个范围。
    - Then a value is uniformly sampled from each range
    然后从每个范围均匀采样一个值
    - We search in the sumtree, the experience where priority score correspond to sample values are retrieved from.
    我们在sumtree中搜索，检索优先级得分对应于样本值的体验。
    - Then, we calculate IS weights for each minibatch element
    然后，我们计算每个小批元素的权重
    """
    def sample(self, n):
        # Create a sample array that will contains the minibatch创建一个包含小批处理的示例数组
        memory_b = []
        # empty创建空矩阵（实际有值）
        b_idx, b_ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, 1), dtype=np.float32)
        # Calculate the priority segment计算优先级段
        # Here, as explained in the paper, we divide the Range[0, ptotal] into n ranges
        # 在这里，正如本文所解释的，我们将范围[0,ptotal]划分为n个范围
        priority_segment = self.tree.total_priority / n  # priority segment优先级段
        # Here we increasing the PER_b each time we sample a new minibatch在这里，每次对一个新的小批进行抽样时，我们都会增加PER_b
        self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sampling])  # max = 1
        # Calculating the max_weight计算max_weight
        p_min = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_priority
        max_weight = (p_min * n) ** (-self.PER_b)
        for i in range(n):
            # A value is uniformly sample from each range值是从每个范围均匀采样的
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)
            # Experience that correspond to each value is retrieved检索与每个值对应的经验
            index, priority, data = self.tree.get_leaf(value)
            # P(j)
            sampling_probabilities = priority / self.tree.total_priority
            #  IS = (1/N * 1/P(i))**b /max wi == (N*P(i))**-b  /max wi
            b_ISWeights[i, 0] = np.power(n * sampling_probabilities, -self.PER_b) / max_weight
            b_idx[i] = index
            experience = [data]
            memory_b.append(experience)
        return b_idx, memory_b, b_ISWeights

    """
    Update the priorities on the tree更新树上的优先级
    """
    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.PER_e  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.PER_a)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

# Here we'll deal with the empty memory problem: we pre-populate our memory by taking random actions and storing the experience.
# 将处理空内存问题:我们通过随机操作和存储体验来预填充内存。
# Instantiate memory初始化内存
memory = Memory(memory_size)
# Render the environment渲染环境
game.new_episode()
for i in range(pretrain_length):
    # If it's the first step
    if i == 0:
        # First we need a state
        state = game.get_state().screen_buffer
        state, stacked_frames = stack_frames(stacked_frames, state, True)
    # Random action随机动作
    action = random.choice(possible_actions)
    # Get the rewards
    reward = game.make_action(action)
    # Look if the episode is finished看看这一集是否结束了
    done = game.is_episode_finished()
    # If we're dead
    if done:
        # We finished the episode结束了这一集
        next_state = np.zeros(state.shape)
        # Add experience to memory给记忆添加经验
        # experience = np.hstack((state, [action, reward], next_state, done))
        experience = state, action, reward, next_state, done
        memory.store(experience)
        # Start a new episode
        game.new_episode()
        # First we need a state
        state = game.get_state().screen_buffer
        # Stack the frames
        state, stacked_frames = stack_frames(stacked_frames, state, True)
    else:
        # Get the next state
        next_state = game.get_state().screen_buffer
        next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
        # Add experience to memory
        experience = state, action, reward, next_state, done
        memory.store(experience)
        # Our state is now the next_state
        state = next_state

# Step 7: Set up Tensorboard
# Setup TensorBoard Writer
writer = tf.summary.FileWriter("/tensorboard/dddqn/1")
## Losses
tf.summary.scalar("Loss", DQNetwork.loss)
write_op = tf.summary.merge_all()

# Step 8: Train our Agent
"""
Our algorithm: 
    Initialize the weights for DQN初始化DQN的权重
    Initialize target value weights w- <- w初始化目标值权值w- <- w
    Init the environment
    Initialize the decay rate (that will use to reduce epsilon) 初始化衰减率(用来减小epsilon)

    For episode to max_episode do 从episode到max_episode、
        Make new episode
        Set step to 0
        Observe the first state s0

        While step < max_steps do:
        Increase decay_rate增加decay_rate
        With ϵ select a random action at, otherwise select at=argmaxaQ(st,a)ϵ选择一个随机的行动,否则选择at= argmaxaQ(st,a)
        Execute action at in simulator and observe reward rt+1and new state st+1在模拟器中执行动作，观察奖励rt+1和新的状态st+1
        Store transition <st,at,rt+1,st+1> in memory D
        Sample random mini-batch from D: <s,a,r,s′>
        Set target Q^=r if the episode ends at +1, otherwise set Q^=r+γQ(s′,argmaxa′Q(s′,a′,w),w−)
        Make a gradient descent step with loss (Q^−Q(s,a))2做一个梯度下降法与损失(Q^−Q (s,a)) 2
        Every C steps, reset: w−←w每一个步骤,重置:w−←w
    endfor 
endfor
"""
"""
This function will do the part
With ϵ select a random action at, otherwise select at=argmaxaQ(st,a)
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
        # Make a random action (exploration)随机行动(探索)
        action = random.choice(possible_actions)
    else:
        # Get action from Q-network (exploitation)从q网络获取行动(利用)
        # Estimate the Qs values state估计Qs值的状态
        Qs = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs_: state.reshape((1, *state.shape))})
        # Take the biggest Q value (= the best action)取最大Q值(=最佳动作)
        choice = np.argmax(Qs)
        action = possible_actions[int(choice)]
    return action, explore_probability

# This function helps us to copy one set of variables to another
# 这个函数帮助我们把一组变量复制到另一组
# In our case we use it when we want to copy the parameters of DQN to Target_network
# 在我们的例子中，当我们想要将DQN的参数复制到Target_network时，我们使用它
# Thanks of the very good implementation of Arthur Juliani https://github.com/awjuliani
def update_target_graph():
    # Get the parameters of our DQNNetwork获取DQNNetwork的参数
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "DQNetwork")
    # Get the parameters of our Target_network获取Target_network的参数
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "TargetNetwork")
    op_holder = []
    # Update our target_network parameters with DQNNetwork parameters使用DQNNetwork参数更新target_network参数
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

# Saver will help us to save our model
saver = tf.train.Saver()
if training == True:
    with tf.Session() as sess:
        # Initialize the variables初始化变量
        sess.run(tf.global_variables_initializer())
        # Initialize the decay rate (that will use to reduce epsilon)初始化衰减率(用来减小epsilon)
        decay_step = 0
        # Set tau = 0
        tau = 0
        # Init the game
        game.init()
        # Update the parameters of our TargetNetwork with DQN_weights使用DQN_weights更新TargetNetwork的参数
        update_target = update_target_graph()
        sess.run(update_target)
        for episode in range(total_episodes):
            # Set step to 0
            step = 0
            # Initialize the rewards of the episode初始化事件的奖励
            episode_rewards = []
            # Make a new episode and observe the first state制作一个新片段，观察第一种状态
            game.new_episode()
            state = game.get_state().screen_buffer
            # Remember that stack frame function also call our preprocess function.记住堆栈帧函数也调用预处理函数。
            state, stacked_frames = stack_frames(stacked_frames, state, True)
            while step < max_steps:
                step += 1
                # Increase the C step
                tau += 1
                # Increase decay_step
                decay_step += 1
                # With ϵ select a random action at, otherwise select a = argmaxQ(st,a)
                action, explore_probability = predict_action(explore_start, explore_stop, decay_rate, decay_step, state,
                                                             possible_actions)
                # Do the action
                reward = game.make_action(action)
                # Look if the episode is finished
                done = game.is_episode_finished()
                # Add the reward to total reward将奖励加到总奖励中
                episode_rewards.append(reward)
                # If the game is finished
                if done:
                    # the episode ends so no next state这一集结束了，所以没有下一个state
                    next_state = np.zeros((120, 140), dtype=np.int)
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                    # Set step = max_steps to end the episode
                    step = max_steps
                    # Get the total reward of the episode获得该集的总奖励
                    total_reward = np.sum(episode_rewards)
                    print('Episode: {}'.format(episode),
                          'Total reward: {}'.format(total_reward),
                          'Training loss: {:.4f}'.format(loss),
                          'Explore P: {:.4f}'.format(explore_probability))
                    # Add experience to memory给记忆添加经验
                    experience = state, action, reward, next_state, done
                    memory.store(experience)
                else:
                    # Get the next state
                    next_state = game.get_state().screen_buffer
                    # Stack the frame of the next_state
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                    # Add experience to memory
                    experience = state, action, reward, next_state, done
                    memory.store(experience)
                    # st+1 is now our current state
                    state = next_state
                ### LEARNING PART学习部分
                # Obtain random mini-batch from memory从内存中获取随机小批处理
                tree_idx, batch, ISWeights_mb = memory.sample(batch_size)
                states_mb = np.array([each[0][0] for each in batch], ndmin=3)
                actions_mb = np.array([each[0][1] for each in batch])
                rewards_mb = np.array([each[0][2] for each in batch])
                next_states_mb = np.array([each[0][3] for each in batch], ndmin=3)
                dones_mb = np.array([each[0][4] for each in batch])
                target_Qs_batch = []

                ### DOUBLE DQN Logic
                # Use DQNNetwork to select the action to take at next_state (a') (action with the highest Q-value)
                # 使用DQNNetwork选择在next_state(a')处要采取的动作(q值最高的动作)
                # Use TargetNetwork to calculate the Q_val of Q(s',a')
                # 使用TargetNetwork计算Q(s'，a')的Q_val
                # Get Q values for next_state获取next_state的Q值
                q_next_state = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs_: next_states_mb})
                # Calculate Qtarget for all actions that state为所有状态的操作计算Qtarget
                q_target_next_state = sess.run(TargetNetwork.output, feed_dict={TargetNetwork.inputs_: next_states_mb})
                # Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gamma * Qtarget(s',a')
                for i in range(0, len(batch)):
                    terminal = dones_mb[i]
                    # We got a'
                    action = np.argmax(q_next_state[i])
                    # If we are in a terminal state, only equals reward如果我们处于终极状态，就只等于奖励
                    if terminal:
                        target_Qs_batch.append(rewards_mb[i])
                    else:
                        # Take the Qtarget for action a'采取行动的Qtarget a'
                        target = rewards_mb[i] + gamma * q_target_next_state[i][action]
                        target_Qs_batch.append(target)
                targets_mb = np.array([each for each in target_Qs_batch])
                _, loss, absolute_errors = sess.run([DQNetwork.optimizer, DQNetwork.loss, DQNetwork.absolute_errors],
                                                    feed_dict={DQNetwork.inputs_: states_mb,
                                                               DQNetwork.target_Q: targets_mb,
                                                               DQNetwork.actions_: actions_mb,
                                                               DQNetwork.ISWeights_: ISWeights_mb})
                # Update priority更新优先级
                memory.batch_update(tree_idx, absolute_errors)
                # Write TF Summaries
                summary = sess.run(write_op, feed_dict={DQNetwork.inputs_: states_mb,
                                                        DQNetwork.target_Q: targets_mb,
                                                        DQNetwork.actions_: actions_mb,
                                                        DQNetwork.ISWeights_: ISWeights_mb})
                writer.add_summary(summary, episode)
                writer.flush()
                if tau > max_tau:
                    # Update the parameters of our TargetNetwork with DQN_weights使用DQN_weights更新TargetNetwork的参数
                    update_target = update_target_graph()
                    sess.run(update_target)
                    tau = 0
                    print("Model updated")
            # Save model every 5 episodes
            if episode % 5 == 0:
                save_path = saver.save(sess, "./models/model.ckpt")
                print("Model Saved")

# Step 9: Watch our Agent play
with tf.Session() as sess:
    game = DoomGame()
    # Load the correct configuration (TESTING)加载正确的配置(测试)
    game.load_config("deadly_corridor_testing.cfg")
    # Load the correct scenario (in our case deadly_corridor scenario)加载正确的场景(在我们的例子中是deadly_corridor场景)
    game.set_doom_scenario_path("deadly_corridor.wad")
    game.init()
    # Load the model
    saver.restore(sess, "./models/model.ckpt")
    game.init()
    for i in range(10):
        game.new_episode()
        state = game.get_state().screen_buffer
        state, stacked_frames = stack_frames(stacked_frames, state, True)
        while not game.is_episode_finished():
            ## EPSILON GREEDY STRATEGYε贪婪策略
            # Choose action a from state s using epsilon greedy.使用贪心从状态s中选择动作a。
            ## First we randomize a number首先我们随机选择一个数字
            exp_exp_tradeoff = np.random.rand()
            explore_probability = 0.01
            if (explore_probability > exp_exp_tradeoff):
                # Make a random action (exploration)随机行动(探索)
                action = random.choice(possible_actions)
            else:
                # Get action from Q-network (exploitation)从q网络获取行动(利用)
                # Estimate the Qs values state估计Qs值的状态
                Qs = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs_: state.reshape((1, *state.shape))})
                # Take the biggest Q value (= the best action)取最大Q值(=最佳动作)
                choice = np.argmax(Qs)
                action = possible_actions[int(choice)]
            game.make_action(action)
            done = game.is_episode_finished()
            if done:
                break
            else:
                next_state = game.get_state().screen_buffer
                next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                state = next_state
        score = game.get_total_reward()
        print("Score: ", score)
    game.close()
