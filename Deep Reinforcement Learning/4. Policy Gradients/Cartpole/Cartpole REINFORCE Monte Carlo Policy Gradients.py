# -*- coding: UTF-8 -*-

# @Thomas Simonini
# https://github.com/simoninithomas/Deep_reinforcement_learning_Course/tree/master/Policy%20Gradients/Cartpole

# Step 1: Import the libraries
import tensorflow as tf
import numpy as np
import gym

# Step 2: Create our environment
env = gym.make('CartPole-v0')
env = env.unwrapped     # 取消限制
# Policy gradient has high variance, seed for reproducability
# 普通的 Policy Gradient 方法, 回合的方差比较大, 所以选一个好点的随机种子
env.seed(1)

print(env.action_space)            # 查看这个环境中可用的 action 有多少个
print(env.observation_space)       # 查看这个环境中 state/observation 有多少个特征值
print(env.observation_space.high)  # 查看 observation 最高取值
print(env.observation_space.low)   # 查看 observation 最低取值

# Step 3: Set up our hyperparameters
## ENVIRONMENT Hyperparameters
state_size = 4
action_size = env.action_space.n

## TRAINING Hyperparameters
max_episodes = 300
learning_rate = 0.01
gamma = 0.95 # Discount rate

# Step 4 : Define the preprocessing functions
def discount_and_normalize_rewards(episode_rewards):
    # np.zeros_like构造一个矩阵discounted_episode_rewards，其维度与episode_rewards一致，并为其初始化为全0；
    # 这个函数方便的构造了新矩阵，无需参数指定shape大小；
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

# Step 5: Create our Policy Gradient Neural Network model
"""
The idea is simple:
    Our state which is an array of 4 values will be used as an input.我们的状态是一个包含4个值的数组，它将被用作输入。
    Our NN is 3 fully connected layers.我们的NN是3个完全连接的层。
    Our output activation function is softmax that squashes the outputs to a
    probability distribution (for instance if we have 4, 2, 6 --> softmax --> (0.4, 0.2, 0.6)
    我们的输出激活函数是softmax，它将输出压缩为a概率分布(比如4 2 6 > softmax >(0。4,0。2,0。6)
"""
# tf.name_scope结合 tf.Variable() 来使用,方便参数命名管理。
with tf.name_scope("inputs"):
    input_ = tf.placeholder(tf.float32, [None, state_size], name="input_")
    actions = tf.placeholder(tf.int32, [None, action_size], name="actions")
    discounted_episode_rewards_ = tf.placeholder(tf.float32, [None, ], name="discounted_episode_rewards")
    # Add this placeholder for having this variable in tensorboard在tensorboard中添加这个变量的占位符
    mean_reward_ = tf.placeholder(tf.float32, name="mean_reward")
    with tf.name_scope("fc1"):
        fc1 = tf.contrib.layers.fully_connected(inputs=input_,
                                                num_outputs=10,
                                                activation_fn=tf.nn.relu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer())
    with tf.name_scope("fc2"):
        fc2 = tf.contrib.layers.fully_connected(inputs=fc1,
                                                num_outputs=action_size,
                                                activation_fn=tf.nn.relu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer())
    with tf.name_scope("fc3"):
        fc3 = tf.contrib.layers.fully_connected(inputs=fc2,
                                                num_outputs=action_size,
                                                activation_fn=None,
                                                weights_initializer=tf.contrib.layers.xavier_initializer())
    with tf.name_scope("softmax"):
        action_distribution = tf.nn.softmax(fc3)

    with tf.name_scope("loss"):
        # tf.nn.softmax_cross_entropy_with_logits computes the cross entropy of the result after applying the softmax function
        # tf.nn。softmax_cross_entropy_with_logits计算 softmax(logits) 和 labels 之间的交叉熵
        # If you have single-class labels, where an object can only belong to one class, you might now consider using
        # 如果您有单类标签，其中一个对象只能属于一个类，那么现在可以考虑使用
        # tf.nn.sparse_softmax_cross_entropy_with_logits so that you don't have to convert your labels to a dense one-hot array.
        # tf.nn。sparse_softmax_cross_entropy_with_logits，这样您就不必将标签转换为密集的单热数组。
        neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=fc3, labels=actions)
        loss = tf.reduce_mean(neg_log_prob * discounted_episode_rewards_)
    with tf.name_scope("train"):
        train_opt = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Step 6: Set up Tensorboard
# Setup TensorBoard Writer
writer = tf.summary.FileWriter("/tensorboard/pg/1")
## Losses
tf.summary.scalar("Loss", loss)
## Reward mean
tf.summary.scalar("Reward_mean", mean_reward_)
write_op = tf.summary.merge_all()

# Step 7: Train our Agent
"""
Create the NN
maxReward = 0       # Keep track of maximum reward
For episode in range(max_episodes):
    episode + 1
    reset environment
    reset stores (states, actions, rewards)
    
    For each step:
        Choose action a
        Perform action a
        Store s, a, r
        If done:
            Calculate sum reward
            Calculate gamma Gt
            Optimize
"""
allRewards = []
total_rewards = 0
maximumRewardRecorded = 0
episode = 0
episode_states, episode_actions, episode_rewards = [], [], []
# 保存模型
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for episode in range(max_episodes):
        episode_rewards_sum = 0
        # Launch the game env.reset()函数用于重置环境，该函数将使得环境的initial observation重置
        state = env.reset()
        env.render()    # env.render()函数用于渲染出当前的智能体以及环境的状态。
        while True:
            # Choose action a, remember WE'RE NOT IN A DETERMINISTIC ENVIRONMENT, WE'RE OUTPUT PROBABILITIES.
            # 选择动作a，记住我们不是在确定性环境中，我们是输出概率。
            action_probability_distribution = sess.run(action_distribution, feed_dict={input_: state.reshape([1, 4])})
            # 将多维数组降为一维，np.ravel()返回的是视图，修改时会影响原始矩阵，默认order="C",按照行进行重组
            action = np.random.choice(range(action_probability_distribution.shape[1]),
                                            p=action_probability_distribution.ravel())  # select action w.r.t the actions prob选择行动w.r。t动作prob
            # Perform a
            # env.step()会返回四个值：observation（object）、reward（float）、done（boolean）、info（dict），其中done表示是否应该reset环境。
            new_state, reward, done, info = env.step(action)
            # Store s, a, r
            episode_states.append(state)
            # For actions because we output only one (the index) we need 2 (1 is for the action taken)
            # 对于操作，因为我们只输出一个(索引)，所以我们需要2(1是所采取的操作)
            # We need [0., 1.] (if we take right) not just the index
            # 我们需要[0。1。(如果我们看对了)不仅仅是指数
            action_ = np.zeros(action_size)
            action_[action] = 1
            episode_actions.append(action_)
            episode_rewards.append(reward)
            if done:
                # Calculate sum reward
                episode_rewards_sum = np.sum(episode_rewards)
                allRewards.append(episode_rewards_sum)
                total_rewards = np.sum(allRewards)
                # Mean reward
                mean_reward = np.divide(total_rewards, episode + 1)
                maximumRewardRecorded = np.amax(allRewards)
                print("==========================================")
                print("Episode: ", episode)
                print("Reward: ", episode_rewards_sum)
                print("Mean Reward", mean_reward)
                print("Max reward so far: ", maximumRewardRecorded)
                # Calculate discounted reward计算折扣奖励
                discounted_episode_rewards = discount_and_normalize_rewards(episode_rewards)
                # Feedforward, gradient and backpropagation前馈，梯度和反向传播
                # np.vstack按行顺序把数组给堆叠起来
                loss_, _ = sess.run([loss, train_opt], feed_dict={input_: np.vstack(np.array(episode_states)),
                                                                  actions: np.vstack(np.array(episode_actions)),
                                                                  discounted_episode_rewards_: discounted_episode_rewards
                                                                  })
                # Write TF Summaries
                summary = sess.run(write_op, feed_dict={input_: np.vstack(np.array(episode_states)),
                                                        actions: np.vstack(np.array(episode_actions)),
                                                        discounted_episode_rewards_: discounted_episode_rewards,
                                                        mean_reward_: mean_reward
                                                        })
                writer.add_summary(summary, episode)
                writer.flush()

                # Reset the transition stores重置转换存储
                episode_states, episode_actions, episode_rewards = [], [], []
                break
            state = new_state
        # Save Model
        if episode % 100 == 0:
            saver.save(sess, "./models/model.ckpt")
            print("Model saved")

with tf.Session() as sess:
    env.reset()
    rewards = []
    # Load the model
    saver.restore(sess, "./models/model.ckpt")
    for episode in range(10):
        state = env.reset()
        step = 0
        done = False
        total_rewards = 0
        print("****************************************************")
        print("EPISODE ", episode)
        while True:
            # Choose action a, remember WE'RE NOT IN A DETERMINISTIC ENVIRONMENT, WE'RE OUTPUT PROBABILITIES.
            # 选择动作a，记住我们不是在确定性环境中，我们是输出概率。
            action_probability_distribution = sess.run(action_distribution, feed_dict={input_: state.reshape([1, 4])})
            # print(action_probability_distribution)
            action = np.random.choice(range(action_probability_distribution.shape[1]),
                                      p=action_probability_distribution.ravel())  # select action w.r.t the actions prob
            new_state, reward, done, info = env.step(action)
            total_rewards += reward
            if done:
                rewards.append(total_rewards)
                print("Score", total_rewards)
                break
            state = new_state
    env.close()
    print("Score over time: " + str(sum(rewards) / 10))
