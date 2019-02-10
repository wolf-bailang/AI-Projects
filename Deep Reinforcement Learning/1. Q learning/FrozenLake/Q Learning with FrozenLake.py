# Q* Learning with FrozenLake

# @Thomas Simonini
# https://github.com/simoninithomas/Deep_reinforcement_learning_Course/tree/master/Q%20learning/FrozenLake

import numpy as np
import gym
import random

# Create the environment
# 创建环境
env = gym.make("FrozenLake-v0")

# Create the Q-table and initialize it
# 动作大小
action_size = env.action_space.n
# 状态大小
state_size = env.observation_space.n
# 创建Q表
qtable = np.zeros((state_size, action_size))
print(qtable)

# Create the hyperparameters
# 定义超参数
total_episodes = 15000        # Total episodes
learning_rate = 0.8           # Learning rate
max_steps = 99                # Max steps per episode
gamma = 0.95                  # Discounting rate

# Exploration parameters
epsilon = 1.0                 # Exploration rate
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.01            # Minimum exploration probability
decay_rate = 0.005            # Exponential decay rate for exploration prob

# Q learning algorithm
# List of rewards
rewards = []
# 2 For life or until learning is stopped
for episode in range(total_episodes):
    # Reset the environment加载环境
    state = env.reset()
    step = 0
    done = False
    total_rewards = 0

    for step in range(max_steps):
        # 3. Choose an action a in the current world state (s)
        ## First we randomize a number
        exp_exp_tradeoff = random.uniform(0, 1)  # 用于生成一个指定范围内的随机浮点数,两格参数中,其中一个是上限,一个是下限。如果a>b,则生成的随机数n,即b<=n<=a

        ## If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
        # 贪心算法
        if exp_exp_tradeoff > epsilon:
            action = np.argmax(qtable[state, :])
        # Else doing a random choice --> exploration
        else:
            action = env.action_space.sample()
        # Take the action (a) and observe the outcome state(s') and reward (r)
        # 转移概率并没有显式表示出来，而是通过 env.step()的结果表示。env.step()返回的 observation满足转移概率。
        # env.step()返回四个值：observation（object）、reward（float）、done（boolean）、info（dict），其中done表示是否应该reset环境。
        new_state, reward, done, info = env.step(action)
        # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        # qtable[new_state,:] : all the actions we can take from new state
        qtable[state, action] = qtable[state, action] + learning_rate * (
                    reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])

        total_rewards += reward
        # Our new state is state
        state = new_state
        # If done (if we're dead) : finish episode
        if done == True:
            break
    # Reduce epsilon (because we need less and less exploration)
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
    rewards.append(total_rewards)

print("Score over time: " + str(sum(rewards) / total_episodes))
print(qtable)

# Use our Q-table to play FrozenLake
env.reset()

for episode in range(5):
    state = env.reset()
    step = 0
    done = False
    print("****************************************************")
    print("EPISODE ", episode)

    for step in range(max_steps):
        # Take the action (index) that have the maximum expected future reward given that state
        action = np.argmax(qtable[state, :])
        new_state, reward, done, info = env.step(action)
        if done:
            # Here, we decide to only print the last state (to see if our agent is on the goal or fall into an hole)
            env.render()  # 用于渲染出当前的智能体以及环境的状态
            # We print the number of step it took.
            print("Number of steps", step)
            break
        state = new_state
env.close()
