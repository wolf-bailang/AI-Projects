from nav_environment import NavigationEnv
import ddpg
import models
import numpy as np
import os
import eval_ddpg

batch_size = 64
eval_eps = 50
RL = ddpg.DDPG(
    model = [models.PolicyNet, models.QNet],
    learning_rate = [0.0001, 0.0001],
    reward_decay = 0.99,
    memory_size = 10000,
    batch_size = batch_size)
'''
is_train = True
render = False
load_model = False
'''
is_train = False
render = True
load_model = True

gif_path = "out/"
model_path = "save/"
if not os.path.exists(model_path):
    os.makedirs(model_path)

if load_model:
    print("Load model ...", model_path)
    RL.save_load_model("load", model_path)

if __name__ == "__main__":
    env = NavigationEnv()
    total_step = 0
    max_success_rate = 0
    success_count = 0
    for eps in range(1000):
        state = env.initialize()
        step = 0
        r_eps = []
        loss_a = loss_c = 0.
        acc_reward = 0.
        
        while(True):
            # Choose action and run
            if is_train:
                action = RL.choose_action(state, eval=False)
            else:
                action = RL.choose_action(state, eval=True)
            state_next, reward, done = env.step(action)
            end = 0 if done else 1
            RL.store_transition(state, action, reward, state_next, end)

            # Render Environment
            im = env.render(gui=render)

            # Record and print information
            r_eps.append(reward)
            acc_reward += reward
            loss_a = loss_c = 0.
            if total_step > batch_size and is_train:
                loss_a, loss_c = RL.learn()

            print('\rEps:{:3d} /{:4d} /{:6d}| action:{:+.2f}| R:{:+.2f}| Loss:[A>{:+.2f} C>{:+.2f}]| Epsilon: {:.3f}| Reps:{:.2f}  '\
                    .format(eps, step, total_step, action[0], reward, loss_a, loss_c, RL.epsilon, acc_reward), end='')
            state = state_next.copy()
            step += 1
            total_step += 1
            if done or step>600:
                print()
                break

        if r_eps[-1] > 2:
            success_count += 1 

        if eps>0 and eps%eval_eps==0:
            # Sucess rate
            success_rate = success_count / eval_eps
            success_count = 0
            
            if success_rate >= max_success_rate:
                max_success_rate = success_rate
                if is_train:
                    print("Save model to " + model_path)
                    RL.save_load_model("save", model_path)

            print("Success Rate:", success_rate, "/", max_success_rate)

            # output GIF
            print("Generate GIF ...")
            eval_ddpg.run(total_eps=4, model_path=model_path, \
                gif_path=gif_path, gif_name=str(eps).zfill(4)+"_eps.gif")
            