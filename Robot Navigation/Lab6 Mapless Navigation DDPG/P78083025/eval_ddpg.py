from nav_environment import NavigationEnv
import ddpg
import models
import numpy as np
import os
from PIL import Image
import cv2

def run(total_eps=2, message=False, render=False, map_path="Maps/map.png",\
        model_path="save/", gif_path="out/", gif_name="test.gif"):
    if not os.path.exists(gif_path):
        os.makedirs(gif_path)
    images = []
    RL = ddpg.DDPG(
        model = [models.PolicyNet, models.QNet],
        learning_rate = [0.0001, 0.0001],
        reward_decay = 0.99,
        memory_size = 10000,
        batch_size = 64)

    RL.save_load_model("load", model_path)

    env = NavigationEnv(path=map_path)
    for eps in range(total_eps):
        step = 0
        max_success_rate = 0
        success_count = 0

        state = env.initialize()
        r_eps = []
        acc_reward = 0.
            
        while(True):
            # Choose action and run
            action = RL.choose_action(state, eval=True)
            state_next, reward, done = env.step(action)
            im = env.render(gui=render)
            im_pil = Image.fromarray(cv2.cvtColor(np.uint8(im*255),cv2.COLOR_BGR2RGB))
            images.append(im_pil)

            # Record and print information
            r_eps.append(reward)
            acc_reward += reward
            
            if message:
                print('\rEps: {:2d}| Step: {:4d} | action:{:+.2f}| R:{:+.2f}| Reps:{:.2f}  '\
                        .format(eps, step, action[0], reward, acc_reward), end='')
            
            state = state_next.copy()
            step += 1
            if done or step>600:
                if message:
                    print()
                break

    print("Save evaluation GIF ...")
    images[0].save(gif_path+gif_name,
        save_all=True, append_images=images[1:], optimize=True, duration=40, loop=0)

if __name__ == "__main__":
    run(message=True, render=True)
            