from lidar_model import LidarModel
from wmr_model import KinematicModel
import cv2
import numpy as np
from utils import *

class NavigationEnv:
    def __init__(self, path="Maps/map.png"):
        # Read Map
        self.img = cv2.flip(cv2.imread(path),0)
        self.img[self.img>128] = 255
        self.img[self.img<=128] = 0
        self.m = np.asarray(self.img)
        self.m = cv2.cvtColor(self.m, cv2.COLOR_RGB2GRAY)
        self.m = self.m.astype(float) / 255.
        self.img = self.img.astype(float)/255.
        self.lmodel = LidarModel(self.m)

    def initialize(self):
        # Set Mobile Car
        self.car = KinematicModel(d=12, wu=12, wv=4, car_w=18, car_f=14, car_r=10, dt=0.1)
        #self.car = KinematicModel()
        self.car.x, self.car.y = self._search_target()
        self.car.yaw = 360*np.random.random()
        self.pos = (self.car.x, self.car.y, self.car.yaw)

        # Set Navigation Target
        self.target = self._search_target()
        self.target_dist = np.sqrt((self.car.x - self.target[0])**2 + (self.car.y - self.target[1])**2)
        target_orien = np.arctan2(self.target[1]-self.car.y, self.target[0]-self.car.x) - np.deg2rad(self.car.yaw)
        target_rel = [self.target_dist*np.cos(target_orien), self.target_dist*np.sin(target_orien)]

        # Initialize Measurement
        self.sdata, self.plist = self.lmodel.measure_2d(self.pos)
        state = self._construct_state(self.sdata, target_rel)
        return state

    def step(self, action):
        # Control and Update
        self.car.control((action[0]+1)/2*self.car.v_range, action[1]*self.car.w_range)
        self.car.update()
        
        # Collision Handling
        p1,p2,p3,p4 = self.car.car_box
        l1 = Bresenham(p1[0], p2[0], p1[1], p2[1])
        l2 = Bresenham(p1[0], p3[0], p1[1], p3[1])
        l3 = Bresenham(p3[0], p4[0], p3[1], p4[1])
        l4 = Bresenham(p4[0], p2[0], p4[1], p2[1])
        check = l1+l2+l3+l4
        collision = False
        for pts in check:
            if self.m[int(pts[1]),int(pts[0])]<0.5:
                collision = True
                self.car.redo()
                self.car.v = -0.5*self.car.v
                break
        
        # Measure Lidar
        self.pos = (self.car.x, self.car.y, self.car.yaw)
        self.sdata, self.plist = self.lmodel.measure_2d(self.pos)

        # TODO(Lab-01): Reward Design
        # Distance Reward
        curr_target_dist = np.sqrt((self.car.x - self.target[0])**2 + (self.car.y - self.target[1])**2)
        #######################################################################################
        # reward_dist = 0
        reward_dist = self.target_dist - curr_target_dist
        #######################################################################################

        # Orientation Reward
        #######################################################################################
        # reward_dist = 0
        orien = np.rad2deg(np.arctan2(self.target[1] - self.car.y, self.target[0] - self.car.x))
        err_orien = (orien - self.car.yaw) % 360
        if err_orien > 180:
            err_orien = 360 - err_orien
        reward_orien = np.deg2rad(err_orien)
        #######################################################################################

        # Action Panelty
        #######################################################################################
        # reward_act = 0
        reward_act = 0.05 if action[0] < (-0.5) else 0
        #######################################################################################

        # Total
        #######################################################################################
        # reward = 0
        w1 = 4 #1
        w2 = 0.2 #0.5
        w3 = 20 #2 
        reward = w1 * reward_dist - w2 * reward_orien - w3 * reward_act
        #######################################################################################
        # Terminal State 
        done = False
        if collision:
            ####################################################################################
            # reward =
            reward = reward-10
            ####################################################################################
            done = True
        if curr_target_dist < 20:
            ####################################################################################
            # reward =
            reward = reward+10
            ####################################################################################
            done = True

        # Relative Coordinate of Target
        self.target_dist = curr_target_dist
        target_orien = np.arctan2(self.target[1]-self.car.y, self.target[0]-self.car.x) - np.deg2rad(self.car.yaw)
        target_rel = [self.target_dist*np.cos(target_orien), self.target_dist*np.sin(target_orien)] 
        state_next = self._construct_state(self.sdata, target_rel)
        return state_next, reward, done
    
    def render(self, gui=True):
        img_ = self.img.copy()
        for pts in self.plist:
            cv2.line(
                img_, 
                (int(1*self.pos[0]), int(1*self.pos[1])), 
                (int(1*pts[0]), int(1*pts[1])),
                (0.0,1.0,0.0), 1)

        cv2.circle(img_, (int(1*self.target[0]), int(1*self.target[1])), 10, (1.0,0.5,0.7), 3)
        img_ = self.car.render(img_)
        img_ = cv2.flip(img_,0)
        if gui:
            cv2.imshow("Mapless Navigation",img_)
            k = cv2.waitKey(1)
        return img_.copy()
    
    def _search_target(self):
        im_h, im_w = self.m.shape[0], self.m.shape[1]
        tx = np.random.randint(0,im_w)
        ty = np.random.randint(0,im_h)

        kernel = np.ones((10,10),np.uint8)  
        m_dilate = 1-cv2.dilate(1-self.m, kernel, iterations=3)
        while(m_dilate[ty, tx] < 0.5):
            tx = np.random.randint(0,im_w)
            ty = np.random.randint(0,im_h)
        return tx, ty
    
    def _construct_state(self, sensor, target):
        state_s = [s/200 for s in sensor]
        state_t = [t/500 for t in target]
        return state_s + state_t

if __name__ == "__main__":
    env = NavigationEnv()
    for i in range(10):
        env.initialize()
        while(True):
            action = 2*np.random.random(2)-1
            sn, r, end = env.step(action)
            print(str(i) + " : ")
            print(sn[20:23], r, end)
            print(len(sn))
            env.render()
            if end:
                break
