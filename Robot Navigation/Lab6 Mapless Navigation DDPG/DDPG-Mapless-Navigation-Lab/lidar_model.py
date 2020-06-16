import numpy as np
import cv2
import sys
from utils import *

class LidarModel:
    def __init__(self,
            img_map,
            sensor_size = 21,
            start_angle = -120.0,
            end_angle = 120.0,
            max_dist = 200.0,
        ):
        self.sensor_size = sensor_size
        self.start_angle = start_angle
        self.end_angle = end_angle
        self.max_dist = max_dist
        self.img_map = img_map
    
    def measure(self, pos):
        sense_data = []
        inter = (self.end_angle-self.start_angle) / (self.sensor_size-1)
        for i in range(self.sensor_size):
            theta = pos[2] + self.start_angle + i*inter
            sense_data.append(self._ray_cast(np.array((pos[0], pos[1])), theta))
        return sense_data
    
    def measure_2d(self, pos):
        sdata = self.measure(pos)
        plist = EndPoint(pos, [self.sensor_size, self.start_angle, self.end_angle], sdata)
        return sdata, plist

    def _ray_cast(self, pos, theta):
        end = np.array((pos[0] + self.max_dist*np.cos(np.deg2rad(theta)), pos[1] + self.max_dist*np.sin(np.deg2rad(theta))))
        x0, y0 = int(pos[0]), int(pos[1])
        x1, y1 = int(end[0]), int(end[1])
        plist = Bresenham(x0, x1, y0, y1)
        i = 0
        dist = self.max_dist
        for p in plist:
            if p[1] >= self.img_map.shape[0] or p[0] >= self.img_map.shape[1] or p[1]<0 or p[0]<0:
                continue
            if self.img_map[p[1], p[0]] < 0.5:
                tmp = np.power(float(p[0]) - pos[0], 2) + np.power(float(p[1]) - pos[1], 2)
                tmp = np.sqrt(tmp)
                if tmp < dist:
                    dist = tmp
        return dist

if __name__ == "__main__":
    img = cv2.flip(cv2.imread("Maps/map.png"),0)
    img[img>128] = 255
    img[img<=128] = 0
    m = np.asarray(img)
    m = cv2.cvtColor(m, cv2.COLOR_RGB2GRAY)
    m = m.astype(float) / 255.
    img = img.astype(float)/255.

    lmodel = LidarModel(m)
    pos = (100,200,0)
    sdata, plist = lmodel.measure_2d(pos)
    img_ = img.copy()
    for pts in plist:
        cv2.line(
            img_, 
            (int(1*pos[0]), int(1*pos[1])), 
            (int(1*pts[0]), int(1*pts[1])),
            (0.0,1.0,0.0), 1)
    cv2.circle(img_,(pos[0],pos[1]),5,(0.5,0.5,0.5),3)
    img_ = cv2.flip(img_,0)
    cv2.imshow("Lidar Test", img_)
    k = cv2.waitKey(0)