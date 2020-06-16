from lidar_model import LidarModel
from wmr_model import KinematicModel
import cv2
import numpy as np
from utils import *

img = cv2.flip(cv2.imread("Maps/map.png"),0)
img[img>128] = 255
img[img<=128] = 0
m = np.asarray(img)
m = cv2.cvtColor(m, cv2.COLOR_RGB2GRAY)
m = m.astype(float) / 255.
img = img.astype(float)/255.

lmodel = LidarModel(m)
car = KinematicModel()
pos = (100,200,0)
car.x = pos[0]
car.y = pos[1]
car.yaw = pos[2]

while(True):
    print("\rState: "+car.state_str(), end="\t")
    car.update()
    pos = (car.x, car.y, car.yaw)
    sdata, plist = lmodel.measure_2d(pos)
    img_ = img.copy()
    for pts in plist:
        cv2.line(
            img_, 
            (int(1*pos[0]), int(1*pos[1])), 
            (int(1*pts[0]), int(1*pts[1])),
            (0.0,1.0,0.0), 1)
    
    img_ = car.render(img_)
    img_ = cv2.flip(img_,0)

    #Collision
    p1,p2,p3,p4 = car.car_box
    l1 = Bresenham(p1[0], p2[0], p1[1], p2[1])
    l2 = Bresenham(p2[0], p3[0], p2[1], p3[1])
    l3 = Bresenham(p3[0], p4[0], p3[1], p4[1])
    l4 = Bresenham(p4[0], p1[0], p4[1], p1[1])
    check = l1+l2+l3+l4
    collision = False
    for pts in check:
        if m[int(pts[1]),int(pts[0])]<0.5:
            collision = True
            car.redo()
            car.v = -0.5*car.v
            break

    #cv2.circle(img,(100,200),5,(0.5,0.5,0.5),3)
    cv2.imshow("Lidar Demo",img_)
    k = cv2.waitKey(1)
    if k == ord("a"):
        car.w += 5
    elif k == ord("d"):
        car.w -= 5
    elif k == ord("w"):
        car.v += 4
    elif k == ord("s"):
        car.v -= 4
    elif k == 27:
        print()
        break