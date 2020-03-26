import cv2
import numpy as np
from utils import *

class RRT():
    def __init__(self,m):
        self.map = m

    def _distance(self, n1, n2):
        d = np.array(n1) - np.array(n2)
        return np.hypot(d[0], d[1])

    def _random_node(self, goal, shape):
        r = np.random.choice(2,1,p=[0.5,0.5])
        if r==1:
            return (float(goal[0]), float(goal[1]))
        else:
            rx = float(np.random.randint(int(shape[1])))
            ry = float(np.random.randint(int(shape[0])))
            return (rx, ry)

    def _nearest_node(self, samp_node):
        min_dist = 99999
        min_node = None
        for n in self.ntree:
            # 计算各点与采样点的距离
            dist = self._distance(n, samp_node)
            if dist < min_dist:
                min_dist = dist
                min_node = n
        return min_node

    def _check_collision(self, n1, n2):
        n1_ = pos_int(n1)
        n2_ = pos_int(n2)
        line = Bresenham(n1_[0], n2_[0], n1_[1], n2_[1])
        for pts in line:
            if self.map[int(pts[1]),int(pts[0])]<0.5:
                return True
        return False

    def _steer(self, from_node, to_node, extend_len):
        ## 从距离最小点向采样点移动 step_size 距离，并进行碰撞检测
        vect = np.array(to_node) - np.array(from_node)
        v_len = np.hypot(vect[0], vect[1])
        v_theta = np.arctan2(vect[1], vect[0])
        # at least extend_len
        if extend_len > v_len:
            extend_len = v_len
        new_node = (from_node[0]+extend_len*np.cos(v_theta), from_node[1]+extend_len*np.sin(v_theta))
        # todo
        ####################################################################################################################################################
        # this "if-statement" is not complete, you need complete this "if-statement"
        # you need to check the path is legal or illegal, you can use the function "self._check_collision"
        # illegal
        if new_node[1]<0 or new_node[1]>=self.map.shape[0] or new_node[0]<0 or new_node[0]>=self.map.shape[1] or self._check_collision(from_node, new_node):
        ####################################################################################################################################################
            return False, None
        # legal
        else:        
            return new_node, self._distance(new_node, from_node)

    def planning(self, start, goal, extend_lens, img=None):
        self.ntree = {}
        self.ntree[start] = None
        self.cost = {}
        self.cost[start] = 0
        goal_node = None
        for it in range(20000):
            print("\r", it, len(self.ntree), end="")
            ## 随机生长采样点
            samp_node = self._random_node(goal, self.map.shape)
            ## 选择rrt树中离采样点最近的点
            near_node = self._nearest_node(samp_node)
            new_node, cost = self._steer(near_node, samp_node, extend_lens)
            if new_node is not False:
                # todo
                # After you add a node, you need to maintain the tree.
                # In the first line you need to assign it’s parent.
                # In the second line you need to calculate the new node’s cost.
                ###################################################################
                # after creat a new node in a tree, we need to maintain something
                # 为新点加入父节点索引
                self.ntree[new_node] = near_node
                self.cost[new_node] = cost
                ###################################################################
            else:
                continue
            # 距离阈值，小于此值将被视作同一个点，不可大于 step_size
            if self._distance(near_node, goal) < extend_lens:
                goal_node = near_node
                break
        

            # Draw
            if img is not None:
                for n in self.ntree:
                    if self.ntree[n] is None:
                        continue
                    node = self.ntree[n]
                    cv2.line(img, (int(n[0]), int(n[1])), (int(node[0]), int(node[1])), (1,0,0), 1)
                # Draw Image
                img_ = cv2.flip(img,0)
                cv2.imshow("test",img_)
                k = cv2.waitKey(1)
                if k == 27:
                    break
        
        # Extract Path
        path = []
        n = goal_node
        while(True):
            path.insert(0,n)
            if self.ntree[n] is None:
                break
            node = self.ntree[n]
            n = self.ntree[n] 
        path.append(goal)
        return path

def pos_int(p):
    return (int(p[0]), int(p[1]))

smooth = True
if __name__ == "__main__":
    # Config
    img = cv2.flip(cv2.imread("../Maps/map2.png"),0)
    img[img>128] = 255
    img[img<=128] = 0
    m = np.asarray(img)
    m = cv2.cvtColor(m, cv2.COLOR_RGB2GRAY)
    m = m.astype(float) / 255.
    m = 1-cv2.dilate(1-m, np.ones((20,20)))
    img = img.astype(float)/255.

    start=(100,200)
    goal=(380,520)
    cv2.circle(img,(start[0],start[1]),5,(0,0,1),3)
    cv2.circle(img,(goal[0],goal[1]),5,(0,1,0),3)

    rrt = RRT(m)
    path = rrt.planning(start, goal, 30, img)

    # Extract Path
    if not smooth:
        for i in range(len(path)-1):
            cv2.line(img, pos_int(path[i]), pos_int(path[i+1]), (0.5,0.5,1), 3)
    else:
        from cubic_spline import *
        path = np.array(cubic_spline_2d(path, interval=4))
        for i in range(len(path)-1):
            cv2.line(img, pos_int(path[i]), pos_int(path[i+1]), (0.5,0.5,1), 3)

    img_ = cv2.flip(img,0)
    cv2.imshow("test",img_)
    k = cv2.waitKey(0)
