import cv2
import numpy as np

class AStar():
    def __init__(self,m):
        self.map = m
        self.initialize()

    def initialize(self):
        self.queue = []
        self.parent = {}
        self.g = {} # Distance from node to goal
        self.goal_node = None

    # estimation
    def _distance(self, a, b):
        # Diagonal distance h(x)
        d = np.max([np.abs(a[0]-b[0]), np.abs(a[1]-b[1])])
        return d
    def planning(self, start=(100,200), goal=(375,520), inter=10, img=None):
        # Initialize 
        self.initialize()
        self.queue.append(start)    # 把起点加入open list。
        self.parent[start] = None
        self.g[start] = 0
        node_goal = None
        # 重复如下过程
        while(1):
            min_dist = 99999
            min_id = -1
            # 遍历open list ，查找F值最小的节点，把它作为当前要处理的节点。
            for i, node in enumerate(self.queue):
                f = self.g[node]
                if f < min_dist:
                    min_dist = f
                    min_id = i

            # pop the nearest node
            # 把这个节点移到close list
            p = self.queue.pop(min_id)

            # meet obstacle, skip
            # 如果它是不可抵达的或者它在 close list 中，忽略它。
            if self.map[p[1],p[0]]<0.5:
                
                continue
            # find goal
            # 停止，当你把终点加入到了 open list 中，此时路径已经找到了，或者查找终点失败，并且 open list 是空的，
            # 此时没有路径。
            if self._distance(p,goal) < inter:
                self.goal_node = p
                break

            # eight direction
            pts_next1 = [(p[0]+inter,p[1]), (p[0],p[1]+inter), (p[0]-inter,p[1]), (p[0],p[1]-inter)]
            pts_next2 = [(p[0]+inter,p[1]+inter), (p[0]-inter,p[1]+inter), (p[0]-inter,p[1]-inter), (p[0]+inter,p[1]-inter)]
            pts_next = pts_next1 + pts_next2

            for pn in pts_next:
                # 如果它不在 open list 中，把它加入 open list ，并且把当前方格设置为它的父亲，记录该方格的 F ， G 值。
                if pn not in self.parent:
                    self.queue.append(pn)
                    self.parent[pn] = p
                    self.g[pn] = self.g[p] + inter
                # 如果它已经在 open list 中，检查这条路径 ( 即经由当前方格到达它那里 ) 是否更好，用 G 值作参考。更小的
                # G 值表示这是更好的路径。如果是这样，把它的父亲设置为当前方格，并重新计算它的 G 和 F 值。如果你的
                # open list 是按 F 值排序的话，改变后你可能需要重新排序。
                elif self.g[pn]>self.g[p] + inter:
                    self.parent[pn] = p
                    self.g[pn] = self.g[p] + inter
            
            if img is not None:
                cv2.circle(img,(start[0],start[1]),5,(0,0,1),3)
                cv2.circle(img,(goal[0],goal[1]),5,(0,1,0),3)
                cv2.circle(img,p,2,(0,0,1),1)
                img_ = cv2.flip(img,0)
                cv2.imshow("A* Test",img_)
                k = cv2.waitKey(1)
                if k == 27:
                    break
        
        # Extract path
        path = []
        p = self.goal_node
        while(True):
            path.insert(0,p)
            if self.parent[p] == None:
                break
            p = self.parent[p]
        if path[-1] != goal:
            path.append(goal)
        return path

smooth = True
if __name__ == "__main__":
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
    astar = AStar(m)
    path = astar.planning(start=start, goal=goal, img=img, inter=20)
    print(path)

    cv2.circle(img,(start[0],start[1]),5,(0,0,1),3)
    cv2.circle(img,(goal[0],goal[1]),5,(0,1,0),3)
    # Extract Path
    if not smooth:
        for i in range(len(path)-1):
            cv2.line(img, path[i], path[i+1], (1,0,0), 2)
    else:
        from cubic_spline import *
        path = np.array(cubic_spline_2d(path, interval=1))
        for i in range(len(path)-1):
            cv2.line(img, pos_int(path[i]), pos_int(path[i+1]), (1,0,0), 1)
    
    img_ = cv2.flip(img,0)
    cv2.imshow("A* Test",img_)
    k = cv2.waitKey(0)