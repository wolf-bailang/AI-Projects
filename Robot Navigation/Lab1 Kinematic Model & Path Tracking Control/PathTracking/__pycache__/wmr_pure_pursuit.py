import numpy as np 

class PurePursuitControl:
    def __init__(self, kp=1, Lfc=10):
        self.path = None
        self.kp = kp
        self.Lfc = Lfc

    def set_path(self, path):
        self.path = path.copy()

    def _search_nearest(self, pos):
        min_dist = 99999999
        min_id = -1
        for i in range(self.path.shape[0]):
            dist = (pos[0] - self.path[i,0])**2 + (pos[1] - self.path[i,1])**2
            if dist < min_dist:
                min_dist = dist
                min_id = i
        return min_id, min_dist

    def feedback(self, state):
        # Check Path
        global targetIdx
        if self.path is None:
            print("No path !!")
            return None, None
        
        # Extract State 
        x, y, yaw, v = state["x"], state["y"], state["yaw"], state["v"]

        # Search Front Target
        min_idx, min_dist = self._search_nearest((x,y))

        # todo
        #####################################################################
        
        # all parameter name (ex:alpha) comes from the Slides
        # You need to finish the pure pursuit control algo

        # step by step
        # first, you need to calculate the look ahead distance Ld by formula
        Ld = self.kp * v + self.Lfc
        # second, you need to find a point(target) on the path which distance between the path and model is as same as the Ld
        ### hint: (you first need to find the nearest point and then find the point(target) backward, this will make your model won't go back)
        ### hint: (if you can not find a point(target) on the path which distance between the path and model is as same as the Ld, you need to find a similar one)
        """
        while min_dist < Ld:#
            #min_idx, min_dist = self._search_nearest((x,y))
            #target_x = np.arange(0, self.path[min_idx, 0] + 1, 1)
            #target_y = [np.sin(ix / 5.0) * ix / 2.0 for ix in target_x]
            #min_dist += 1
            min_idx, min_dist = self._search_nearest((x, y))
            #min_idx, min_dist = self._search_nearest((self.path[min_idx, 0],self.path[min_idx, 1]))
        # third, you need to calculate alpha
        alpha = np.arctan2(y - self.path[min_idx, 1], x - self.path[min_idx, 0]) - yaw#np.deg2rad()
        # now, you can calculate the ω
        w = (2.0 * v * np.sin(alpha)) / Ld
        next_w = w / 360
        target = [self.path[min_idx, 0]+1, self.path[min_idx, 1]+1]
        """
        for i in range(min_idx, self.path.shape[0]):
            dist = (self.path[i, 0] - self.path[min_idx, 0]) ** 2 + (self.path[i, 1] - self.path[min_idx, 1]) ** 2
            if dist >= Ld:
                # xg, yg, yawg, vg = self.path[i,0], self.path[i,1], self.path[i,2], self.path[i,3]
                targetIdx = i
                break
        alpha = np.arctan2(y - self.path[targetIdx, 1], x - self.path[targetIdx, 0]) - np.deg2rad(yaw)
        w = -2 * v * np.sin(alpha) / Ld
        next_w = np.rad2deg(w)
        target = [self.path[targetIdx, 0], self.path[targetIdx, 1]]  #self.path[targetIdx]

        # The next_w is Pure Pursuit Control's output
        # The target is the point on the path which you find
        #####################################################################
        return next_w, target

if __name__ == "__main__":
    import cv2
    import path_generator
    import sys
    sys.path.append("../")
    from wmr_model import KinematicModel

    # Path
    path = path_generator.path2()
    img_path = np.ones((600,600,3))
    for i in range(path.shape[0]-1):
        cv2.line(img_path, (int(path[i,0]), int(path[i,1])), (int(path[i+1,0]), int(path[i+1,1])), (1.0,0.5,0.5), 1)

    # Initialize Car
    car = KinematicModel()
    car.init_state((50,300,0))
    controller = PurePursuitControl()
    controller.set_path(path)

    while(True):
        print("\rState: "+car.state_str(), end="\t")

        # ================= Control Algorithm ================= 
        # PID Longitude Control
        end_dist = np.hypot(path[-1,0]-car.x, path[-1,1]-car.y)
        target_v = 40 if end_dist > 265 else 0
        next_a = 0.1*(target_v - car.v)

        # Pure Pursuit Lateral Control
        state = {"x":car.x, "y":car.y, "yaw":car.yaw, "v":car.v}
        next_delta, target = controller.feedback(state)
        car.control(next_a,next_delta)
        # =====================================================
        
        # Update & Render
        car.update()
        img = img_path.copy()
        cv2.circle(img,(int(target[0]),int(target[1])),3,(1,0.3,0.7),2) # target points
        img = car.render(img)
        img = cv2.flip(img, 0)
        cv2.imshow("Pure-Pursuit Control Test", img)
        k = cv2.waitKey(1)
        if k == ord('r'):
            _init_state(car)
        if k == 27:
            print()
            break