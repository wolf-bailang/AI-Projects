import numpy as np 

class StanleyControl:
    def __init__(self, kp=0.5):
        self.path = None
        self.kp = kp

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

    # State: [x, y, yaw, delta, v, l]
    def feedback(self, state):
        # Check Path
        if self.path is None:
            print("No path !!")
            return None, None
        
        # Extract State 
        x, y, yaw, delta, v, l = state["x"], state["y"], state["yaw"], state["delta"], state["v"], state["l"]

        # todo
        #############################################################################

        # all parameter name (ex:alpha) comes from the Slides
        # You need to finish the Stanley control algo

        # step by step
        # first you need to find the nearest point on the path(centered on the front wheel, previous work all on the back wheel)
        xf = x + l*np.cos(np.deg2rad(yaw))
        yf = y + l * np.sin(np.deg2rad(yaw))
        min_idx, min_dist = self._search_nearest((xf, yf))
        # second you need to calculate the theta_e by use the "nearest point's yaw" and "model's yaw"
        theta = yaw
        """
        if theta < -180:
            theta = -(180 + theta)
        elif theta > 180:
            theta = -(-180 + theta)
        """
        if theta < -180:
            theta = 360 + theta
        elif theta > 180:
            theta = -360 + theta

        theta_p = self.path[min_idx, 2]
        theta_e = (theta_p - theta)
        #print("    theta = ",theta)
        #print("    theta_p = ", theta_p)
        #print("    theta_e = ", theta_e)

        # third you need to calculate the v front(vf) and error(e)
        vf = v / np.cos(np.deg2rad(delta)) # self.path[min_idx, 3]
        e = (xf - self.path[min_idx, 0]) * np.cos(np.deg2rad(theta_p + 90)) + (yf - self.path[min_idx, 1]) * np.sin(np.deg2rad(theta_p + 90))
        # now, you can calculate the delta
        """
        if v == 0:
            next_delta = np.rad2deg(theta_e)
        else:
            delta = np.arctan2(-self.kp * e /vf ,1.0) + np.deg2rad(theta_e)
            next_delta = np.rad2deg(delta)
        """
        if vf == 0:
            delta = theta_e
        else:
            delta = np.rad2deg(np.arctan(-self.kp * e / vf)) + theta_e
        next_delta = int(delta) #% 360
        target = [self.path[min_idx, 0], self.path[min_idx, 1]]
        # The next_delta is Stanley Control's output
        # The target is the point on the path which you find
        ###############################################################################
        return next_delta, target

if __name__ == "__main__":
    import cv2
    import path_generator
    import sys
    sys.path.append("../")
    from bicycle_model import KinematicModel

    # Path
    path = path_generator.path2()
    img_path = np.ones((600,600,3))
    for i in range(path.shape[0]-1):
        cv2.line(img_path, (int(path[i,0]), int(path[i,1])), (int(path[i+1,0]), int(path[i+1,1])), (1.0,0.5,0.5), 1)
    
    # Initialize Car
    car = KinematicModel()
    start = (50,300,0)
    car.init_state(start)
    controller = StanleyControl(kp=0.5)
    controller.set_path(path)

    while(True):
        print("\rState: "+car.state_str(), end="\t")

        # PID Longitude Control
        end_dist = np.hypot(path[-1,0]-car.x, path[-1,1]-car.y)
        target_v = 40 if end_dist > 265 else 0
        next_a = 0.1*(target_v - car.v)

        # Stanley Lateral Control
        state = {"x":car.x, "y":car.y, "yaw":car.yaw, "delta":car.delta, "v":car.v, "l":car.l}
        next_delta, target = controller.feedback(state)
        car.control(next_a, next_delta)
        
        # Update State & Render
        car.update()
        img = img_path.copy()
        cv2.circle(img,(int(target[0]),int(target[1])),3,(1,0.3,0.7),2) # target points
        img = car.render(img)
        img = cv2.flip(img, 0)
        cv2.imshow("Stanley Control Test", img)
        k = cv2.waitKey(1)
        if k == ord('r'):
            car.init_state(start)
        if k == 27:
            print()
            break
