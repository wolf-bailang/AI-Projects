import numpy as np 

class PidControl:
    def __init__(self, kp=0.4, ki=0.0001, kd=0.5):
        self.path = None
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.acc_ep = 0
        self.last_ep = 0
    
    def set_path(self, path):
        self.path = path.copy()
        self.acc_ep = 0
        self.last_ep = 0
    
    def _search_nearest(self, pos):
        min_dist = 99999999
        min_id = -1
        for i in range(self.path.shape[0]):
            dist = (pos[0] - self.path[i,0])**2 + (pos[1] - self.path[i,1])**2
            if dist < min_dist:
                min_dist = dist
                min_id = i
        return min_id, min_dist
    
    # todo
    def feedback(self, state):
        # Check Path
        if self.path is None:
            print("No path !!")
            return None, None
        
        # Extract State
        x, y, dt = state["x"], state["y"], state["dt"]

        # Search Nesrest Target
        min_idx, min_dist = self._search_nearest((x,y))

        # todo
        ############################################################################
        
        # all parameter name (ex:alpha) comes from the Slides
        # You need to finish the PID control algo

        # step by step
        # first, you need to calculate the angle(between model and the nearest point(target) on the path), you can use the parameter "self.path" and "min_idx" to get it
        ang = np.arctan2(self.path[min_idx, 1] - y, self.path[min_idx, 0] - x)
        # second, you need to calculate the error(e(t)) in PID control, you can use the parameter "min_dist" and "angle" to get it
        ep = min_dist * np.sin(ang)
        self.acc_ep += dt * ep
        diff_ep = (ep - self.last_ep) / dt
        # now, you can caculate the P, I and D
        next_w = self.kp * ep + self.ki * self.acc_ep + self.kd * diff_ep
        self.last_ep = ep
        # The next_w is PID Control's output
        ############################################################################
        return next_w, self.path[min_idx]
        

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

    # Initial Car
    car = KinematicModel()
    car.init_state((50,300,0))
    controller = PidControl()
    controller.set_path(path)

    while(True):
        print("\rState: "+car.state_str(), end="\t")

        # PID Longitude Control
        end_dist = np.hypot(path[-1,0]-car.x, path[-1,1]-car.y)
        target_v = 40 if end_dist > 262 else 0
        next_a = 0.1*(target_v - car.v)

        # PID Lateral Control
        state = {"x":car.x, "y":car.y, "yaw":car.yaw, "dt":car.dt}
        next_w, target = controller.feedback(state)
        car.control(next_a, next_w)
        car.update()

        # Update State & Render
        img = img_path.copy()
        cv2.circle(img,(int(target[0]),int(target[1])),3,(0.7,0.3,1),2)
        img = car.render(img)
        img = cv2.flip(img, 0)
        cv2.imshow("demo", img)
        k = cv2.waitKey(1)
        if k == ord('r'):
            init_state(car)
        if k == 27:
            print()
            break
