import numpy as np
import cv2
import sys
import random
sys.path.append("../")
from wmr_model import KinematicModel

class Particle:
    def __init__(self, pos, Qt, motion_noise):
        self.init_pos(pos)
        self.Qt = Qt
        self.motion_noise = motion_noise

    # Initialize particle pose and map
    def init_pos(self, pos):
        self.pos = list(pos)
        self.path = [self.pos]
        self.landmarks = {}

    # Copy the memory of whole particle.
    def deepcopy(self):
        pt = Particle(self.pos, self.Qt, self.motion_noise)
        pt.pos = self.pos.copy()
        pt.path = self.path.copy()
        for lid in self.landmarks:
            pt.landmarks[lid] = self.landmarks[lid]
        return pt

    # Sample next pose from motion model.
    def sample(self, control):
        self.pos = motion_model(self.pos, control, self.motion_noise)
        self.path.append(self.pos)
        return self.pos

    # Predict the observation of landmark.
    def observation_model(self, lx, ly):
        x, y, yaw = self.pos
        # TODO(Lab-5): Compute the prediction of observation.
        # [Hint 1] The parameter "lx,ly" is the location of landmark.
        #'''
        z_r = np.sqrt((lx-x)**2 + (ly-y)**2)
        z_th = np.arctan2(ly-y,lx-x) - yaw
        z_th = normalize_angle(z_th)
        #'''
        return (z_r, z_th)

    # Linearized Observation Matrix
    def compute_H(self, lx, ly):
        x, y, yaw = self.pos
        # TODO(Lab-6): Contruct the matrix of linearized observation model.
        # [Hint 1] The parameter "lx,ly" is the location of landmark.
        #'''
        q = (lx-x)**2 + (ly-y)**2
        H =np.array([[(lx-x)/np.sqrt(q),(ly-y)/np.sqrt(q)],[-(ly-y)/(q),(lx-x)/(q)]])
        #'''
        return H

    # Update one landmark given the observation
    def update_landmark(self, z, lid):
        x, y, yaw = self.pos
        if lid not in self.landmarks:
            # TODO(Lab-7): Add new landmark (mean and covariance).
            # [Hint 1] The parameter "z" is a list of one landmark [r, phi].
            # [Hint 2] The observation noise is "self.Qt (numpy array)".
            # [Hint 3] The mean of landmark "mu" is a numpy array with shape (2,1).
            #'''
            c = np.cos(yaw+z[1])
            s = np.sin(yaw+z[1])            
            mu = np.array([[x+z[0]*c], [y+z[0]*s]])
            H = self.compute_H(mu[0,0], mu[1,0])
            Hinv = np.linalg.inv(H)
            sig = Hinv @ self.Qt @ Hinv.T
            self.landmarks[lid] = {"mu":mu, "sig":sig}
            #'''
            p = 1.0
        else:
            # TODO(Lab-8): Update existing landmark (mean and covariance).
            #'''
            mu = self.landmarks[lid]["mu"]
            sig = self.landmarks[lid]["sig"]
            z_hat = self.observation_model(mu[0,0], mu[1,0])
            H = self.compute_H(mu[0,0], mu[1,0])
            Q = H @ sig @ H.T + self.Qt
            K = sig @ H.T @ np.linalg.inv(Q)
            e = np.array(z) - np.array(z_hat)
            e[1] = normalize_angle(e[1])
            self.landmarks[lid]["mu"] = mu + (K @ e).reshape(2,1)
            self.landmarks[lid]["sig"] = (np.eye(2)- K @ H) @ sig
            #'''
            p = multi_normal(np.array(z).reshape(2,1),np.array(z_hat).reshape(2,1), Q)
        return p
    
    # Update observed landmarks and get likelihood.
    def update(self, zlist, idlist):
        likelihood = 1
        for i in range(len(zlist)):
            p = self.update_landmark(zlist[i], idlist[i])
            likelihood *= p
        return likelihood

class ParticleFilter:
    def __init__(self, init_pos, R, motion_noise, psize=20):
        self.psize = psize
        self.weights = np.ones(self.psize) / self.psize
        self.particles = []
        for i in range(self.psize):
            self.particles.append(Particle(init_pos, R, motion_noise))
        self.Neff = self.psize

    # Sample next pose of particles
    def sample(self, control):
        for i in range(self.psize):
            self.particles[i].sample(control)

    # Update the map and weight of particles given the observation.
    def update(self, zlist, idlist):
        likelihood = np.zeros(self.psize)
        # Update observation
        for i in range(self.psize):
            likelihood[i] = self.particles[i].update(zlist, idlist)
        
        # Update weight
        self.weights = self.weights * likelihood
        if np.sum(self.weights) != 0:
            self.weights /= np.sum(self.weights)
        else:
            self.weights = np.ones(self.psize) / self.psize
    
    # Resampling Process
    def resample(self):
        # TODO(Lab-10): Compute Neff for evaluating the particle distribution.
        # [Hint 1] The particle weight is "self.weight (numpy array)".
        #'''
        self.Neff = 1/np.sum(self.weights**2)
        #'''
        if self.Neff < self.psize/2:
            re_id = np.random.choice(self.psize, self.psize, p=list(self.weights))
            new_particles = []
            for i in range(self.psize):
                new_particles.append(self.particles[re_id[i]].deepcopy())
            self.particles = new_particles
            self.weights = np.ones(self.psize) / self.psize

###################################
# Utility Functions
###################################
def motion_model(pos, control, motion_noise=[0]*6):
    x, y, yaw = pos
    v, w, delta_t = control
    
    # TODO(Lab-1): Noise Control
    #'''
    v_hat = v + np.sqrt(motion_noise[0] * v**2 + motion_noise[1] * w**2) * np.random.randn()
    w_hat = w + np.sqrt(motion_noise[2] * v**2 + motion_noise[3] * w**2) * np.random.randn()
    w_hat = normalize_angle(w_hat)
    g_hat = np.sqrt(motion_noise[4] * v**2 + motion_noise[5] * w**2) * np.random.randn()
    #'''
    if w_hat != 0:
        # TODO(Lab-2): motion prediction (with angular velocity)
        # '''
        x_next = x - (v_hat/w_hat)*np.sin(yaw) + (v_hat/w_hat)*np.sin(yaw + w_hat*delta_t)
        y_next = y + (v_hat/w_hat)*np.cos(yaw) - (v_hat/w_hat)*np.cos(yaw + w_hat*delta_t)
        yaw_next = yaw + w_hat*delta_t + g_hat*delta_t
        #'''
    else:
        # TODO(Lab-3): motion prediction (without angular velocity)
        # '''
        x_next = x + v_hat*np.cos(yaw)*delta_t
        y_next = y + v_hat*np.sin(yaw)*delta_t
        yaw_next = yaw + g_hat * delta_t
        #'''
    return [x_next, y_next, yaw_next]

def multi_normal(x, mean, cov):
    # TODO(Lab-9): Compute likelihood of multivariate normal distribution.
    #'''
    err = x - mean
    err[1,0] = normalize_angle(err[1,0])
    w=np.exp(-0.5 * err.T @ np.linalg.inv(cov)@err) / (2 * np.pi * np.sqrt(np.linalg.det(cov)))
    #'''
    return w

def normalize_angle(ang):
    temp = (ang + np.pi) % (2*np.pi) - np.pi
    return temp

def draw_path(img, path, start_cut, color):
    path_size = len(path)
    start = 0 if path_size<start_cut else path_size-start_cut
    for i in range(start, path_size-1):
        cv2.line(img, (int(path[i][0]),int(path[i][1])), (int(path[i+1][0]),int(path[i+1][1])), color, 1)

def draw_ellipse(img, mean, cov, color):
    w, v = np.linalg.eig(cov)
    angle = normalize_angle(np.arctan2(v[0,1], v[0,0]))
    angle = np.rad2deg(angle)
    # 95% convidence interval (2*var)
    ax1 = 3 if 2*np.sqrt(w[0])<3 else 2*np.sqrt(w[0])
    ax2 = 3 if 2*np.sqrt(w[1])<3 else 2*np.sqrt(w[1])
    cv2.ellipse(img, (int(mean[0]),int(mean[1])), (int(ax1),int(ax2)), -angle, 0, 360, color, 1)

###################################
# Main Function
###################################
def main():
    # Parameters
    N_PARTICLES = 40  # Number of Particles
    N_LANDMARKS = 80  # Number of Landmarks
    PERCEPTUAL_RANGE = 120 # Landmark Detection Range
    MOTION_NOISE = np.array([1e-5, 1e-2, 1e-5, 1e-2, 1e-5, 1e-2]) # Motion Noise
    Qt_sim = np.diag([4, np.deg2rad(4)]) ** 2    # Observation Noise

    # Create landmarks
    landmarks = []
    for i in range(N_LANDMARKS):
        rx = np.random.randint(10,490)
        ry = np.random.randint(10,490)
        landmarks.append((rx, ry))

    # Initialize environment
    window_name = "Fast-SLAM Demo"
    cv2.namedWindow(window_name)
    img = np.ones((500,500,3))
    init_pos = (250,100,0)
    car = KinematicModel(motion_noise=MOTION_NOISE)
    car.init_state(init_pos)
    car.v = 24
    car.w = np.deg2rad(10)
    path_odometry = [(init_pos)]

    # Create particle filter
    pf = ParticleFilter((car.x, car.y, car.yaw), Qt_sim, MOTION_NOISE, psize=N_PARTICLES)
    while(True):
        ###################################
        # Simulate Controlling
        ###################################
        u = (car.v,car.w,car.dt)
        car.update()
        # TODO(Lab-4): Remove the comment after complete the motion model.
        #'''
        pos_odometry = motion_model(path_odometry[-1], u)
        path_odometry.append(pos_odometry)
        #'''
        print("\rState: "+car.state_str() + " | Neff:"+str(pf.Neff)[0:7], end="\t")

        ###################################
        # Simulate Observation
        ###################################
        rvec = np.array(landmarks) - np.array((car.x, car.y))
        dist = np.hypot(rvec[:,0], rvec[:,1])
        landmarks_id = np.where(dist < PERCEPTUAL_RANGE)[0]    # Detected Landmark ids
        landmarks_detect = np.array(landmarks)[landmarks_id]    # Detected Landmarks
        z = []
        for i in range(landmarks_detect.shape[0]):
            lm = landmarks_detect[i]
            r = dist[landmarks_id[i]]
            phi = np.arctan2(lm[1]-car.y, lm[0]-car.x) - car.yaw
            # Add Observation Noise
            r = r + np.random.randn() * Qt_sim[0, 0] ** 0.5
            phi = phi + np.random.randn() * Qt_sim[1, 1] ** 0.5
            phi = normalize_angle(phi)
            z.append((r,phi))
        
        ###################################
        # SLAM Algorithm
        ###################################
        # TODO(Lab-0): Remove the comment after complete the class.
        #'''
        pf.sample(u)
        pf.update(z, landmarks_id)
        pf.resample()
        #'''

        ###################################
        # Render Canvas
        ###################################
        img_ = img.copy()
        # Draw landmark
        for lm in landmarks:
            cv2.circle(img_, lm, 3, (0.1,0.7,0.1), 1)
        for i in range(landmarks_detect.shape[0]):
            lm = landmarks_detect[i]
            cv2.line(img_, (int(car.x),int(car.y)), (int(lm[0]),int(lm[1])), (0,1,0), 1)
        # Draw path
        for i in range(N_PARTICLES):
            draw_path(img_, pf.particles[i].path, 100, (1,0.7,0.7))
        bid = np.argmax(np.array(pf.weights))
        draw_path(img_, pf.particles[bid].path, 1000, (1,0,0))  # Draw Best Path
        draw_path(img_, path_odometry, 1000, (0,0,0))    # Draw Odometry Path
        # Draw particle pose
        for i in range(N_PARTICLES):
            cv2.circle(img_, (int(pf.particles[i].pos[0]),int(pf.particles[i].pos[1])), 2, (1,0,0), 1)
            # Draw maps of particles
            '''
            for lm in pf.particles[i].landmarks:
                lx = pf.particles[i].landmarks[lm]["mu"][0,0]
                ly = pf.particles[i].landmarks[lm]["mu"][1,0]
                cv2.circle(img_, (int(lx),int(ly)), 2, (1,0,0), 1)
            '''
        # Draw map of best particle
        for lid in pf.particles[bid].landmarks:
            mean = pf.particles[bid].landmarks[lid]["mu"].reshape(2)
            cov = pf.particles[bid].landmarks[lid]["sig"]
            draw_ellipse(img_, mean, cov, (0,0,1))
        
        cv2.circle(img_, (int(car.x),int(car.y)), PERCEPTUAL_RANGE, (0,1,0), 1)  # Draw Detect Range
        img_ = car.render(img_) # Render Car
        img_ = cv2.flip(img_, 0)
        cv2.imshow(window_name ,img_)

        ###################################
        # Keyboard
        ###################################
        k = cv2.waitKey(1)#1
        if k == ord("w"):
            car.v += 4
        elif k == ord("s"):
            car.v += -4
        if k == ord("a"):
            car.w += 5
        elif k == ord("d"):
            car.w += -5
        elif k == ord("r"):
            car.init_state(init_pos)
            pf = ParticleFilter((car.x, car.y, car.yaw), Qt_sim, MOTION_NOISE, psize=N_PARTICLES)
            path_odometry = [(init_pos)]
            print("Reset!!")
        if k == 27:
            print()
            break
        
if __name__ == "__main__":
    np.random.seed(0)
    main()