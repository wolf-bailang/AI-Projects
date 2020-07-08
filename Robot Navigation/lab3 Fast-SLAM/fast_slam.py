Tfrom wmr_model import KinematicModel
import numpy as np
import cv2
import sys
import random
sys.path.append("../")


class Particle:
    def __init__(self, pos, R, params):
        self.init_pos(pos)
        self.R = R
        self.params = params

    def init_pos(self, pos):
        self.pos = list(pos)
        self.path = [self.pos]
        self.landmarks = {}

    def deepcopy(self):
        p = Particle(self.pos, self.R, self.params)
        p.pos = self.pos.copy()
        p.path = self.path.copy()
        for lid in self.landmarks:
            p.landmarks[lid] = self.landmarks[lid]
        return p

    def sampling(self, control):
        v, w, delta_t = control
        v_hat = v + random.gauss(0, self.params[0]*v**2+self.params[1]*w**2)
        w_hat = w + random.gauss(0, self.params[2]*v**2+self.params[3]*w**2)
        w_rad = np.deg2rad(w_hat)
        g_hat = random.gauss(0, self.params[4]*v**2+self.params[5]*w**2)

        if w_hat != 0:
            x_next = self.pos[0] - (v_hat/w_rad)*np.sin(np.deg2rad(self.pos[2])) + (
                v_hat/w_rad)*np.sin(np.deg2rad(self.pos[2]+w_hat*delta_t))
            y_next = self.pos[1] + (v_hat/w_rad)*np.cos(np.deg2rad(self.pos[2])) - (
                v_hat/w_rad)*np.cos(np.deg2rad(self.pos[2]+w_hat*delta_t))
            yaw_next = self.pos[2] + w_hat*delta_t + g_hat
        else:
            x_next = self.pos[0] + v_hat * \
                np.cos(np.deg2rad(self.pos[2]))*delta_t
            y_next = self.pos[1] + v_hat * \
                np.sin(np.deg2rad(self.pos[2]))*delta_t
            yaw_next = self.pos[2] + g_hat

        self.pos = [x_next, y_next, yaw_next]
        self.path.append(self.pos)
        return self.pos

    def observation_model(self, lm):
        delta_x = lm["mu"][0, 0] - self.pos[0]
        delta_y = lm["mu"][1, 0] - self.pos[1]
        q = delta_x**2 + delta_y**2
        z_r = np.sqrt(q)
        z_th = np.rad2deg(np.arctan2(delta_y, delta_x)) - self.pos[2]
        z_th = z_th % 360
        if z_th > 180:
            z_th -= 360
        return (z_r, z_th)

    def multi_normal(self, x, mean, cov):
        den = 2 * np.pi * np.sqrt(np.linalg.det(cov))
        num = np.exp(-0.5*np.transpose((x - mean)
                                       ).dot(np.linalg.inv(cov)).dot(x - mean))
        result = num/den
        return result[0][0]

    def compute_H(self, fx, fy):
        dx = fx - self.pos[0]
        dy = fy - self.pos[1]
        d2 = dx**2 + dy**2
        d = np.sqrt(d2)
        H = np.array([[dx / d, dy / d],
                      [-dy / d2, dx / d2]])
        return H

    def update_landmark(self, z, lid):
        if lid not in self.landmarks:
            # Add New Landmark
            c = np.cos(np.deg2rad(self.pos[2]+z[1]))
            s = np.sin(np.deg2rad(self.pos[2]+z[1]))
            mu = np.array([[self.pos[0] + z[0]*c], [self.pos[1] + z[0]*s]])
            sig = np.eye(2)*100
            self.landmarks[lid] = {"mu": mu, "sig": sig}
            p = 1.0
        else:
            # Update Old Landmark
            # todo
            """
            """
            self.landmarks[lid]["mu"] = mu + K@e
            self.landmarks[lid]["sig"] = (np.eye(2) - K@H) @ sig
            p = self.multi_normal(np.array(z_pre).reshape(
                2, 1), np.array(z).reshape(2, 1), Q)
        return p

    def update_obs(self, zlist, idlist):
        loglike = 0
        # todo
        # call update_landmark to help you calculate loglike
        """
        """
        return loglike


class ParticleFilter:
    def __init__(self, init_pos, R, params, psize=20):
        self.psize = psize
        self.weights = []
        self.particles = []
        for i in range(self.psize):
            self.particles.append(Particle(init_pos, R, params))
            self.weights.append(1 / float(self.psize))
        self.Neff = self.psize

    def init_pf(self, pos):
        for i in range(self.psize):
            self.particles[i].init_pos(pos)
            self.weights.append(1 / float(self.psize))

    def prediction(self, control):
        for i in range(self.psize):
            self.particles[i].sampling(control)

    def update_obs(self, zlist, idlist):
        loglike = []
        for i in range(self.psize):
            loglike.append(self.particles[i].update_obs(zlist, idlist))
        total = 0
        max_loglike = np.max(loglike)
        for i in range(self.psize):
            self.weights[i] = self.weights[i]*np.exp(loglike[i]-max_loglike)
            total += self.weights[i]
        # Normalize
        for i in range(self.psize):
            if total == 0:
                self.weights[i] = 1 / float(self.psize)
            else:
                self.weights[i] = self.weights[i]/total

    def resample(self):
        # todo
        # calculate Neff
        """
        """
        if self.Neff < self.psize/2:
            re_id = np.random.choice(
                self.psize, self.psize, p=list(self.weights))
            new_particles = []
            for i in range(self.psize):
                new_particles.append(self.particles[re_id[i]].deepcopy())
                self.weights[i] = 1 / float(self.psize)
            self.particles = new_particles


def main():
    # Parameters
    lsize = 60  # Number of Landmarks
    detect_dist = 120  # Landmark Detection Range
    psize = 30  # Number of Particles
    params = [0.00005, 0.00005, 0.00005, 0.00005,
              0.0001, 0.0001]    # Motion Noise
    R_sim = np.diag([2.0, 2.0]) ** 2    # Observation Noise

    # Create Landmark
    landmarks = []
    for i in range(lsize):
        rx = np.random.randint(10, 490)
        ry = np.random.randint(10, 490)
        landmarks.append((rx, ry))

    # Initialize Environment
    window_name = "Fast-SLAM Demo"
    cv2.namedWindow(window_name)
    img = np.ones((500, 500, 3))
    init_pos = (100, 200, 0)
    car = KinematicModel()
    car.init_state(init_pos)

    # Create Particle Filter
    pf = ParticleFilter((car.x, car.y, car.yaw), R_sim, params, psize=psize)

    while(True):
        ###################################
        # Simulate Control
        ###################################
        u = (car.v, car.w, car.dt)
        # Add Control Noise
        car.v += np.random.randn() * 0.00002*u[0]**2 + 0.00002*u[1]**2
        car.w += np.random.randn() * 0.00002*u[0]**2 + 0.00002*u[1]**2
        car.update()
        print("\rState: "+car.state_str() +
              " | Neff:"+str(pf.Neff)[0:7], end="\t")

        ###################################
        # Simulate Observation
        ###################################
        r = np.array(landmarks) - np.array((car.x, car.y))
        dist = np.hypot(r[:, 0], r[:, 1])
        detect_ids = np.where(dist < detect_dist)[0]    # Detected Landmark ids
        detect_lms = np.array(landmarks)[detect_ids]    # Detected Landmarks
        z = []    # Observation
        for i in range(detect_lms.shape[0]):
            lm = detect_lms[i]
            r = np.sqrt((car.x - lm[0])**2 + (car.y - lm[1])**2)
            phi = np.rad2deg(np.arctan2(lm[1]-car.y, lm[0]-car.x)) - car.yaw
            # Add Observation Noise
            r = r + np.random.randn() * R_sim[0, 0] ** 0.5
            phi = phi + np.random.randn() * R_sim[1, 1] ** 0.5
            # Normalize Angle
            phi = phi % 360
            if phi > 180:
                phi -= 360
            z.append((r, phi))

        ###################################
        # SLAM Algorithm
        ###################################
        pf.prediction(u)
        pf.update_obs(z, detect_ids)
        pf.resample()

        ###################################
        # Render Canvas
        ###################################
        img_ = img.copy()
        # Draw Landmark
        for lm in landmarks:
            cv2.circle(img_, lm, 3, (0.2, 0.5, 0.2), 1)
        for i in range(detect_lms.shape[0]):
            lm = detect_lms[i]
            cv2.line(img_, (int(car.x), int(car.y)),
                     (int(lm[0]), int(lm[1])), (0, 1, 0), 1)
        # Draw Trajectory
        start_cut = 100
        for j in range(psize):
            path_size = len(pf.particles[j].path)
            start = 0 if path_size < start_cut else path_size-start_cut
            for i in range(start, path_size-1):
                cv2.line(img_, (int(pf.particles[j].path[i][0]), int(pf.particles[j].path[i][1])), (int(
                    pf.particles[j].path[i+1][0]), int(pf.particles[j].path[i+1][1])), (1, 0.7, 0.7), 1)
        # Draw Best Trajectory
        start_cut = 1000
        bid = np.argmax(np.array(pf.weights))
        path_size = len(pf.particles[bid].path)
        start = 0 if path_size < start_cut else path_size-start_cut
        for i in range(start, path_size-1):
            cv2.line(img_, (int(pf.particles[bid].path[i][0]), int(pf.particles[bid].path[i][1])), (int(
                pf.particles[bid].path[i+1][0]), int(pf.particles[bid].path[i+1][1])), (1, 0, 0), 1)
        # Draw Particle
        for j in range(psize):
            cv2.circle(img_, (int(pf.particles[j].pos[0]), int(
                pf.particles[j].pos[1])), 2, (1, 0.0, 0.0), 1)
        cv2.circle(img_, (int(car.x), int(car.y)), detect_dist, (0, 1, 0), 1)
        # Draw Best Map
        for lid in pf.particles[bid].landmarks:
            lx = pf.particles[bid].landmarks[lid]["mu"][0, 0]
            ly = pf.particles[bid].landmarks[lid]["mu"][1, 0]
            ls = pf.particles[bid].landmarks[lid]["sig"][0, 0] ** 0.5
            # Draw Ellipse
            w, v = np.linalg.eig(pf.particles[bid].landmarks[lid]["sig"])
            angle = np.arctan2(v[0, 1], v[0, 0])
            if angle < 0:
                angle += 2*np.pi
            angle = np.rad2deg(angle)
            ax1 = np.sqrt(w[0])
            ax1 = 3 if ax1 < 3 else ax1
            ax2 = np.sqrt(w[1])
            ax2 = 3 if ax2 < 3 else ax2
            cv2.ellipse(img_, (int(lx), int(ly)), (int(ax1),
                                                   int(ax2)), angle, 0, 360, (0, 0, 1), 1)

        # Render Car
        img_ = car.render(img_)
        img_ = cv2.flip(img_, 0)
        cv2.imshow(window_name, img_)

        ###################################
        # Keyboard
        ###################################
        k = cv2.waitKey(1)
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
            pf.init_pf(init_pos)
            print("Reset!!")
        if k == 27:
            print()
            break


if __name__ == "__main__":
    main()
