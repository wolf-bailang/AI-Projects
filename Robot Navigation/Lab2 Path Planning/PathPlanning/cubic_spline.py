import numpy as np
import cv2

def pos_int(p):
    return (int(p[0]), int(p[1]))

def cubic_spline(x,y,interval=2):
    size = len(y)
    h = [x[i+1]-x[i] for i in range(len(x)-1)]
    A = np.zeros((size, size), dtype=np.float)
    for i in range(size):
        if i==0:
            A[i,0] = 1
            #A[i,0] = -h[i+1]
            #A[i,1] = h[i+1] + h[i] 
            #A[i,2] = -h[i]
        elif i==size-1:
            A[i,-1] = 1
            #A[i,-3] = -h[i-1]
            #A[i,-2] = h[i-2] + h[i-1] 
            #A[i,-1] = -h[i-1]
        else:
            A[i,i-1] = h[i-1]
            A[i,i] = 2*(h[i-1]+h[i])
            A[i,i+1] = h[i]
    #print(A)

    B = np.zeros((size,1), dtype=np.float)
    for i in range(1,size-1):
        B[i,0] = (y[i+1]-y[i])/h[i] - (y[i]-y[i-1])/h[i-1]
    B = 6*B
    #print(B)

    Ainv = np.linalg.pinv(A)
    m = Ainv.dot(B).T[0].tolist()
    a = [y[i] for i in range(size-1)]
    b = [(y[i+1]-y[i])/h[i] - h[i]*m[i]/2 - h[i]*(m[i+1]-m[i])/6 for i in range(size-1)]
    c = [m[i]/2 for i in range(size-1)]
    d = [(m[i+1]-m[i])/(6*h[i]) for i in range(size-1)]

    y_list = []
    dy_list = []
    ddy_list = []
    i = 0
    x_ = x[0]
    while(True):
        if x_ >= x[-1]:
            x_ = x[-1]
        elif x_ > x[i+1]:
            i += 1 
        y_ = a[i] + b[i]*(x_-x[i]) + c[i]*(x_-x[i])**2 + d[i]*(x_-x[i])**3
        dy = b[i] + 2.0*c[i]*(x_-x[i]) + 3.0*d[i]*(x_-x[i])**2
        ddy = 2.0*c[i] + 6.0*d[i]*(x_-x[i])
        y_list.append(y_)
        dy_list.append(dy)
        ddy_list.append(ddy)
        if x_ == x[-1]:
            break
        x_ += interval
    return y_list, dy_list, ddy_list

def cubic_spline_2d(path, interval=2):
    x = [path[i][0] for i in range(len(path))]
    y = [path[i][1] for i in range(len(path))]
    x_diff = [path[i+1][0]-path[i][0] for i in range(len(path)-1)]
    y_diff = [path[i+1][1]-path[i][1] for i in range(len(path)-1)]
    dist = np.hypot(np.array(x_diff),np.array(y_diff))
    dist_cum = np.cumsum(dist).tolist()
    dist_cum.insert(0,0)
    #print(dist, dist_cum)

    x_list, dx_list, ddx_list = cubic_spline(dist_cum,x,interval)
    y_list, dy_list, ddy_list = cubic_spline(dist_cum,y,interval)
    dx, ddx, dy, ddy = np.array(dx_list), np.array(ddx_list), np.array(dy_list), np.array(ddy_list)
    yaw_list = np.rad2deg(np.arctan2(dy,dx))
    curv_list = (ddy * dx - ddx * dy) / ((dx ** 2 + dy ** 2)**(3 / 2))
    path_smooth = [(x_list[i], y_list[i], yaw_list[i], curv_list[i]) for i in range(len(x_list))]
    
    #print(path_smooth)
    return path_smooth

if __name__ == "__main__":
    path = [(20,50), (40,100), (80,120), (160,60)]
    path_smooth = cubic_spline_2d(path)

    img = np.ones((200,200,3), dtype=np.float)
    for p in path:
        cv2.circle(img, p, 3, (0.5,0.5,0.5), 2)

    for i in range(len(path_smooth)-1):
        cv2.line(img, pos_int(path_smooth[i]), pos_int(path_smooth[i+1]), (1.0,0.4,0.4), 1)

    img = cv2.flip(img,0)
    cv2.imshow("Cubic Spline Test", img)
    cv2.waitKey(0)