import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

class CBF():

    def __init__(self):
        
        # self.obs_x = 3.0
        # self.obs_y = 3.0
        self.obs_D = 0.5 + 0.2 # safe distance plus radius of the robot

        self.obs_x = np.array([1.0, 2.5])
        self.obs_y = np.array([2.0, 3.0])

        self.goal_x = 3.0
        self.goal_y = 5.0
        
        self.vel_publisher = rospy.Publisher('/bebop/cmd_vel',
                                             Twist,
                                             queue_size=1)
        self.vel = Twist()

        rospy.Subscriber('/bebop/odom',
                         Odometry,
                         self.odom_callback,
                         queue_size=1)

    def odom_callback(self, odom_msg):
        pos_x = odom_msg.pose.pose.position.x
        pos_y = odom_msg.pose.pose.position.y

        x_points.append(pos_x)
        y_points.append(pos_y)

        hx = np.sqrt((pos_x - self.obs_x)**2 + (pos_y - self.obs_y)**2) - self.obs_D
        print(f"hx: {hx}")
        # grad_hx = np.hstack(((pos_x-self.obs_x)/hx, (pos_y-self.obs_y)/hx))
        grad_hx = np.vstack(((pos_x-self.obs_x)/(hx+self.obs_D), (pos_y-self.obs_y)/(hx+self.obs_D)))
        print(f"grad_hx: {grad_hx}")


        self.xi = cp.Variable(2)
        self.v_x = self.xi[0]
        self.v_y = self.xi[1]
        self.k = 1.0
        self.alpha = 0.6

        cost = (self.v_x + self.k*(pos_x-self.goal_x))**2 + (self.v_y + self.k*(pos_y-self.goal_y))**2
        # print(f"cost: {cost}")
        # constraints = [grad_hx[0]*self.v_x + grad_hx[1]*self.v_y + self.alpha * hx >= 0, cp.norm(self.xi, 2) <= 0.3]
        constraints = [grad_hx[0][0]*self.v_x + grad_hx[1][0]*self.v_y + self.alpha * hx[0] >= 0,
                       grad_hx[0][1]*self.v_x + grad_hx[1][1]*self.v_y + self.alpha * hx[1] >= 0,
                       cp.norm(self.xi, 2) <= 0.3]
        # print(f"constraints: {constraints}")

        prob = cp.Problem(cp.Minimize(cost),
                          constraints)
        prob.solve()

        self.v_x = self.xi.value[0]
        self.v_y = self.xi.value[1]

        self.vel.linear.x = self.v_x
        self.vel.linear.y = self.v_y

        self.vel_publisher.publish(self.vel)


if __name__ == "__main__":
    rospy.init_node("control_barrier_function_test")
    x_points = []
    y_points = []
    
    node = CBF()
    rospy.spin()

    plt.figure(figsize=(10, 10))
    plt.title('Drone Trajectory under CBF')
    plt.scatter(x_points, y_points)    

    plt.scatter(node.obs_x, node.obs_y)
    theta = np.linspace(0, 2*np.pi, 100)
    r = 0.5
    
    # circle 1
    x1 = node.obs_x[0] + r * np.cos(theta)
    y1 = node.obs_y[0] + r * np.sin(theta)
    plt.plot(x1, y1)

    # circle 2
    x2 = node.obs_x[1] + r * np.cos(theta)
    y2 = node.obs_y[1] + r * np.sin(theta)
    plt.plot(x2, y2)

    plt.scatter(node.goal_x, node.goal_y)

    plt.show()