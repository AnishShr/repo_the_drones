#! /usr/bin/env python3

import cvxpy as cp
import matplotlib.pyplot as plt
from cmath import inf
import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped, Twist
from tf import transformations
import numpy as np
import math
import time

#-------------------------------------------------------------------------------------------------------------------------------------

class cbf_bebop:
    def __init__(self):
        
        self.laser_sub = rospy.Subscriber("/scan", LaserScan, self.handle_laser)
        self.pose_sub = rospy.Subscriber("/vrpn_client_node/bebop/pose", PoseStamped, self.handle_pose)
        # Subscribe to the goal topic to get the goal position given using rviz's 2D Navigation Goal option.            
        self.goal_sub = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.handle_goal)

        self.box1_pose = rospy.wait_for_message("/vrpn_client_node/box1/pose", PoseStamped, timeout=5)
        self.box2_pose = rospy.wait_for_message("/vrpn_client_node/box2/pose", PoseStamped, timeout=5)
        self.box3_pose = rospy.wait_for_message("/vrpn_client_node/box3/pose", PoseStamped, timeout=5)
        
        self.vel_publisher = rospy.Publisher('/bebop/velocity', Twist, queue_size=1)
        
        self.vel = Twist()

        self.laser = None
        self.local_laser = None
        self.odom = None

        self.goal_x = None
        self.goal_y = None

        self.obs_D = 0.4
        
        # self.position_x = []
        # self.position_y = []

#-------------------------------------------------------------------------------------------------------------------------------------

    def handle_goal(self, goal_msg):
        # self.goal_x = goal_msg.goal.target_pose.pose.position.x
        # self.goal_y = goal_msg.goal.target_pose.pose.position.y

        self.goal_x = goal_msg.pose.position.x
        self.goal_y = goal_msg.pose.position.y


#-------------------------------------------------------------------------------------------------------------------------------------

    def handle_laser(self, laser_msg):

        if self.goal_x is not None:

            start_time = time.time()
            angle_min = laser_msg.angle_min
            angle_inc = laser_msg.angle_increment
            angles = np.arange(laser_msg.angle_min, laser_msg.angle_max, laser_msg.angle_increment)
            ranges = np.array(laser_msg.ranges)
            
            self.laser = []

            for i, dist in enumerate(ranges):
                if np.isfinite(dist) == True:
                    # print(dist)
                    self.laser.append((angle_min + i*angle_inc, dist))
        
            self.laser = np.array(self.laser)
            
            # print(f"laser data shape: {self.laser.shape}")

            # take 20 % of the obstacles
            max_obs_number = int(0.2 * self.laser.shape[0]) 

            # print(f"# of obstacles to consider: {max_obs_number}")

            closest_obs_indices = self.laser[:, 1].argsort()[:max_obs_number]

            # print(f"closest obs indices: {closest_obs_indices}")

            self.closest_lasers = self.laser[closest_obs_indices]
            
            # closest_angles = np.take(self.laser[:, 0], closest_obs_indices)
            # closest_distances = np.take(self.laser[:, 1], closest_obs_indices)

            closest_angles = self.closest_lasers[:, 0]
            closest_distances = self.closest_lasers[:, 1]

            # print(f"closest laser scan angles: {closest_angles}") 
            # print(f"closest laser scan distances: {closest_distances}") 


            # obs_angle = self.laser[:, 0]
            # obs_distance = self.laser[:, 1]

            # # print(f"obs_angle: {obs_angle}")
            # # print(f"obs_distance: {obs_distance}")

            x_obs = closest_distances * np.cos(closest_angles)
            y_obs = closest_distances * np.sin(closest_angles)

            print(len(x_obs))

            # print(f"x_obs: {x_obs}")
            # print(f"y_obs: {y_obs}")

            x_obs_global = x_obs + self.pos_x
            y_obs_global = y_obs + self.pos_y

            xi = cp.Variable(2)
            v_x = xi[0]
            v_y = xi[1]
            k = 0.6
            alpha = 0.5

            cost = (v_x + k*(self.pos_x-self.goal_x))**2 + (v_y + k*(self.pos_y-self.goal_y))**2      
            constraints = [cp.norm(xi, 2) <= 0.3]

            for i in range(len(x_obs)):
                hx = np.sqrt((self.pos_x - x_obs_global[i])**2 + (self.pos_y - y_obs_global[i])**2) - self.obs_D
                grad_hx = np.vstack(((self.pos_x-x_obs_global[i])/(hx+self.obs_D),
                                    (self.pos_y-y_obs_global[i])/(hx+self.obs_D)))

                constraints.append(grad_hx[0]*v_x + grad_hx[1]*v_y + alpha * hx >= 0)
            
            prob = cp.Problem(cp.Minimize(cost),
                            constraints)
            # prob.solve(solver=cp.ECOS)
            prob.solve()

            v_x = xi.value[0]
            v_y = xi.value[1]
            # theta = np.arctan2(v_y, v_x)

            # v = np.sqrt((v_x**2) + (v_y**2))
            # v = np.clip(v, 0, 0.3)
            # vel_x = v * np.cos(theta)
            # vel_y = v * np.sin(theta)

            rot_matrix = np.array([[np.cos(self.yaw), np.sin(self.yaw)],
                                    [-np.sin(self.yaw), np.cos(self.yaw)]])
            # v = np.dot(rot_matrix, np.array([vel_x, vel_y]))
            v = np.dot(rot_matrix, np.array([v_x, v_y]))

            # self.vel.linear.x = v_x
            # self.vel.linear.y = v_y

            self.vel.linear.x = v[0]
            self.vel.linear.y = v[1]

            print(f"velocities: {v[0], v[1]}")

            self.vel_publisher.publish(self.vel) 

            end_time = time.time()

            print(f"Took {end_time-start_time} seconds")
            print("-------------------------------------------------------")
        
#-------------------------------------------------------------------------------------------------------------------------------------
    
    def handle_pose(self, pose_msg):
        # self.pose = odom_data.pose

        # self.pos_x = odom_msg.pose.pose.position.x
        # self.pos_y = odom_msg.pose.pose.position.y

        self.pos_x = pose_msg.pose.position.x
        self.pos_y = pose_msg.pose.position.y
        
        orientation = pose_msg.pose.orientation
        quaternion = [orientation.x, orientation.y, orientation.z, orientation.w] 
        roll, pitch, yaw = transformations.euler_from_quaternion(quaternion)

        self.yaw = yaw

        # self.position_x.append(self.pos_x)
        # self.position_y.append(self.pos_y)
      
        
#-------------------------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    rospy.init_node("CBF_BEBOP", anonymous=True)
    node = cbf_bebop()
    rospy.spin()

    # plt.figure(figsize=(10, 10))
    # plt.title('Drone Trajectory under CBF')
    # plt.scatter(node.position_x, node.position_y)  

    # box1_x = node.box1_pose.pose.position.x
    # box1_y = node.box1_pose.pose.position.y

    # box2_x = node.box2_pose.pose.position.x
    # box2_y = node.box2_pose.pose.position.y

    # box3_x = node.box3_pose.pose.position.x
    # box3_y = node.box3_pose.pose.position.y

    # theta = np.linspace(0, 2*np.pi, 100)
    # r = 0.5

    # # box 1
    # x1 = box1_x + r * np.cos(theta)
    # y1 = box1_y + r * np.sin(theta)
    # plt.plot(x1, y1)

    # # box 2
    # x2 = box2_x + r * np.cos(theta)
    # y2 = box2_y + r * np.sin(theta)
    # plt.plot(x2, y2)

    # # box 3
    # x3 = box3_x + r * np.cos(theta)
    # y3 = box3_y + r * np.sin(theta)
    # plt.plot(x3, y3)

    # plt.show()