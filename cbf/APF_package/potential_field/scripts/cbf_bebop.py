#! /usr/bin/env python3

import cvxpy as cp
import matplotlib.pyplot as plt
from cmath import inf
from visualization_msgs.msg import Marker,MarkerArray
import roslib, sys, rospy
import tf2_ros
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry,Path
from geometry_msgs.msg import Point, PoseStamped, Twist
from move_base_msgs.msg import MoveBaseActionGoal
from tf import transformations
import numpy as np
import math
import time
#------------------------------------------

class cbf_bebop:
    def __init__(self):

        # self.sample_rate = rospy.get_param("~sample_rate", 10)
        self.mode = "simulation" # simulation

        # Subscribe to the global planner using the move base package. The global plan is the path that the robot would ideally follow if 
        # there are no unknown/dynamic obstacles. In the videos this is highlighted by green color.
        # self.global_path_sub = rospy.Subscriber("/move_base/NavfnROS/plan", Path, self.handle_global_path)
        # self.global_path_sub = rospy.Subscriber("/move_base/GlobalPlanner/plan", Path, self.handle_global_path)

        if self.mode == "simulation":
            # self.goal_sub = rospy.Subscriber("/move_base/goal", MoveBaseActionGoal, self.handle_goal)
            self.laser_sub = rospy.Subscriber("/bebop/laser_scan", LaserScan, self.handle_laser)
            self.odom_sub = rospy.Subscriber("/bebop/odom", Odometry, self.handle_odom)
            self.vel_publisher = rospy.Publisher('/bebop/cmd_vel', Twist, queue_size=1)
            # Subscribe to the goal topic to get the goal position given using rviz's 2D Navigation Goal option.
            # self.goal_sub = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.handle_goal)
        else:
            self.laser_sub = rospy.Subscriber("/scan", LaserScan, self.handle_laser)
            self.pose_sub = rospy.Subscriber("/vrpn_client_node/bebop/pose", PoseStamped, self.handle_odom)
            # Subscribe to the goal topic to get the goal position given using rviz's 2D Navigation Goal option.
            self.goal_sub = rospy.Subscriber("/move_base/GlobalPlanner/plan", PoseStamped, self.handle_goal)
            
        
        self.vel = Twist()

        self.laser = None
        self.local_laser = None
        self.odom = None


        self.goal_x = 5.0 #2.0 #3.0
        self.goal_y = 7.0 #5.0 #4.0

        self.obs_D = 0.5
        
        self.position_x = []
        self.position_y = []

#------------------------------------------

    def handle_goal(self, goal_msg):
        self.goal_x = goal_msg.goal.target_pose.pose.position.x
        self.goal_y = goal_msg.goal.target_pose.pose.position.y


#------------------------------------------

    def handle_laser(self, laser_msg):
        # print(laser_msg)
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

        # take 30 % of the obstacles
        max_obs_number = int(0.3 * self.laser.shape[0]) 

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
        k = 0.4
        alpha = 0.9

        cost = (v_x + k*(self.pos_x-self.goal_x))**2 + (v_y + k*(self.pos_y-self.goal_y))**2      
        constraints = []

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
        theta = np.arctan2(v_y, v_x)

        v = np.sqrt((v_x**2) + (v_y**2))
        v = np.clip(v, 0, 0.3)
        vel_x = v * np.cos(theta)
        vel_y = v * np.sin(theta)

        rot_matrix = np.array([[np.cos(self.yaw), np.sin(self.yaw)],
                                [-np.sin(self.yaw), np.cos(self.yaw)]])
        v = np.dot(rot_matrix, np.array([vel_x, vel_y]))

        # self.vel.linear.x = v_x
        # self.vel.linear.y = v_y

        self.vel.linear.x = v[0]
        self.vel.linear.y = v[1]

        self.vel_publisher.publish(self.vel) 

        end_time = time.time()

        print(f"Took {end_time-start_time} seconds")


        # # self.laser = np.array(self.laser)
        # ranges = laser_msg.ranges

        # # Find the index of the minimum range value
        # min_index = ranges.index(min(ranges))

        # # Get the corresponding angle for the closest point
        # angle_min = laser_msg.angle_min
        # angle_increment = laser_msg.angle_increment
        # closest_point_angle = angle_min + min_index * angle_increment

        # # Get the range of the closest point
        # closest_point_range = ranges[min_index]      

        # local_laser = np.array([closest_point_angle, closest_point_range])
        # print(f"closest point angle, range: {self.local_laser}")

        # closest_angle = local_laser[0]
        # closest_distance = local_laser[1]

        # x_obs = closest_distance * np.cos(closest_angle)
        # y_obs = closest_distance * np.sin(closest_angle)

        # x_obs_global = x_obs + self.pos_x
        # y_obs_global = y_obs + self.pos_y

        # print(f"Pos X: {self.pos_x}")
        # print(f"Pos Y: {self.pos_y}")

        # hx = np.sqrt((self.pos_x - x_obs_global)**2 + (self.pos_y - y_obs_global)**2) - self.obs_D
        # grad_hx = np.vstack(((self.pos_x-x_obs_global)/(hx+self.obs_D), (self.pos_y-y_obs_global)/(hx+self.obs_D)))

        # xi = cp.Variable(2)
        # v_x = xi[0]
        # v_y = xi[1]
        # k = 1.0
        # alpha = 0.5

        # cost = (v_x + k*(self.pos_x-self.goal_x))**2 + (v_y + k*(self.pos_y-self.goal_y))**2      
        # constraints = [grad_hx[0]*v_x + grad_hx[1]*v_y + alpha * hx >= 0]                        
        
        # prob = cp.Problem(cp.Minimize(cost),
        #                 constraints)
        # prob.solve(solver=cp.ECOS)

        # v_x = xi.value[0]
        # v_y = xi.value[1]
        # theta = np.arctan2(v_y, v_x)

        # v = np.sqrt((v_x**2) + (v_y**2))
        # v = np.clip(v, 0, 0.3)
        # vel_x = v * np.cos(theta)
        # vel_y = v * np.sin(theta)

        # rot_matrix = np.array([[np.cos(self.yaw), np.sin(self.yaw)],
        #                         [-np.sin(self.yaw), np.cos(self.yaw)]])
        # v = np.dot(rot_matrix, np.array([vel_x, vel_y]))

        # # self.vel.linear.x = v_x
        # # self.vel.linear.y = v_y

        # self.vel.linear.x = v[0]
        # self.vel.linear.y = v[1]

        # self.vel_publisher.publish(self.vel) 

        
       
        
#------------------------------------------
    
    def handle_odom(self, odom_msg):
        # self.pose = odom_data.pose

        self.pos_x = odom_msg.pose.pose.position.x
        self.pos_y = odom_msg.pose.pose.position.y

        orientation = odom_msg.pose.pose.orientation
        quaternion = [orientation.x, orientation.y, orientation.z, orientation.w] 
        roll, pitch, yaw = transformations.euler_from_quaternion(quaternion)

        self.yaw = yaw

        self.position_x.append(self.pos_x)
        self.position_y.append(self.pos_y)

        # pos_x = odom_msg.pose.pose.position.x
        # pos_y = odom_msg.pose.pose.position.y

        # print(f"Current position: {pos_x, pos_y}")

        # # dist_to_goal = np.sqrt((pos_x-self.goal_x)**2 + (pos_y-self.goal_y)**2)

        # # print(f"Pos X: {pos_x}")        
        # # print(f"Pos Y: {pos_y}")     

        # closest_obs = self.local_laser
        # # angle, distance = self.local_laser
        # closest_angle = closest_obs[0]
        # closest_distance = closest_obs[1]

        # x_obs = closest_distance * np.cos(closest_angle)
        # y_obs = closest_distance * np.sin(closest_angle)

        # x_obs_global = x_obs + pos_x
        # y_obs_global = y_obs + pos_y

        # # print(f"x obs global: {x_obs_global}")
        # # print(f"y obs global: {y_obs_global}")

        # hx = np.sqrt((pos_x - x_obs_global)**2 + (pos_y - y_obs_global)**2) - self.obs_D
        # grad_hx = np.vstack(((pos_x-x_obs_global)/(hx+self.obs_D), (pos_y-y_obs_global)/(hx+self.obs_D)))

        # xi = cp.Variable(2)
        # v_x = xi[0]
        # v_y = xi[1]
        # k = 1.0
        # alpha = 0.4

        # cost = (v_x + k*(pos_x-self.goal_x))**2 + (v_y + k*(pos_y-self.goal_y))**2
     
        # constraints = [grad_hx[0]*v_x + grad_hx[1]*v_y + alpha * hx >= 0,                        
        #                cp.norm(xi, 2) <= 0.3]
        
        # prob = cp.Problem(cp.Minimize(cost),
        #                 constraints)
        # prob.solve()


        # v_x = xi.value[0]
        # v_y = xi.value[1]

        # self.vel.linear.x = v_x
        # self.vel.linear.y = v_y

        # self.vel_publisher.publish(self.vel)  
        # 
        # ----- 

        # if self.local_laser is not None:

        #     closest_obs = self.local_laser
        #     # angle, distance = self.local_laser
        #     closest_angle = closest_obs[0]
        #     closest_distance = closest_obs[1]

        #     x_obs = closest_distance * np.cos(closest_angle)
        #     y_obs = closest_distance * np.sin(closest_angle)

        #     x_obs_global = x_obs + pos_x
        #     y_obs_global = y_obs + pos_y

        #     # print(f"x obs global: {x_obs_global}")
        #     # print(f"y obs global: {y_obs_global}")

        #     hx = np.sqrt((pos_x - x_obs_global)**2 + (pos_y - y_obs_global)**2) - self.obs_D
        #     grad_hx = np.vstack(((pos_x-x_obs_global)/(hx+self.obs_D), (pos_y-y_obs_global)/(hx+self.obs_D)))

        #     self.xi = cp.Variable(2)
        #     self.v_x = self.xi[0]
        #     self.v_y = self.xi[1]
        #     self.k = 1.0
        #     self.alpha = 0.4

        #     cost = (self.v_x + self.k*(pos_x-self.goal_x))**2 + (self.v_y + self.k*(pos_y-self.goal_y))**2
        #     # print(f"cost: {cost}")
        #     # constraints = [grad_hx[0]*self.v_x + grad_hx[1]*self.v_y + self.alpha * hx >= 0, cp.norm(self.xi, 2) <= 0.3]
        #     constraints = [grad_hx[0]*self.v_x + grad_hx[1]*self.v_y + self.alpha * hx >= 0,                        
        #                 cp.norm(self.xi, 2) <= 0.3]
        #     # print(f"constraints: {constraints}")

        #     prob = cp.Problem(cp.Minimize(cost),
        #                     constraints)
        #     prob.solve()

        #     self.v_x = self.xi.value[0]
        #     self.v_y = self.xi.value[1]

        #     self.vel.linear.x = self.v_x
        #     self.vel.linear.y = self.v_y

        #     self.vel_publisher.publish(self.vel)

            # hx = closest_distance    
            # grad_hx = np.array([-np.cos(closest_angle), -np.sin(closest_angle)])

            # self.xi = cp.Variable(2)
            # self.v_x = self.xi[0]
            # self.v_y = self.xi[1]
            # self.k = 1.0
            # self.alpha = 0.4
            
            # cost = (self.v_x + self.k*(pos_x-self.goal_x))**2 + (self.v_y + self.k*(pos_y-self.goal_y))**2
            # constraints = [grad_hx[0]*self.v_x + grad_hx[1]*self.v_y + self.alpha * hx >= 0,
            #                cp.norm(self.xi, 2) <= 0.3]
            # # print(f"constraints: {constraints}")

            # prob = cp.Problem(cp.Minimize(cost),
            #                 constraints)
            # prob.solve()

            # self.v_x = self.xi.value[0]
            # self.v_y = self.xi.value[1]
            # # print(f"v_x, vy: {self.v_x, self.v_y}")
            # # theta = np.arctan2(self.v_y, self.v_x)

            # # v = np.sqrt(self.v_x**2 + self.v_y**2)
            # # v = np.clip(v, 0, 0.2)
            # # v_x = v * np.cos(theta)
            # # v_y = v * np.sin(theta)

            # self.vel.linear.x = self.v_x
            # self.vel.linear.y = self.v_y

            # self.vel_publisher.publish(self.vel)
            # # print(f"Velocities published: {self.v_x, self.v_y}")
        
        
        # end_time = time.time()

        # print(f"Time taken by cvxpy to compute: {end_time-start_time} seconds")
# ------------------------------------------------------------------


if __name__ == "__main__":
    rospy.init_node("CBF_BEBOP", anonymous=True)
    node = cbf_bebop()
    rospy.spin()

    plt.figure(figsize=(10, 10))
    plt.title('Drone Trajectory under CBF')
    plt.scatter(node.position_x, node.position_y)  

    # obs_x = np.array([1.0, 2.5])
    # obs_y = np.array([2.0, 3.0])

    # obs_x = np.array([0.0, 3.0])
    # obs_y = np.array([2.0, 3.0])

    obs_x = np.array([3.0, 5.0])
    obs_y = np.array([3.0, 5.0])
    
    theta = np.linspace(0, 2*np.pi, 100)
    r = 0.5
    # circle 1
    x1 = obs_x[0] + r * np.cos(theta)
    y1 = obs_y[0] + r * np.sin(theta)
    plt.plot(x1, y1)

    # circle 2
    x2 = obs_x[1] + r * np.cos(theta)
    y2 = obs_y[1] + r * np.sin(theta)
    plt.plot(x2, y2)

    plt.scatter(node.goal_x, node.goal_y)

    plt.show()


    
