#! /usr/bin/env python3

import roslib, sys, rospy
from sensor_msgs.msg import LaserScan, Joy
from geometry_msgs.msg import Point, Twist, PoseStamped
from std_msgs.msg import Float32
import numpy as np
import math
from nav_msgs.msg import Odometry
from tf import transformations

#-------------------------------------------------

class safe_teleop:
    def __init__(self):
        rospy.init_node("command_velocity")
        self.mode = "hardware" # simulation

        rospy.Subscriber("potential_field_vector", Point, self.handle_potential_field)
        # self.cmd_pub = rospy.Publisher("cmd_vel_robotont", Twist,queue_size=10)
        self.cmd_pub = None
        if self.mode == "simulation":
            self.cmd_pub = rospy.Publisher("bebop/cmd_vel", Twist,queue_size=10)
            
        else:
            self.cmd_pub = rospy.Publisher("bebop/velocity", Twist,queue_size=10)
            self.pose_sub = rospy.Subscriber("vrpn_client_node/bebop/pose", PoseStamped, self.pose_callback, queue_size=1)

        self.yaw = None
        self.odom = None
        self.obstacle_vector = None
        self.min_vel_x = -1.0
        self.max_vec_x = 1.0
        self.min_vel_y = -1.0
        self.max_vec_y = 1.0
        self.drive_scale = 0.1 # scaling factor to scale the net force

#-------------------------------------------------

    def start(self):
        rate = rospy.Rate(rospy.get_param("~cmd_rate", 10))
        while not rospy.is_shutdown():
            cmd = self.compute_motion_cmd()
            if cmd != None:
                self.cmd_pub.publish(cmd)
            rate.sleep()

#---------------------------------------------------
    
    def pose_callback(self, pose_msg):
        orientation = pose_msg.pose.orientation
        quaternion = [orientation.x, orientation.y, orientation.z, orientation.w] 
        roll, pitch, yaw = transformations.euler_from_quaternion(quaternion)

        self.yaw = yaw


#-------------------------------------------------

    def compute_motion_cmd(self):
        if (self.obstacle_vector is None):
            cmd = None
        else:
            cmd = Twist()

            # We use velocity based potential field,that is, the gradient/force is directly commanded as velocities
            # instead of force or acceleration. 

            if self.yaw is not None:

                vel_x = self.obstacle_vector[0]*self.drive_scale
                vel_y = self.obstacle_vector[1]*self.drive_scale
                theta = np.arctan2(vel_y, vel_x)
                
                v = np.sqrt((vel_x**2) + (vel_y**2))
                v = np.clip(v, 0, 0.3)
                vel_x = v * np.cos(theta)
                vel_y = v * np.sin(theta)

                rot_matrix = np.array([[np.cos(self.yaw), np.sin(self.yaw)],
                                    [-np.sin(self.yaw), np.cos(self.yaw)]])
                v = np.dot(rot_matrix, np.array([vel_x, vel_y]))

                # cmd.linear.x = np.clip(vel_x,self.min_vel_x,self.max_vec_x)
                # cmd.linear.y = np.clip(vel_y,self.min_vel_y,self.max_vec_y)

                cmd.linear.x = v[0]
                cmd.linear.y = v[1]

        return cmd

#-------------------------------------------------

    def handle_potential_field(self, potential_field):
        self.obstacle_vector = np.array([potential_field.x, potential_field.y])

if __name__ == "__main__":
    st = safe_teleop()
    st.start()