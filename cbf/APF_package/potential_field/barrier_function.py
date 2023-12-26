#! /usr/bin/env python3

from cmath import inf
from visualization_msgs.msg import Marker,MarkerArray
import roslib, sys, rospy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry,Path
from geometry_msgs.msg import Point,PoseStamped
import numpy as np
import math
import time
import cvxpy as cp
#------------------------------------------

class barrier_function:
    def __init__(self):
        rospy.init_node("barrier_function")

        self.sample_rate = rospy.get_param("~sample_rate", 10)

        # Subscribe to the global planner using the move base package. The global plan is the path that the robot would ideally follow if 
        # there are no unknown/dynamic obstacles. In the videos this is highlighted by green color.
        self.global_path_sub = rospy.Subscriber("/move_base/NavfnROS/plan", Path, self.handle_global_path)

        self.laser_sub = rospy.Subscriber("/scan", LaserScan, self.handle_laser)
        self.odom_sub = rospy.Subscriber("/odom", Odometry, self.handle_odom)
        
        # Subscribe to the goal topic to get the goal position given using rviz's 2D Navigation Goal option.
        self.goal_sub = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.handle_goal)

        # Publish the potential field vector topic which will be subscribed by the command_velocity node in order to
        # compute velocities.
        self.potential_field_pub = rospy.Publisher("potential_field_vector", Point,queue_size=10)

        # We store the path data gotten from the global planner above and display it. We have written a custom publisher 
        # in order to get more flexibility while displaying the paths.
        self.global_path_pub = rospy.Publisher("global_path",Path,queue_size=10)

        # This is a publisher to publish the robot path. In the videos it is highlighted by red color.
        self.robot_path_pub = rospy.Publisher("robot_path",Path,queue_size=10)
        
        self.path_robot = Path()
        self.path_robot.header.frame_id = 'map'

        # self.xi = cp.Variable(2)

        ## TODO Choose suitable values
        self.eta = "" # scaling factor for repulsive force
        self.zeta = "" # scaling factor for attractive force
        self.q_star = "" # threshold distance for obstacles
        self.d_star = "" # threshoild distance for goal

        self.laser = None
        self.odom = None
        self.goal = None

        self.path_data = Path()
        self.path_data.header.frame_id = 'map'
        
        self.position_x = []
        self.position_y = []
        self.position_all = []
        
        # Boolean variables used for proper display of robot path and global path
        self.bool_goal = False
        self.bool_path = False
#------------------------------------------

    
    def start(self):
    
        rate = rospy.Rate(self.sample_rate)
        while not rospy.is_shutdown():
            if(self.path_data):
                self.global_path_pub.publish(self.path_data)
           
            self.robot_path_publish()
            net_force=self.barrier_function(np.array([[4.907160,0.007870],[-1.535369,2.519666]]))
            # print(net_force)

            # net_force = np.array([0.5,0.5]) ## What should be the net force?
            self.publish_sum(net_force[0],net_force[1])

            rate.sleep()
    
    def barrier_function(self,objects_cord):
        if self.odom == None or self.goal == None:
            return (0,0)

        odom_data = self.odom
        pos_x = odom_data.pose.pose.position.x
        pos_y = odom_data.pose.pose.position.y

        goal_X=self.goal.pose.position.x
        goal_y=self.goal.pose.position.y

        # print(self.laser.shape)
        xi = cp.Variable(2)

        

        v_x, v_y = xi

        k=1
        alpha = 0.4

        cost = (v_x + k*(pos_x-goal_X))**2 + (v_y + k*(pos_y-goal_y))**2

        local_laser=self.laser

        closest_laser_index=local_laser[:,1].argsort()[:50]
        chosen_beam=local_laser[closest_laser_index]
        # print(chosen_beam)



        beam_grad_x=-np.cos(chosen_beam[:,0])
        beam_grad_y=-np.sin(chosen_beam[:,0])
        range_to_obs=chosen_beam[:,1]
        



        # constraints = [grad_hx*v_x + grad_hy*v_y + alpha * dist_obs >= 0]
        constraints = [beam_grad_x*v_x + beam_grad_y*v_y + alpha * range_to_obs >= 0]

        prob = cp.Problem(cp.Minimize(cost), constraints)
        
        prob.solve()



        return xi.value

#------------------------------------------
    def robot_path_publish(self):
        if(self.odom):
            odom_data = self.odom
            if(self.bool_path == True):
                self.bool_path = False
                self.path_robot = Path()
                self.path_robot.header.frame_id = 'map'
            pose = PoseStamped()
            pose.header = odom_data.header
            pose.pose = odom_data.pose.pose
            self.path_robot.poses.append(pose)
            self.robot_path_pub.publish(self.path_robot)

#------------------------------------------





    def handle_laser(self, laser_data):
        angles = np.arange(-3.141590118408203, 3.141590118408203, 0.008738775737583637)
        ranges = np.array(laser_data.ranges)

# Filter the data
        valid_indices = np.isfinite(ranges)
        filtered_data = np.column_stack((angles[valid_indices], ranges[valid_indices]))

# Print or use filtered_data as needed
        # print(filtered_data)
        # print(laser_data)
        self.laser = filtered_data

        
#------------------------------------------
    
    def handle_odom(self, odom_data):
        self.odom = odom_data
#------------------------------------------
    
    def handle_goal(self, goal_data):
        self.bool_goal = True
        self.bool_path = True
        self.goal = goal_data
#------------------------------------------
    def publish_sum(self, x, y):
        vector = Point(x, y, 0)
        self.potential_field_pub.publish(vector)

#------------------------------------------
    def publish_dist_to_goal(self, dist):
        dist_to_goal = Float32(dist)
        self.dist_to_goal_pub.publish(dist_to_goal)
#------------------------------------------

    def handle_global_path(self, path_data):
        if(self.bool_goal == True):
            self.bool_goal = False
            self.path_data = path_data
            i=0
            while(i <= len(self.path_data.poses)-1):
                self.position_x.append(self.path_data.poses[i].pose.position.x)
                self.position_y.append(self.path_data.poses[i].pose.position.y)
                i=i+1
            self.position_all = [list(double) for double in zip(self.position_x,self.position_y)]

if __name__ == "__main__":
    pf = barrier_function()
    pf.start()
