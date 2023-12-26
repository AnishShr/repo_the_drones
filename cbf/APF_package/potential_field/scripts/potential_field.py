#! /usr/bin/env python3

from cmath import inf
from visualization_msgs.msg import Marker,MarkerArray
import roslib, sys, rospy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry,Path
from geometry_msgs.msg import Point, PoseStamped
import numpy as np
import math
import time
#------------------------------------------

class potential_field:
    def __init__(self):
        rospy.init_node("potential_field")

        self.sample_rate = rospy.get_param("~sample_rate", 10)
        self.mode = "simulation" # simulation

        # Subscribe to the global planner using the move base package. The global plan is the path that the robot would ideally follow if 
        # there are no unknown/dynamic obstacles. In the videos this is highlighted by green color.
        # self.global_path_sub = rospy.Subscriber("/move_base/NavfnROS/plan", Path, self.handle_global_path)
        self.global_path_sub = rospy.Subscriber("/move_base/GlobalPlanner/plan", Path, self.handle_global_path)

        if self.mode == "simulation":
            self.laser_sub = rospy.Subscriber("/bebop/laser_scan", LaserScan, self.handle_laser)
            self.odom_sub = rospy.Subscriber("/bebop/odom", Odometry, self.handle_odom)
            # Subscribe to the goal topic to get the goal position given using rviz's 2D Navigation Goal option.
            self.goal_sub = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.handle_goal)
        else:
            self.laser_sub = rospy.Subscriber("/scan", LaserScan, self.handle_laser)
            self.odom_sub = rospy.Subscriber("/vrpn_client_node/bebop/pose", PoseStamped, self.handle_odom)
            # Subscribe to the goal topic to get the goal position given using rviz's 2D Navigation Goal option.
            self.goal_sub = rospy.Subscriber("/move_base/GlobalPlanner/plan", PoseStamped, self.handle_goal)
        

        # Publish the potential field vector topic which will be subscribed by the command_velocity node in order to
        # compute velocities.
        self.potential_field_pub = rospy.Publisher("potential_field_vector", Point,queue_size=10)

        # We store the path data gotten from the global planner above and display it. We have written a custom publisher 
        # in order to get more flexibility while displaying the paths.
        self.global_path_pub = rospy.Publisher("global_path",Path,queue_size=10)

        # This is a publisher to publish the robot path. In the videos it is highlighted by red color.
        self.robot_path_pub = rospy.Publisher("robot_path",Path,queue_size=10)
        
        self.path_robot = Path()
        self.path_robot.header.frame_id = 'world'

        ## TODO Choose suitable values
        self.eta = 10 # scaling factor for repulsive force
        self.zeta = 2 # scaling factor for attractive force
        self.q_star = 1 # threshold distance for obstacles
        self.d_star = 2 # threshoild distance for goal

        self.laser = None
        self.odom = None
        self.goal = None

        self.path_data = Path()
        self.path_data.header.frame_id = 'world'
        
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
            # net_force = "" ## What should be the net force?
            # net_force = np.array([0.5, 0.5])
            # net_force = np.array([0, 0])
            net_force = self.compute_attractive_force() + self.compute_repulsive_force()
            # net_force = self.compute_attractive_force()
            self.publish_sum(net_force[0],net_force[1])

            rate.sleep()

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
            # pose.pose = odom_data.pose.pose
            pose.pose = odom_data.pose
            self.path_robot.poses.append(pose)
            self.robot_path_pub.publish(self.path_robot)

#------------------------------------------

    def compute_repulsive_force(self):
        if self.laser == None or self.odom == None or self.goal == None:
            return (0,0)
    
        laser_data = self.laser
        ranges = np.array(laser_data.ranges)
        angle = laser_data.angle_min
        resolution = laser_data.angle_increment
        vector_sum = np.array([0.0,0.0])
        
        vector = np.array([0, 0])

        # The lidar outputs 360 degree data and therefore we get data of 360 points within the range of the sensor.
        # Each obstacle detected by the sensor will occupy some points out of these 360 points. We treat each single point as 
        # an individual obstacle.
        for distance in ranges:
            if(distance<self.q_star):
                # mag = "" ## What should be the magnitude of the repulsive force generated by each obstacle point?
                grad = self.eta * (1/(self.q_star) - 1/(distance)) * 1/(distance**2)
            else:
                # mag = "" ## What should be the magnitude of repulsive force outside the region of "influence"?
                grad = 0
             
            # This is the negative gradient direction
            # vector = (-1) * vector
            # vector = np.array([grad, grad])
            # vector[0] = grad * np.cos(angle)
            # vector[1] = grad * np.sin(angle)

            vector[0] = grad * np.cos(angle)
            vector[1] = grad * np.sin(angle)

            vector_sum += vector ## You need to add the effect of all obstacle points
            angle += resolution
        
        

        # Normalization of the repulsive forces
        return np.array([vector_sum[0],vector_sum[1]])*(1/len(ranges))
#------------------------------------------

    def compute_attractive_force(self):
        if self.odom == None or self.goal == None:
            return (0,0)

        odom_data = self.odom
        # pos_x = odom_data.pose.pose.position.x
        # pos_y = odom_data.pose.pose.position.y
        pos_x = odom_data.pose.position.x
        pos_y = odom_data.pose.position.y
        pos = []
        pos.append(pos_x)
        pos.append(pos_y)

        closest_waypoint = []
        while(not closest_waypoint or closest_waypoint is None or not closest_waypoint[1]):
            closest_waypoint = self.closest_waypoint(pos, self.position_all)

        # dist_to_goal = np.sqrt((pos_x - self.goal.pose.position.x)**2 + (pos_y - self.goal.pose.position.y)**2)
        

        index_waypoint = closest_waypoint[0]

        if index_waypoint < len(self.position_all) - 50:
            desired_waypoint = index_waypoint + 50
        else:
            desired_waypoint = len(self.position_all) - 1

        # print(self.position_all[desired_waypoint][0])
        

        dist_to_goal = np.sqrt((pos_x - self.position_all[desired_waypoint][0])**2 + (pos_y - self.position_all[desired_waypoint][1])**2)

        # If the closest waypoint is the last point on the global path then you might want to direct the robot towards the goal position
        ########

        ## Your stuff

        ########
    
        # You can use a lookahead distance concept in order to drive the robot. Feel free to customize this block of code.
        # The robot follows the waypoint which is say,10 points ahead of the closest waypoint. If the robot is within 10 points
        # from the goal ,use the goal position instead to drive the robot.
        
        ###

        ## Your stuff

        ###

        if (dist_to_goal<=self.d_star):
            # mag = "" # Magnitude of attractive force
            # vector = self.zeta * (np.array(pos) - np.array([self.goal.pose.position.x, self.goal.pose.position.y]))
            vector = self.zeta * (np.array(pos) - np.array([self.position_all[desired_waypoint]]))
        else :
            # mag = ""
            # vector = self.d_star * self.zeta * (np.array(pos) - np.array([self.goal.pose.position.x, self.goal.pose.position.y])) / (dist_to_goal)
            vector = self.d_star * self.zeta * (np.array(pos) - np.array([self.position_all[desired_waypoint]])) / (dist_to_goal)

        vector = (-1) * vector
        # print(vector)

        # return np.array([vector[0], vector[1]])
        return np.array([vector[0][0],vector[0][1]])
#------------------------------------------

    def closest_waypoint(self,point, points):
        i=0
        pt=[]
        dist = math.inf
        for p in points:
            if(math.dist(p,point)<dist):
                dist = math.dist(p,point)
                pt = p
                i = points.index(pt)
        return [i,pt]
#------------------------------------------

    def handle_laser(self, laser_data):
        self.laser = laser_data
        
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
            self.position_x = []
            self.position_y = []
if __name__ == "__main__":
    pf = potential_field()
    pf.start()
