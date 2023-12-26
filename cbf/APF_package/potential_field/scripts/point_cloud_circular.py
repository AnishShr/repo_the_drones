#!/usr/bin/env python3
import rospy
import math
import sys

from sensor_msgs.msg import PointCloud2
import std_msgs.msg
import sensor_msgs.point_cloud2 as pcl2
# import geometry_msgs.msg
from geometry_msgs.msg import PoseStamped
import numpy as np
from message_filters import ApproximateTimeSynchronizer, Subscriber
import pdb
# from  #TODO # import Obsposelist message
h = 1.0
radius = 1.0
pcl_pub = None
def create_a_circle(center_x,center_y):
    # print(center_x, center_y)
    list_circular_point= []
    points = np.arange(0,2*np.pi,np.pi/180)
    x_points = center_x.reshape(-1,1) + 0.5 * np.cos(points) #TODO #get list of  x coordinates which is on the circumference of circle with center center_x 
    y_points = center_y.reshape(-1,1) + 0.5 * np.sin(points) #get list of  y coordinates which is on the circumference of circle with center center_y 
    z_points = h * np.ones((center_x.shape[0],points.shape[0])) #get the list z_points , in our case z points represent constant height h ,defined above .This list should be of same lenght of x_points  

    # points = [(x, y, z) for x, y, z in zip(x_points, y_points, z_points)] #TODO stack (x_points,y_points,z_points) 
    #points [[x1,y1,z1],[x2,y2,z2]......[xn,yn,zn]]
    return x_points, y_points, z_points
    

def filter_callback(box1_msg, box2_msg, box3_msg):
    global pcl_pub

    list_centers_x = np.array([box1_msg.pose.position.x, box2_msg.pose.position.x, box3_msg.pose.position.x])
    list_centers_y = np.array([box1_msg.pose.position.y, box2_msg.pose.position.y, box3_msg.pose.position.y])
                    
    # list_centers_x = np.array([box1_msg.pose.position.x, box2_msg.pose.position.x])
    # list_centers_y = np.array([box1_msg.pose.position.y, box2_msg.pose.position.y])
    
    list_circular_point_list = []
    
    # for i in range(len(list_centers)):
    center_x = list_centers_x
    center_y = list_centers_y
    xp,yp,zp = create_a_circle(center_x, center_y)

    # list_circular_point_list.append(points_circle)
    xp,yp,zp = xp.reshape(-1),yp.reshape(-1),zp.reshape(-1)
    cloud_points = np.vstack((xp,yp,zp)).T
    # print(list_circular_point_list[0])
    
    # cloud_points =  list_circular_point_list #[[1.0, 1.0, 0.0],[1.0, 2.0, 0.0]]
    #header
    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()
    #header.frame_id = 'bebopbase_footprint'
    header.frame_id = 'world'
    #create pcl from points
    scaled_polygon_pcl = pcl2.create_cloud_xyz32(header, cloud_points)
    # print(scaled_polygon_pcl.shape)
    #publish    
    pcl_pub.publish(scaled_polygon_pcl)

    
    
if __name__ == '__main__':
    try:
        rospy.init_node('pcl2_obs_pub')
        pcl_pub = rospy.Publisher("/obs1_pcl_topic", PointCloud2, queue_size=10) 
        r = rospy.Rate(10) # 10hz
        
        #TODO subscribe to the topic publishing list of obstacle's poses  (created in lab8_part1) .callback function for this topic publish_pointcloud
        # rospy.Subscriber(
        #     '/vrpn_client_node/hat/pose',
        #     PoseStamped,
        #     publish_pointcloud, 
        #     queue_size=1
        # )

        box1_topic = '/vrpn_client_node/box1/pose'
        box2_topic = '/vrpn_client_node/box2/pose'
        box3_topic = '/vrpn_client_node/box3/pose'

        box1_sub = Subscriber(box1_topic, PoseStamped)
        box2_sub = Subscriber(box2_topic, PoseStamped)
        box3_sub = Subscriber(box3_topic, PoseStamped)
        ats = ApproximateTimeSynchronizer([box1_sub, box2_sub, box3_sub], queue_size=10, slop=0.1)
        # ats = ApproximateTimeSynchronizer([box1_sub, box2_sub], queue_size=1, slop=0.1)
        ats.registerCallback(filter_callback)

        rospy.spin()
    except rospy.ROSInterruptException:  pass
    