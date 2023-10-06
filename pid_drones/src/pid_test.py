import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Twist
import numpy as np

goal_position = PoseStamped()
goal_position.header.frame_id = "world"
goal_position.pose.position.x = 10.0
goal_position.pose.position.y = 0.0
goal_position.pose.position.z = 1.0

kp = 1
kd = 0.01

input_vel = Twist()

vel_pub = rospy.Publisher('cmd_vel',
                        Twist,
                        queue_size=1)



def odom_callback(msg):
    current_x = msg.pose.pose.position.x
    current_y = msg.pose.pose.position.y
    current_z = msg.pose.pose.position.z

    target_x = goal_position.pose.position.x
    target_y = goal_position.pose.position.y
    target_z = goal_position.pose.position.z

    # error = np.abs(current_x - target_x)
    error = target_x - current_x
    rospy.loginfo("error: %s", str(error))
    print("-----------------------------")
    vx = kp * error + kd * error
    vx = np.clip(vx, 0, 0.2)

    input_vel.linear.x = vx

    vel_pub.publish(input_vel)

    # if error <= 0.1:
    #     rospy.loginfo("Goal reached")
    #     rospy.signal_shutdown("Goal Reached")
    #     print("-----------------------------")
    #     print("-----------------------------")

def odom_listener():

    rospy.init_node('pid_test')
    
    rospy.Subscriber('bebop/odom',
                    Odometry,
                    callback=odom_callback,
                    queue_size=1)
    rospy.spin()

if __name__ == "__main__":
    odom_listener()








