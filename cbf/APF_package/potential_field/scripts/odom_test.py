import rospy

from nav_msgs.msg import Odometry

class CBF():

    def __init__(self):

        rospy.Subscriber('/bebop/odom',
                         Odometry,
                         self.odom_callback,
                         queue_size=1)

    def odom_callback(self, odom_msg):
        pos_x = odom_msg.pose.pose.position.x
        pos_y = odom_msg.pose.pose.position.y

        print(f"Current position: {pos_x, pos_y}")        


if __name__ == "__main__":
    rospy.init_node("control_barrier_function_test")
    node = CBF()
    rospy.spin()

    