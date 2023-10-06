import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Twist
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

class TrajectoryTracking:
    def __init__(self, ref_traj_x, ref_traj_y, mode):
        
        self.mode = mode

        self.v = 0.5
        self.dt = 0.1

        self.ref_traj_x = ref_traj_x
        self.ref_traj_y = ref_traj_y

        self.path_cubic_spline = self.path_spline(self.ref_traj_x, self.ref_traj_y)
        self.cs_x_path = self.path_cubic_spline[0]
        self.cs_y_path = self.path_cubic_spline[1]
        self.cs_phi_path = self.path_cubic_spline[2]
        self.cs_arc_length = self.path_cubic_spline[3]
        self.cs_arc_vec = self.path_cubic_spline[4]

        #pid coeffs
        self.kp = 0.8 #3.0 #0.8 
        # self.kd = (1/2) * np.sqrt(self.kp) # 0.47 # 0.5  
        self.kd = 2.0 * np.sqrt(self.kp)
        # self.kd = 0.6     

        self.drone_traj_x = []
        self.drone_traj_y = []

        self.goal_x = self.ref_traj_x[-1]
        self.goal_y = self.ref_traj_y[-1]

        self.cmd_vel = Twist()

        self.prev_pose = None
        self.prev_time = None

        self.vx = None
        self.vy = None

        self.x_cur = 0.0
        self.y_cur = 0.0

        self.x_dot_cur = 0.0
        self.y_dot_cur = 0.0

        if self.mode == "simulation":
            self.vel_pub = rospy.Publisher('bebop/cmd_vel',
                                    Twist,
                                    queue_size=1)
            
            rospy.Subscriber('bebop/odom',
                         Odometry,
                         self.bebop_callback,
                         queue_size=1)
            
        elif self.mode == "hardware": 
            self.vel_pub = rospy.Publisher('bebop/velocity',
                                    Twist,
                                    queue_size=1)

            rospy.Subscriber('vrpn_client_node/bebop/pose',
                            PoseStamped,
                            self.bebop_callback,
                            queue_size=1)


    def path_spline(self, x_path, y_path):
        x_diff = np.diff(x_path)
        y_diff = np.diff(y_path)

        phi = np.unwrap(np.arctan2(y_diff, x_diff))
        phi_init = phi[0]
        phi = np.hstack(( phi_init, phi  ))

        arc = np.cumsum( np.sqrt( x_diff**2 + y_diff**2 ))
        arc_length = arc[-1]
        arc_vec = np.linspace(0, arc_length, np.shape(x_path)[0])
        
        cs_x_path = CubicSpline(arc_vec, x_path)
        cs_y_path = CubicSpline(arc_vec, y_path)
        cs_phi_path = CubicSpline(arc_vec, phi)

        return cs_x_path, cs_y_path, cs_phi_path, arc_length, arc_vec

    def waypoint_generator(self, x_global_init, y_global_init, x_path_data, y_path_data, arc_vec, cs_x_path, cs_y_path, cs_phi_path, arc_length):
        idx = np.argmin( np.sqrt((x_global_init-x_path_data)**2+(y_global_init-y_path_data)**2))
        
        arc_curr = arc_vec[idx]
        arc_pred = arc_curr + self.v*self.dt
        
        x_waypoints = cs_x_path(arc_pred)
        y_waypoints =  cs_y_path(arc_pred)
        phi_Waypoints = cs_phi_path(arc_pred)

        x_dot = cs_x_path.derivative()
        x_dot_waypoints = x_dot(arc_pred)*self.v

        y_dot = cs_y_path.derivative()
        y_dot_waypoints = y_dot(arc_pred)*self.v

        return x_waypoints, y_waypoints, phi_Waypoints, x_dot_waypoints, y_dot_waypoints
    
    def bebop_callback(self, msg):
               

        if self.mode == "simulation":

            self.x_cur = msg.pose.pose.position.x
            self.y_cur = msg.pose.pose.position.y

            self.x_dot_cur = msg.twist.twist.linear.x
            self.y_dot_cur = msg.twist.twist.linear.y
        
        elif self.mode == "hardware":

            self.x_cur = msg.pose.position.x
            self.y_cur = msg.pose.position.y

            current_time = rospy.Time.now()

            if self.prev_pose is not None:
                
                delta_t = (current_time - self.prev_time).to_sec()
                
                self.x_dot_cur = (self.x_cur - self.prev_pose.position.x) / delta_t
                self.y_dot_cur = (self.y_cur - self.prev_pose.position.y) / delta_t                    

                print(f"vel_x: {self.x_dot_cur}")
                print(f"vel_y: {self.y_dot_cur}")


            self.prev_pose = msg.pose
            self.prev_time = current_time


        actual_traj_x.append(self.x_cur)
        actual_traj_y.append(self.y_cur)

        x_path_data = self.ref_traj_x
        y_path_data = self.ref_traj_y

        cs_x_path = self.cs_x_path
        cs_y_path = self.cs_y_path
        cs_phi_path = self.cs_phi_path

        arc_vec = self.cs_arc_vec
        arc_length = self.cs_arc_length

        # desired x, y and z
        x_waypoints, y_waypoints, phi_waypoints, x_dot_waypoints, y_dot_waypoints = self.waypoint_generator(x_global_init = self.x_cur, 
                                                                          y_global_init = self.y_cur, 
                                                                          x_path_data = x_path_data, 
                                                                          y_path_data = y_path_data, 
                                                                          arc_vec = arc_vec, 
                                                                          cs_x_path = cs_x_path, 
                                                                          cs_y_path = cs_y_path, 
                                                                          cs_phi_path = cs_phi_path, 
                                                                          arc_length = arc_length)        



        rospy.loginfo("Desired x, y, phi: %s, %s, %s", x_waypoints, y_waypoints, phi_waypoints)
        rospy.loginfo("Error x, y: %s, %s", x_waypoints-self.x_cur, y_waypoints-self.y_cur)

        if self.mode == "simulation":
            self.vx = self.kp * (x_waypoints - self.x_cur) + self.kd * (x_dot_waypoints - self.x_dot_cur)
            self.vy = self.kp * (y_waypoints - self.y_cur) + self.kd * (y_dot_waypoints - self.y_dot_cur)
        
        elif self.mode == "hardware":
            self.vx = self.kp * (x_waypoints - self.x_cur) + self.kd * (x_dot_waypoints - self.x_dot_cur)
            self.vy = self.kp * (y_waypoints - self.y_cur) + self.kd * (y_dot_waypoints - self.y_dot_cur)

        v = np.sqrt((self.vx**2)+(self.vy**2))
        theta = np.arctan2(self.vy, self.vx)

        v = np.clip(v, 0, 0.3)
        self.vx = v * np.cos(theta)
        self.vy = v * np.sin(theta)

        self.cmd_vel.linear.x = self.vx
        self.cmd_vel.linear.y = self.vy
        

        self.vel_pub.publish(self.cmd_vel)
        rospy.loginfo("Velocities published: \n %s \n", str(self.cmd_vel))
        rospy.loginfo("--------------------------------------")

        
    

    
if __name__ == "__main__":

    # mode = "simulation"
    mode = "hardware"

    if mode == "simulation":
        # ref_traj = np.load("/home/anish/gym-pybullet-drones/reference_trajectories/ref_circle.npy")
        ref_traj = np.load("/home/anish/gym-pybullet-drones/reference_trajectories/xyz_ref_traj1.npy")
    elif mode == "hardware":
        # ref_traj = np.load("/home/anish/gym-pybullet-drones/reference_trajectories/ref_circle_0_5.npy")
        # ref_traj = np.load("/home/anish/gym-pybullet-drones/reference_trajectories/ref_circle_r1.npy")
        
        ref_traj = np.load("/home/anish/drones_ws/trajectories/evaluation 1-20231003/circle.npy")
        # ref_traj = np.load("/home/anish/drones_ws/trajectories/evaluation 1-20231003/double_circle.npy")
        # ref_traj = np.load("/home/anish/drones_ws/trajectories/evaluation 1-20231003/sine.npy")
        
        
    cur_x = 3.81003737449646
    cur_y = -0.5075851678848267

    ref_traj_x = ref_traj[0] + cur_x
    ref_traj_y = ref_traj[1] + cur_y

    actual_traj_x = []
    actual_traj_y = []

    rospy.init_node("trajectory_tracking")
    
    node = TrajectoryTracking(ref_traj_x, ref_traj_y, mode)

    rospy.spin()

    plt.figure(figsize=(10, 10))
    plt.scatter(ref_traj_x, ref_traj_y)
    plt.scatter(actual_traj_x, actual_traj_y)
    
    center_x = (np.max(ref_traj_x) + np.min(ref_traj_x))/2
    center_y = (np.max(ref_traj_y) + np.min(ref_traj_y))/2
    
    plt.scatter(center_x, center_y)

    plt.legend(['Reference Trajectory', 'Actual Trajectory'])
    plt.xlabel("X-Coordinates")
    plt.ylabel("Y-Coordinates")
    plt.grid()
    plt.show()
    




