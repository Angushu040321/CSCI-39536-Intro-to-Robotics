import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from math import atan2
import csv
import os
from geometry_msgs.msg import Twist #import Twist message type 


def yaw_from_quaternion(qx, qy, qz, qw):
    # For 2D robots, yaw (heading) can be extracted from quaternion.
    # This simplified form comes from standard yaw computation.
    # yaw = atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    # We only need z, w for planar motion shortcut:
    return atan2(2.0*(qw*qz), 1.0 - 2.0*(qz*qz))

class OdomLogger(Node):
    def __init__(self):
        super().__init__('odom_logger')
        self.sub = self.create_subscription(Odometry, '/odom', self.cb, 50)

        self.sub_cmd = self.create_subscription(Twist, '/cmd_vel', self.cb_cmd, 50) #a subscription for commanded twist messages 

        

        # CSV file in the package directory (or choose your own path)
        self.csv_path = os.path.expanduser('~/hunter-intro-to-robotics/ros2_ws/odom_cmd_log.csv')
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.writer = csv.writer(self.csv_file)
        # Header row changed to include commanded velocities 
        self.writer.writerow([
    't_sec', 'x', 'y', 'yaw',
    'lin_x', 'lin_y', 'lin_z',
    'ang_x', 'ang_y', 'ang_z',
    'cmd_lin_x', 'cmd_ang_z'
])

        self.start_time = None
        self.last_cmd = (0.0, 0.0) #last commanded(lin_x, ang_z))
        self.get_logger().info(f"Logging /odom to {self.csv_path}")


    def cb_cmd(self, msg: Twist):
        self.last_cmd = (msg.linear.x, msg.angular.z)  #callback for commanded twist messages

    def cb(self, msg: Odometry):
        # time stamp (seconds from first message)
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.start_time is None:
            self.start_time = t
        t_rel = t - self.start_time

        # pose
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        yaw = yaw_from_quaternion(q.x, q.y, q.z, q.w)

        # twist (velocities)
        lin = msg.twist.twist.linear
        ang = msg.twist.twist.angular
        
        cmd_lin, cmd_ang = self.last_cmd #commanded velocities 

        self.writer.writerow([
    f"{t_rel:.3f}", f"{x:.4f}", f"{y:.4f}", f"{yaw:.4f}",
    f"{lin.x:.4f}", f"{lin.y:.4f}", f"{lin.z:.4f}",
    f"{ang.x:.4f}", f"{ang.y:.4f}", f"{ang.z:.4f}",
    f"{cmd_lin:.4f}", f"{cmd_ang:.4f}" #added to csv 
])

    def destroy_node(self):
        try:
            self.csv_file.close()
        except Exception:
            pass
        super().destroy_node()

def main():
    rclpy.init()
    node = OdomLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

