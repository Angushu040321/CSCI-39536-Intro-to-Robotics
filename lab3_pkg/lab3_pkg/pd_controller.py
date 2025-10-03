import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from math import atan2, sqrt, pi

def wrap_to_pi(a):
    # Keep angles in (-pi, pi]
    while a >  pi: a -= 2*pi
    while a <= -pi: a += 2*pi
    return a

class PDWaypoint(Node):
    def __init__(self):
        super().__init__('pd_waypoint')

        # === PARAMETERS / GAINS (FILL THESE) ===
        self.Kp_lin = self.declare_parameter('Kp_lin', 0.35).value     #proportional gain for linear velocity (how fast robot drives forward)
        self.Kp_ang = self.declare_parameter('Kp_ang', 4.0).value     #proportional gain for angular velocity (how strongly robot turns)
        self.Kd_ang = self.declare_parameter('Kd_ang', 0.05).value     #derivative gain for angular velocity (prevent overshoot/oscillation on turns)
        self.max_lin = self.declare_parameter('max_lin', 0.35).value #maxmimum velocity robot can go forward
        self.max_ang = self.declare_parameter('max_ang', 1.2).value #maximum angular velocity (how fast robot can turn)
        self.goal_tol = self.declare_parameter('goal_tol', .08).value #tolerance for reaching a waypoint (if a robot is within this distance of a waypoint, it is considered to have reached it)

        # Waypoints (a square). You can change/extend these.
        self.waypoints = [
            (0.8, 0.0),
            (0.8, 0.8),
            (0.0, 0.8),
            (0.0, 0.0),
        ]
        self.wp_idx = 0

       
        #set initial state variables
        self.x = 0.0; self.y = 0.0; self.yaw = 0.0
        self.prev_heading_err = 0.0
        self.prev_time = None

        # ROS I/O
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_cb, 50)
        self.timer = self.create_timer(0.05, self.control_step)  # 20 Hz

        self.get_logger().info("PD waypoint follower started.")

    def odom_cb(self, msg: Odometry):
        # Pose
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        # planar yaw (shortcut like Part 1)
        self.yaw = atan2(2.0*(q.w*q.z), 1.0 - 2.0*(q.z*q.z))

    def control_step(self):
        if self.wp_idx >= len(self.waypoints):
            # Arrived all goals: stop
            self.cmd_pub.publish(Twist())
            return

        # Current goal
        gx, gy = self.waypoints[self.wp_idx]

        # Position & heading errors
        dx = gx - self.x
        dy = gy - self.y
        dist_err = sqrt(dx*dx + dy*dy)

        # Desired heading to goal
        goal_heading = atan2(dy, dx)
        heading_err = wrap_to_pi(goal_heading - self.yaw)

        # Time delta for derivative
        now = self.get_clock().now().nanoseconds * 1e-9
        dt = 0.0
        if self.prev_time is not None:
            dt = max(1e-3, now - self.prev_time)  # avoid divide-by-zero
        self.prev_time = now

        # PD on heading (angular velocity)
        d_heading = (heading_err - self.prev_heading_err) / dt if dt > 0.0 else 0.0
        self.prev_heading_err = heading_err

        w = self.Kp_ang * heading_err + self.Kd_ang * d_heading
        v = self.Kp_lin * dist_err

        # Simple saturation
        v = max(min(v, self.max_lin), -self.max_lin)
        w = max(min(w, self.max_ang), -self.max_ang)

        # Slow down when turning sharply (blend)
        turn_scale = max(0.0, 1.0 - abs(heading_err)/pi)  # in [0,1]
        v *= turn_scale

        # Goal reached?
        if dist_err < self.goal_tol:
            self.wp_idx += 1
            self.get_logger().info(f"Reached waypoint {self.wp_idx}/{len(self.waypoints)}")
            # brief stop at each corner
            self.cmd_pub.publish(Twist())
            return

        # Publish
        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = w
        self.cmd_pub.publish(cmd)

def main():
    rclpy.init()
    node = PDWaypoint()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
