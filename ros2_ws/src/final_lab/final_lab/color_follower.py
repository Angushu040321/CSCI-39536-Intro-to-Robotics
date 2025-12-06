#!/usr/bin/env python3
"""
Color Follower for TurtleBot3 - Final Lab

This node subscribes to camera images from TurtleBot3, detects colored objects 
(red, green, blue), and moves the robot toward the detected object by publishing 
TwistStamped messages to cmd_vel.

"""

import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
from sensor_msgs.msg import Image
from geometry_msgs.msg import TwistStamped
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import cv2


class ColorFollower(Node):
    def __init__(self):
        super().__init__('color_follower')
        
        # Declare parameters
        self.declare_parameter('image_topic', '/camera/image_raw')  # Try /camera/image_raw, /camera/rgb/image_raw, or /rgb/image_raw
        self.declare_parameter('max_linear', 0.6)   #speed
        self.declare_parameter('max_angular', 1.8)  # turning
        self.declare_parameter('color', '')  # '' = detect any color, or 'red'/'green'/'blue'
        self.declare_parameter('min_area', 300)     # area of detection
        self.declare_parameter('check_topics', True)  # whether to check available topics on startup
        
        # Get parameter values
        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.max_linear = self.get_parameter('max_linear').get_parameter_value().double_value
        self.max_angular = self.get_parameter('max_angular').get_parameter_value().double_value
        self.target_color = self.get_parameter('color').get_parameter_value().string_value.lower()
        self.min_area = self.get_parameter('min_area').get_parameter_value().integer_value
        self.check_topics = self.get_parameter('check_topics').get_parameter_value().bool_value
        
        # If enabled, try to find available camera topics
        if self.check_topics:
            self.image_topic = self.find_camera_topic()
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Set up subscribers and publishers
        qos_profile = rclpy.qos.QoSProfile(depth=10)
        self.image_sub = self.create_subscription(
            Image, 
            self.image_topic, 
            self.image_callback, 
            qos_profile
        )
        self.cmd_pub = self.create_publisher(TwistStamped, 'cmd_vel', qos_profile)
        
        # Define HSV color ranges for detection
        # Format: (lower_hsv, upper_hsv) 
        self.color_ranges = {
            'red': [
                (np.array([0, 80, 50]), np.array([10, 255, 255])),      # Lower red range
                (np.array([170, 80, 50]), np.array([180, 255, 255]))   # Upper red range
            ],
            'green': [
                (np.array([35, 40, 40]), np.array([85, 255, 255]))      # Green range
            ],
            'blue': [
                (np.array([95, 40, 40]), np.array([135, 255, 255]))     # Blue range
            ]
        }
        
        # Control parameters
        self.angular_gain = 1.2   # turning toward object
        self.distance_gain = 0.5  # adjust speed based on object size
        
        self.get_logger().info(f'Color Follower started!')
        self.get_logger().info(f'Listening to: {self.image_topic}')
        self.get_logger().info(f'Target color: {self.target_color if self.target_color else "any"}')
        self.get_logger().info(f'Max speeds: linear={self.max_linear:.2f}, angular={self.max_angular:.2f}')
        
    def find_camera_topic(self):
        #Try to find an available camera topic
        import time
        time.sleep(0.5)  # brief wait for topics to be available
        
        try:
            topic_names = [topic_name for topic_name, _ in self.get_topic_names_and_types()]
            
            # Common camera topic patterns to try
            possible_topics = [
                '/camera/image_raw',
                '/camera/rgb/image_raw', 
                '/rgb/image_raw',
                '/image_raw',
                '/camera/color/image_raw'
            ]
            
            for topic in possible_topics:
                if topic in topic_names:
                    self.get_logger().info(f'Found camera topic: {topic}')
                    return topic
                    
            # If none found, list available image topics
            image_topics = [t for t in topic_names if 'image' in t.lower()]
            if image_topics:
                self.get_logger().warn(f'No standard camera topics found. Available image topics: {image_topics}')
                return image_topics[0]  # Use the first one found
            else:
                self.get_logger().error('No image topics found! Make sure TurtleBot3 has a camera (use waffle or waffle_pi model)')
                return self.image_topic  # Fall back to parameter value
                
        except Exception as e:
            self.get_logger().warn(f'Could not check topics: {e}')
            return self.image_topic

    def detect_color_blobs(self, hsv_image, color_name):
        #Detect blobs of specified color in HSV image
        ranges = self.color_ranges.get(color_name, [])
        combined_mask = None
        
        for lower, upper in ranges:
            mask = cv2.inRange(hsv_image, lower, upper)
            if combined_mask is None:
                combined_mask = mask
            else:
                combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        if combined_mask is None:
            return [], None
            
        # Apply morphological operations to clean up noise
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter by minimum area
        valid_contours = [c for c in contours if cv2.contourArea(c) > self.min_area]
        
        return valid_contours, combined_mask

    def get_blob_center_and_area(self, contour):
        #Calculate centroid and area of a contour
        M = cv2.moments(contour)
        if M['m00'] == 0:
            return None, 0
        
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        area = cv2.contourArea(contour)
        
        return (cx, cy), area

    def compute_control_commands(self, blob_center, blob_area, image_width, image_height):
        # Compute linear and angular velocities based on blob position and size
        if blob_center is None:
            return 0.0, 0.0
            
        cx, cy = blob_center
        image_center_x = image_width // 2
        
        # Calculate angular velocity (proportional to horizontal offset)
        error_x = cx - image_center_x
        normalized_error = error_x / float(image_center_x)  # -1 to 1
        angular_vel = -self.angular_gain * normalized_error * self.max_angular
        
        # Calculate linear velocity (slower when object is closer/larger)
        area_ratio = blob_area / float(image_width * image_height)
        # Slow down as we get closer (larger area), but maintain minimum speed
        linear_vel = self.max_linear * (1.0 - min(0.8, area_ratio * 10.0))
        linear_vel = max(0.1, linear_vel)  # Minimum forward speed
        
        return linear_vel, angular_vel

    def image_callback(self, msg):
        """Process incoming camera images"""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f'CV Bridge error: {e}')
            return
        
        # Convert to HSV for color detection
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        height, width = hsv.shape[:2]
        
        # Determine which colors to check
        colors_to_check = [self.target_color] if self.target_color else ['red', 'green', 'blue']
        
        best_blob = None
        best_area = 0
        best_color = None
        
        # Find the largest valid color blob
        for color in colors_to_check:
            contours, _ = self.detect_color_blobs(hsv, color)
            
            for contour in contours:
                center, area = self.get_blob_center_and_area(contour)
                if area > best_area:
                    best_area = area
                    best_blob = center
                    best_color = color
        
        # Compute and publish movement commands
        if best_blob is not None:
            linear_vel, angular_vel = self.compute_control_commands(
                best_blob, best_area, width, height
            )
            
            # Log detection info
            self.get_logger().debug(
                f'Found {best_color} blob at {best_blob}, '
                f'area={best_area:.0f}, cmd: lin={linear_vel:.2f}, ang={angular_vel:.2f}'
            )
        else:
            # No object detected - rotate slowly to search
            linear_vel = 0.0
            angular_vel = 0.3  # Always rotate to search when no target found
            self.get_logger().debug('No color blobs detected - searching...')
        
        # Create and publish TwistStamped message
        twist_msg = TwistStamped()
        twist_msg.header.stamp = Clock().now().to_msg()
        twist_msg.header.frame_id = ''
        twist_msg.twist.linear.x = float(linear_vel)
        twist_msg.twist.linear.y = 0.0
        twist_msg.twist.linear.z = 0.0
        twist_msg.twist.angular.x = 0.0
        twist_msg.twist.angular.y = 0.0
        twist_msg.twist.angular.z = float(angular_vel)
        
        self.cmd_pub.publish(twist_msg)


def main(args=None):
    rclpy.init(args=args)
    
    color_follower = ColorFollower()
    
    try:
        rclpy.spin(color_follower)
    except KeyboardInterrupt:
        color_follower.get_logger().info('Shutting down color follower...')
    finally:
        # Send stop command before shutdown
        stop_msg = TwistStamped()
        stop_msg.header.stamp = Clock().now().to_msg()
        color_follower.cmd_pub.publish(stop_msg)
        
        color_follower.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()