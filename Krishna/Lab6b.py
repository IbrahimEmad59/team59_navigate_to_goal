#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import Image, LaserScan
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient
from cv_bridge import CvBridge
import cv2
import pickle
import time
import math
from collections import Counter

class TurtleBotMazeNavigator(Node):
    def __init__(self):
        super().__init__('turtlebot_maze_navigator')

        # Publishers and Subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.image_sub = self.create_subscription(Image, '/simulated_camera/image_raw', self.image_callback, 10)
        self.laser_sub = self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)
        
        # Subscribers
        self.odom_subscriber = self.create_subscription(
            PoseStamped,
            '/fixed_odom',  # Odometry data from your transformation node
            self.odom_callback,
            10)
        
        # Action Client for Navigation
        self.nav_to_pose_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')

        # Load the trained model
        model_path = '/home/ibrahim/ros2_ws/src/team59_navigate_to_goal/Krishna/sign_classifier_model.pkl'
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

        # CV Bridge for converting ROS image messages to OpenCV images
        self.bridge = CvBridge()

        # Movement and navigation parameters
        self.speed = 0.2
        self.turn_speed = 0.5
        self.wall_distance_threshold = 0.5  # Wall detection threshold in meters

        # Sign classification state
        self.current_command = None
        self.classification_threshold = 5
        self.classification_results = []
        self.sign_detected = False

        # LIDAR state
        self.latest_scan = None
        self.is_close_to_wall = False

        # Waypoint generation
        self.waypoints = []
        self.current_waypoint_index = 0

        # Define a predefined maze grid with potential waypoints
        # This is a simplistic representation and should be customized to your specific maze
        self.maze_grid = [
            [(-3.0, 1.5) ,(-1.5, 1.5), (0.0, 1.5), (1.5, 1.5), (3.0, 1.5), (4.5, 1.5)],
            [(-3.0, 0.0), (-1.5, 0.0), (0.0, 0.0), (1.5, 0.0), (3.0, 0.0), (4.5, 0.0)],
            [(-3.0, -1.5), (-1.5, -1.5), (0.0, -1.5), (1.5, -1.5), (3.0, -1.5), (4.5, -1.5)]
        ]

    def odom_callback(self, msg):
        """
        Callback to update the robot's current position and orientation from odometry data.
        """
        self.current_x = msg.pose.position.x
        self.current_y = msg.pose.position.y
        # Orientation is a quaternion, let's convert to Euler angles (yaw)
        orientation_q = msg.pose.orientation
        _, _, self.current_yaw = self.euler_from_quaternion(orientation_q)

    def euler_from_quaternion(self, quat):
        """
        Convert quaternion to Euler angles (roll, pitch, yaw).
        """
        # Quaternion to Euler conversion (roll, pitch, yaw)
        siny_cosp = 2 * (quat.w * quat.z + quat.x * quat.y)
        cosy_cosp = 1 - 2 * (quat.y * quat.y + quat.z * quat.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return 0.0, 0.0, yaw  # We assume roll and pitch are 0 in a 2D plane

    def image_callback(self, msg):
        """Process camera feed and classify signs with majority voting."""
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        processed_image = self.preprocess_image(cv_image)
        prediction = self.classify_sign(processed_image)

        self.classification_results.append(prediction)

        if len(self.classification_results) >= self.classification_threshold:
            self.current_command = self.majority_vote(self.classification_results)
            self.classification_results = []
            self.sign_detected = True
            self.get_logger().info(f"Sign classified as: {self.current_command}")
            self.generate_waypoints()

    def laser_callback(self, scan_msg):
        """Detect wall proximity using LIDAR data."""
        # Focus on front-facing range: -pi/8 to pi/8
        center_range = scan_msg.ranges[len(scan_msg.ranges)//2 - len(scan_msg.ranges)//16 : 
                                       len(scan_msg.ranges)//2 + len(scan_msg.ranges)//16]
        
        if min(center_range) < self.wall_distance_threshold:
            self.is_close_to_wall = True
            self.get_logger().info("Wall detected! Preparing for sign classification.")
        else:
            self.is_close_to_wall = False

    def preprocess_image(self, image):
        """Preprocess image for model input."""
        image = cv2.resize(image, (64, 64))
        image = self.model.crop_image_to_sign(image)  # Use the cropping function from the original script
        if image is None:
            return None
        features = self.model.extract_features(image)  # Use feature extraction from original script
        return features.reshape(1, -1)

    def classify_sign(self, image):
        """Classify sign using the trained model."""
        if image is not None:
            return self.model.predict(image)[0]
        return None

    def majority_vote(self, classifications):
        """Take majority vote from classifications."""
        counter = Counter(classifications)
        return counter.most_common(1)[0][0]

    def generate_waypoints(self):
        """Generate waypoints based on detected sign."""
        # Simplistic waypoint generation logic
        # You'll need to customize this based on your specific maze layout
        if self.current_command == 0:  # Empty wall
            self.waypoints.append(self.get_next_waypoint('empty'))
        elif self.current_command == 1:  # Left turn
            self.waypoints.append(self.get_next_waypoint('left'))
        elif self.current_command == 2:  # Right turn
            self.waypoints.append(self.get_next_waypoint('right'))
        elif self.current_command == 3:  # Do not enter
            self.waypoints.append(self.get_next_waypoint('Do not enter'))
        elif self.current_command == 4:  # Stop
            self.waypoints.append(self.get_next_waypoint('stop'))
        elif self.current_command == 5:  # Goal
            self.waypoints.append(self.get_next_waypoint('goal'))

    def get_next_waypoint(self, direction):
        """Retrieve next waypoint based on current position and sign."""
        # Placeholder implementation. Customize for your specific maze
        current_x, current_y = self.maze_grid[self.current_waypoint_index]
        if self.current_waypoint_index == 0: 
            if self.current_yaw == 0:
                if direction == 'left':
                    return (-3.0, 1.5, -math.pi/2)
                elif direction == 'right':
                    return (1.5, 1.5, math.pi/2)
                elif direction == 'Do not enter':
                    return (-3.0, -1.5, math.pi)
                elif direction == 'stop':
                    return (-3.0, -1.5, math.pi)
                elif direction == 'goal':
                    self.stop_robot()
                    return 
            elif self.current_yaw == -math.pi:
                if direction == 'left':
                    return (-3.0, -1.5, math.pi)
                elif direction == 'right':
                    return (-3.0, 1.5, math.pi/2)
                elif direction == 'Do not enter':
                    return (1.5, 1.5, math.pi/2)
                elif direction == 'stop':
                    return (1.5, 1.5, math.pi/2)
                elif direction == 'goal':
                    self.stop_robot()
                    return 
        elif self.current_waypoint_index == 1:
            if self.current_yaw == 0:
                if direction == 'left':
                    return (-3.0, 1.5, -math.pi/2)
                elif direction == 'right':
                    return (1.5, 1.5, math.pi/2)
                elif direction == 'Do not enter':
                    return (-1.5, 0.0, math.pi)
                elif direction == 'stop':
                    return (-1.5, 0.0, math.pi)
                elif direction == 'goal':
                    self.stop_robot()
                    return 
        elif self.current_waypoint_index == 2:
            if self.current_yaw == 0:
                if direction == 'left':
                    return (-3.0, 1.5, -math.pi/2)
                elif direction == 'right':
                    return (1.5, 1.5, math.pi/2)
                elif direction == 'Do not enter':
                    return (0.0, -1.5, math.pi)
                elif direction == 'stop':
                    return (0.0, -1.5, math.pi)
                elif direction == 'goal':
                    self.stop_robot()
                    return 
        elif self.current_waypoint_index == 3:
            if self.current_yaw == 0:
                if direction == 'left':
                    return (-3.0, 1.5, -math.pi/2)
                elif direction == 'right':
                    return (1.5, 1.5, math.pi/2)
                elif direction == 'Do not enter':
                    return (0.0, 1.5, -math.pi)
                elif direction == 'stop':
                    return (0.0, 1.5, -math.pi)
                elif direction == 'goal':
                    self.stop_robot()
                    return 
            elif self.current_yaw == math.pi/2:
                if direction == 'left':
                    return (1.5, 1.5, 0.0)
                elif direction == 'right':
                    return (0.0, 1.5, -math.pi)
                elif direction == 'Do not enter':
                    return (-3.0, 1.5, -math.pi/2)
                elif direction == 'stop':
                    return (-3.0, 1.5, -math.pi/2)
                elif direction == 'goal':
                    self.stop_robot()
                    return 
        elif self.current_waypoint_index == 4:
            if self.current_yaw == 0:
                if direction == 'left':
                    return (3.0, 1.5, -math.pi/2)
                elif direction == 'right':
                    return (3.5, 1.5, math.pi/2)
                elif direction == 'Do not enter':
                    return (3.0, 1.5, -math.pi)
                elif direction == 'stop':
                    return (3.0, 1.5, -math.pi)
                elif direction == 'goal':
                    self.stop_robot()
                    return 
            elif self.current_yaw == -math.pi/2:
                if direction == 'left':
                    return (3.0, 1.5, math.pi)
                elif direction == 'right':
                    return (3.0, 1.5, 0.0)
                elif direction == 'Do not enter':
                    return (3.5, 1.5, math.pi/2)
                elif direction == 'stop':
                    return (3.5, 1.5, math.pi/2)
                elif direction == 'goal':
                    self.stop_robot()
                    return 
            elif self.current_yaw == -math.pi:
                if direction == 'left':
                    return (3.5, 1.5, math.pi/2)
                elif direction == 'right':
                    return (3.0, 1.5, -math.pi/2)
                elif direction == 'Do not enter':
                    return (3.0, 1.5, 0.0)
                elif direction == 'stop':
                    return (3.0, 1.5, 0.0)
                elif direction == 'goal':
                    self.stop_robot()
                    return 
        elif self.current_waypoint_index == 5:
            if self.current_yaw == 0:
                if direction == 'left':
                    return (3.0, 1.5, -math.pi/2)
                elif direction == 'right':
                    return (3.5, 1.5, math.pi/2)
                elif direction == 'Do not enter':
                    return (3.5, -1.5, -math.pi)
                elif direction == 'stop':
                    return (3.5, -1.5, -math.pi)
                elif direction == 'goal':
                    self.stop_robot()
                    return 
            elif self.current_yaw == math.pi/2:
                if direction == 'left':
                    return (3.5, 1.5, 0.0)
                elif direction == 'right':
                    return (3.5, -1.5, -math.pi)
                elif direction == 'Do not enter':
                    return (3.0, 1.5, -math.pi/2)
                elif direction == 'stop':
                    return (3.0, 1.5, -math.pi/2)
                elif direction == 'goal':
                    self.stop_robot()
                    return
        elif self.current_waypoint_index == 6:
            if self.current_yaw == math.pi/2:
                if direction == 'left':
                    return (-3.0, 1.5, 0.0)
                elif direction == 'right':
                    return (-3.0, -1.5, -math.pi)
                elif direction == 'Do not enter':
                    return (-3.0, 0.0, -math.pi/2)
                elif direction == 'stop':
                    return (-3.0, 0.0, -math.pi/2)
                elif direction == 'goal':
                    self.stop_robot()
                    return 
            elif self.current_yaw == -math.pi/2:
                if direction == 'left':
                    return (-3.0, -1.5, -math.pi)
                elif direction == 'right':
                    return (-3.0, 1.5, 0.0)
                elif direction == 'Do not enter':
                    return (-3.0, 0.0, math.pi/2)
                elif direction == 'stop':
                    return (-3.0, 0.0, math.pi/2)
                elif direction == 'goal':
                    self.stop_robot()
                    return
        elif self.current_waypoint_index == 7:
            if self.current_yaw == math.pi/2:
                if direction == 'left':
                    return (-1.5, 1.5, 0.0)
                elif direction == 'right':
                    return (-1.5, 0.0, -math.pi)
                elif direction == 'Do not enter':
                    return (-1.5, 0.0, -math.pi/2)
                elif direction == 'stop':
                    return (-1.5, 0.0, -math.pi/2)
                elif direction == 'goal':
                    self.stop_robot()
                    return 
            elif self.current_yaw == -math.pi/2:
                if direction == 'left':
                    return (-1.5, 0.0, -math.pi)
                elif direction == 'right':
                    return (-1.5, 1.5, 0.0)
                elif direction == 'Do not enter':
                    return (-1.5, 0.0, math.pi/2)
                elif direction == 'stop':
                    return (-1.5, 0.0, math.pi/2)
                elif direction == 'goal':
                    self.stop_robot()
                    return 
            elif self.current_yaw == -math.pi:
                if direction == 'left':
                    return (-1.5, 0.0, math.pi/2)
                elif direction == 'right':
                    return (-1.5, 0.0, -math.pi/2)
                elif direction == 'Do not enter':
                    return (-1.5, 1.5, 0.0)
                elif direction == 'stop':
                    return (-1.5, 1.5, 0.0)
                elif direction == 'goal':
                    self.stop_robot()
                    return 
        elif self.current_waypoint_index == 8:
            if self.current_yaw == math.pi/2:
                if direction == 'left':
                    return (0.0, 1.5, 0.0)
                elif direction == 'right':
                    return (0.0, -1.5, -math.pi)
                elif direction == 'Do not enter':
                    return (0.0, 0.0, -math.pi/2)
                elif direction == 'stop':
                    return (0.0, 0.0, -math.pi/2)
                elif direction == 'goal':
                    self.stop_robot()
                    return 
            elif self.current_yaw == -math.pi/2:
                if direction == 'left':
                    return (0.0, -1.5, -math.pi)
                elif direction == 'right':
                    return (-0.0, 1.5, 0.0)
                elif direction == 'Do not enter':
                    return (0.0, 0.0, math.pi/2)
                elif direction == 'stop':
                    return (0.0, 0.0, math.pi/2)
                elif direction == 'goal':
                    self.stop_robot()
                    return 
            

    def navigate_to_waypoint(self):
        """Navigate to the next generated waypoint."""
        if self.waypoints and self.current_waypoint_index < len(self.waypoints):
            goal_pose = PoseStamped()
            goal_pose.header.frame_id = 'map'
            goal_pose.pose.position.x = self.waypoints[self.current_waypoint_index][0]
            goal_pose.pose.position.y = self.waypoints[self.current_waypoint_index][1]
            goal_pose.pose.orientation = self.waypoints[self.current_waypoint_index][2]
            
            goal_msg = NavigateToPose.Goal()
            goal_msg.pose = goal_pose

            self.nav_to_pose_client.wait_for_server()
            self.nav_to_pose_client.send_goal_async(goal_msg)

    def run(self):
        """Main navigation loop."""
        rate = self.create_rate(10)  # 10 Hz
        while rclpy.ok():
            if self.is_close_to_wall:
                # If close to wall, prioritize sign classification and waypoint generation
                self.navigate_to_waypoint()
            rate.sleep()

    def stop_robot(self):
        """
        Stop the robot by publishing zero velocities.
        """
        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = 0.0
        self.velocity_publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = TurtleBotMazeNavigator()
    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().info("Node stopped by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()