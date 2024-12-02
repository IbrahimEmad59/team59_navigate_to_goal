#!/usr/bin/env python3
import rclpy
import cv2
import time
import tf2_ros
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped, Point
from sensor_msgs.msg import Image, LaserScan, CompressedImage
from cv_bridge import CvBridge
import pickle
import math
from collections import Counter
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy


class TurtleBotMazeNavigator(Node):
    def __init__(self):
        super().__init__('turtlebot_maze_navigator')

        #Set up QoS Profiles for passing images over WiFi
        image_qos_profile = QoSProfile(depth=5)
        image_qos_profile.history = QoSHistoryPolicy.KEEP_LAST
        image_qos_profile.durability = QoSDurabilityPolicy.VOLATILE 
        image_qos_profile.reliability = QoSReliabilityPolicy.BEST_EFFORT 

        # Publishers and Subscribers
        self.nav_goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        # self.image_sub = self.create_subscription(Image, '/simulated_camera/image_raw', self.image_callback, 10)
        self.image_sub = self.create_subscription(CompressedImage,"/image_raw/compressed", self.image_callback, image_qos_profile)

        self.laser_sub = self.create_subscription(LaserScan, '/scan', self.laser_callback, image_qos_profile)
        self.odom_subscriber = self.create_subscription(Point, '/fixed_odom', self.odom_callback, 10)

        # Initialize tf2 components
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # # Map Grid
        # self.maze_grid = [
        #     [(-2.0, 1.0), (-1.0, 1.0), (0.0, 1.0), (1.0, 1.0), (2.0, 1.0), (3.0, 1.0)],
        #     [(-2.0, 0.0), (-1.0, 0.0), (0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)],
        #     [(-2.0, -1.0), (-1.0, -1.0), (0.0, -1.0), (1.0, -1.0), (2.0, -1.0), (3.0, -1.0)]
        # ]


        # Map Grid
        self.maze_grid = [
            [(0.0, 1.84), (0.92, 1.84), (2.0, 2.0), (3.0, 2.0), (4.0, 2.0), (5.0, 2.0)],
            [(0.0, 0.92), (0.92, 0.92), (2.0, 1.0), (3.0, 1.0), (4.0, 1.0), (5.0, 1.0)],
            [(0.0, 0.0), (0.92, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0), (5.0, 0.0)]
        ]

        # Load the trained model
        model_path = '/home/ibrahim/ros2_ws/src/team59_navigate_to_goal/Krishna/sign_classifier_model.pkl'
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

        # Movement parameters
        self.speed = 0.2
        self.turn_speed = 0.2
        self.wall_distance_threshold = 0.5 # Wall detection threshold in meters

        # Sign classification state
        self.current_command = None
        self.classification_results = []
        self.classification_threshold = 5
        self.sign_detected = False

        # LIDAR state
        self.is_close_to_wall = False

        # Odometry state
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0

        # State for turning
        self.target_yaw = None

        self.point_selected = False

    def image_callback(self, msg):
        """Process camera feed and classify signs with majority voting."""
        # cv_image = CvBridge().imgmsg_to_cv2(msg, desired_encoding='bgr8')
        cv_image = CvBridge().compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')

        
        processed_image = self.preprocess_image(cv_image)
        prediction = self.classify_sign(processed_image)

        if prediction is not None:
            self.classification_results.append(prediction)
        
        # self.get_logger().info(f"Prediction is: {prediction}")        
        # self.get_logger().info(f"Classification number: {len(self.classification_results)}")
        
        if len(self.classification_results) >= self.classification_threshold:
            self.current_command = self.majority_vote(self.classification_results)
            self.classification_results = []
            self.sign_detected = True
            # self.get_logger().info(f"Sign classified as: {self.current_command}")

    def laser_callback(self, scan_msg):
        """Detect wall proximity using LIDAR data."""
        self.ranges = np.array(scan_msg.ranges)
        self.ranges = np.where(np.isnan(self.ranges), 0.0, self.ranges)  # Filter out invalid readings (e.g., 0.0)
        self.ranges = np.where(np.isinf(self.ranges), 0.0, self.ranges)  # Filter out invalid readings (e.g., 0.0)

        right_range = scan_msg.ranges[len(scan_msg.ranges)//16]
        left_range = scan_msg.ranges[15 * len(scan_msg.ranges)//16]
        diff = right_range - left_range
        # self.get_logger().info(f"Wall detected at {max(self.ranges)}")
        
        if (right_range < self.wall_distance_threshold or left_range < self.wall_distance_threshold) and diff < self.wall_distance_threshold:
            # self.get_logger().info("Close to a wall. Ready to classify.")
            if self.current_command is not None and self.point_selected is not True:
                # Determine the waypoint based on the sign and current position
                waypoint,target_yew = self.select_waypoint(self.ranges)
                local_goal = self.find_closest_grid_point(waypoint)
                self.get_logger().info(f"Global waypoint: {waypoint}")
                self.get_logger().info(f"Global goal: {local_goal}")

                if local_goal:
                    self.publish_goal(local_goal,target_yew)
                    self.is_close_to_wall = False
        else:
            # self.get_logger().info("Far from walls. Moving forward.")
            self.is_close_to_wall = False
    
    def odom_callback(self, msg):
        """Update the robot's current position and orientation from odometry data."""
        self.current_x = msg.x
        self.current_y = msg.y
        self.current_yaw = msg.z

    def euler_from_quaternion(self, quat):
        """Convert quaternion to Euler angles (roll, pitch, yaw)."""
        siny_cosp = 2 * (quat.w * quat.z + quat.x * quat.y)
        cosy_cosp = 1 - 2 * (quat.y * quat.y + quat.z * quat.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        if yaw < 0:
            yaw += 2 * math.pi
        return 0.0, 0.0, yaw

    def preprocess_image(self, image):
        """Preprocess the image for model input."""
        image = crop_image_to_sign(image)  # Use the cropping function from the original script
        if image is None:
            # self.get_logger().info("Image is None")
            return None
        features = extract_features(image)  # Use feature extraction from original script
        # self.get_logger().info(f"Image feature is {features}")
        return features.reshape(1, -1)

    def classify_sign(self, image):
        """Classify sign using the trained model."""
        # self.get_logger().info("Classifing image")
        if image is not None:
            # self.get_logger().info(f"Image classified as {self.model.predict(image)[0]}")
            return self.model.predict(image)[0]
        return None

    def majority_vote(self, classifications):
        """Take majority vote from classifications."""
        counter = Counter(classifications)
        return counter.most_common(1)[0][0]

    def select_waypoint(self, distances):
        """Select a waypoint in the global frame based on the current command and LIDAR data."""
        if self.current_command == 1:  # Turn left
            relative_direction = 270  # Relative left
            world_direction = 90
            target_yaw = (self.current_yaw + math.pi / 2) % (2 * math.pi) * 0
        elif self.current_command == 2:  # Turn right
            relative_direction = 90  # Relative right
            world_direction = 270
            target_yaw = (self.current_yaw + 3*math.pi / 2) % (2 * math.pi) * 0
        elif self.current_command == 3:  # Turn around
            relative_direction = 180  # Relative backward
            world_direction = 180
            target_yaw = (self.current_yaw + math.pi) % (2 * math.pi) * 0
        elif self.current_command == 4:  # Stop
            world_direction = 180
            relative_direction = 180  # Relative backward
            target_yaw = (self.current_yaw + math.pi) % (2 * math.pi) * 0
        elif self.current_command == 5: # Goal Reached
            target_yaw = self.current_yaw
            self.stop_robot() 
        else:
            relative_direction = 0
            world_direction = 0
            target_yaw = self.current_yaw

        # Extract a range of LIDAR readings corresponding to the intended direction
        lidar_sector_start = int(0.85 * world_direction / 360 * len(distances))
        lidar_sector_end = int(1.15 * world_direction / 360 * len(distances))
        lidar_sector = distances[lidar_sector_start:lidar_sector_end]

        # Use the maximum distance from LIDAR readings for waypoint determination
        distance = max(lidar_sector)

        # Transform direction (degrees) into radians for trigonometric calculations
        relative_direction_rad = math.radians(relative_direction)
        local_waypoint = (
            math.cos(relative_direction_rad) * distance,
            -math.sin(relative_direction_rad) * distance
        )
        self.get_logger().info(f"Local waypoint: {local_waypoint}")

        # Transform local waypoint into the global frame
        current_pose = {'x': self.current_x, 'y': self.current_y, 'yaw': self.current_yaw}
        global_waypoint = self.transform_to_global(local_waypoint, current_pose)

        return global_waypoint,target_yaw

    def transform_to_global(self, local_point, pose):
        """
        Transform a local point (x, y) into the global frame using the robot's pose.
        :param local_point: Tuple (x, y) in the local frame
        :param pose: Current robot pose containing position and orientation (quaternion)
        :return: Global frame point (x, y)
        """
        x_local, y_local = local_point
        x_global = pose['x']
        y_global = pose['y']
        theta = pose['yaw']  # Orientation in radians

        # Transform to global coordinates
        x_global += x_local * math.cos(theta) - y_local * math.sin(theta)
        y_global += x_local * math.sin(theta) + y_local * math.cos(theta)

        return (x_global, y_global)
    
    def find_closest_grid_point(self, waypoint):
        """Find the closest grid vertex to the waypoint."""
        min_distance = float('inf')
        closest_point = None

        for row in self.maze_grid:
            for point in row:
                distance = math.sqrt((waypoint[0] - point[0])**2 + (waypoint[1] - point[1])**2)
                if distance < min_distance:
                    min_distance = distance
                    closest_point = point
        self.point_selected = True
        return closest_point

    def publish_goal(self, goal, yaw):
        """Publish the goal to Nav2 with a specified orientation."""
        goal_msg = PoseStamped()
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.header.frame_id = "map"  # Ensure this matches your Nav2 frame
        goal_msg.pose.position.x = goal[0]
        goal_msg.pose.position.y = goal[1]
        quaternion = self.quaternion_from_euler(0, 0, yaw)
        goal_msg.pose.orientation.x = quaternion[0]
        goal_msg.pose.orientation.y = quaternion[1]
        goal_msg.pose.orientation.z = quaternion[2]
        goal_msg.pose.orientation.w = quaternion[3]
        self.nav_goal_pub.publish(goal_msg)
        self.get_logger().info(f"Published goal: {goal} with orientation {yaw}")

    def quaternion_from_euler(self, roll, pitch, yaw):
        """Convert Euler angles to a quaternion."""
        qx = math.sin(roll / 2) * math.cos(pitch / 2) * math.cos(yaw / 2) - math.cos(roll / 2) * math.sin(pitch / 2) * math.sin(yaw / 2)
        qy = math.cos(roll / 2) * math.sin(pitch / 2) * math.cos(yaw / 2) + math.sin(roll / 2) * math.cos(pitch / 2) * math.sin(yaw / 2)
        qz = math.cos(roll / 2) * math.cos(pitch / 2) * math.sin(yaw / 2) - math.sin(roll / 2) * math.sin(pitch / 2) * math.cos(yaw / 2)
        qw = math.cos(roll / 2) * math.cos(pitch / 2) * math.cos(yaw / 2) + math.sin(roll / 2) * math.sin(pitch / 2) * math.sin(yaw / 2)
        return [qx, qy, qz, qw]
    
    def stop_robot(self):
        """Stop the robot."""
        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = 0.0
        self.cmd_vel_pub.publish(msg)
    
def crop_image_to_sign(img, debug=False):
    """Improved cropping function to focus on the sign with better masking and contour selection."""
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define color ranges in HSV for blue, green, and red
    # color_ranges = {
    #     "blue": ((100, 10, 10), (160, 230, 230)),
    #     "green": ((40, 100, 50), (90, 230, 230)),
    #     "red": ((0, 175, 175), (20, 230, 230)),
    #     "red2": ((155, 100, 100), (180, 230, 230))
    # }
    # Simulation colors
    color_ranges = {
        "blue": ((100, 10, 10), (160, 230, 230)),
        "green": ((40, 100, 50), (120, 230, 255)),
        "red": ((0, 175, 138), (20, 230, 230)),
        "red2": ((155, 100, 100), (180, 230, 230))
    }

    # Create masks for each color and combine them
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for color, (lower, upper) in color_ranges.items():
        lower_bound = np.array(lower, dtype=np.uint8)
        upper_bound = np.array(upper, dtype=np.uint8)
        current_mask = cv2.inRange(hsv, lower_bound, upper_bound)
        mask |= current_mask

        # Debug: Display individual color masks
        if debug:
            cv2.imshow(f"Mask for {color}", current_mask)

    # Apply morphological operations (close and open) to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Close small holes inside the objects
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # Remove noise around the objects

    # Apply Gaussian blur to smooth the mask
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Debug: Display cleaned mask
    if debug:
        cv2.imshow("Current Frame", img)
        cv2.imshow("Cleaned Mask with Morphological Operations and Blurring", mask)
        cv2.waitKey(0)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the contour closest to the center of the image
        img_center = (img.shape[1] // 2, img.shape[0] // 2)  # (x_center, y_center)
        closest_contour = None
        min_distance = float('inf')

        for contour in contours:
            # Calculate the center of the contour
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                distance = math.sqrt((cX - img_center[0])**2 + (cY - img_center[1])**2)
                if distance < min_distance:
                    min_distance = distance
                    closest_contour = contour

        # If a valid contour is found, crop the image around it
        if closest_contour is not None and cv2.contourArea(closest_contour) >= 500:  # Area threshold
            x, y, w, h = cv2.boundingRect(closest_contour)
            cropped_img = img[y:y+h, x:x+w]

            # Debugging: Display cropped image
            if debug:
                cv2.imshow("Cropped Image", cropped_img)
                cv2.waitKey(0)

            # Resize to uniform dimensions (e.g., 64x64 for training)
            processed_image = cv2.resize(cropped_img, (64, 64), interpolation=cv2.INTER_AREA)

            return processed_image

    return None  # Return None if no valid contour is found

def convert_to_cmyk(image):
    """Convert an image from BGR to simulated CMYK color space."""
    bgr = image.astype(np.float32) / 255.0  # Normalize BGR values to [0, 1]
    k = 1 - np.max(bgr, axis=2)  # Key (Black channel)
    c = (1 - bgr[..., 2] - k) / (1 - k + 1e-8)  # Cyan channel
    m = (1 - bgr[..., 1] - k) / (1 - k + 1e-8)  # Magenta channel
    y = (1 - bgr[..., 0] - k) / (1 - k + 1e-8)  # Yellow channel
    
    cmyk = np.stack((c, m, y, k), axis=2)
    return cmyk.astype(np.uint8)

def extract_HOG(image):
    winSize = (64, 64) 
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 9
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    return hog.compute(image).flatten()

def extract_features(image, debug=False):
    """Extract HOG features, CMYK histograms, LBP features, and directional features."""
    # Validate input image
    if image is None or image.size == 0:
        raise ValueError("Invalid input image: Image is None or empty")
    
    # Resize the image to a standard size (if not already resized)
    resized_img = cv2.resize(image, (64, 64))

    # Debugging: display resized image 
    if debug:
        cv2.imshow("Resized Image", resized_img)
        cv2.waitKey(0)  # Wait for a key press to display each image

    # Convert to CMYK and grayscale for HOG
    cmyk_img = convert_to_cmyk(resized_img)
    gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

    # Convert to CMYK and calculate histograms for each channel
    hist_c = extract_HOG(cv2.resize(cmyk_img[..., 0], (64, 64)))
    hist_m = extract_HOG(cv2.resize(cmyk_img[..., 1], (64, 64)))
    hist_y = extract_HOG(cv2.resize(cmyk_img[..., 2], (64, 64)))
    hist_k = extract_HOG(cv2.resize(cmyk_img[..., 3], (64, 64)))

    gray_features = extract_HOG(cv2.resize(gray_img, (64, 64)))

    # Concatenate all features: HOG, CMYK histograms, and gray feature
    features = np.concatenate((gray_features, hist_c, hist_m, hist_y, hist_k)).astype(np.float32)
    return features


def main(args=None):
    rclpy.init(args=args)
    node = TurtleBotMazeNavigator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()