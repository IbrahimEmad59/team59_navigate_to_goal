#!/usr/bin/env python3
import rclpy
import cv2
import time
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped, Point
from sensor_msgs.msg import Image, LaserScan, CompressedImage
from cv_bridge import CvBridge
import pickle
import math
from collections import Counter
## Test ##
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
##
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
        ## Test ##
        self.nav_goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        ##  
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        # self.image_sub = self.create_subscription(Image, '/simulated_camera/image_raw', self.image_callback, 10)
        self.image_sub = self.create_subscription(CompressedImage,"/image_raw/compressed", self.image_callback, image_qos_profile)

        self.laser_sub = self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)
        self.odom_subscriber = self.create_subscription(Point, '/fixed_odom', self.odom_callback, 10)

        # Timer for periodic control updates
        self.timer = self.create_timer(0.1, self.control_loop)  # 10 Hz timer

        # Load the trained model
        model_path = '/home/ibrahim/ros2_ws/src/team59_navigate_to_goal/Krishna/sign_classifier_model.pkl'
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

        # Movement parameters
        self.speed = 0.2
        self.turn_speed = 0.2
        self.wall_distance_threshold = 0.5  # Wall detection threshold in meters
        self.yaw_tolerance = 0.05  # Tolerance for yaw error in radians

        # Sign classification state
        self.current_command = None
        self.classification_results = []
        self.classification_threshold = 5
        self.sign_detected = False

        self.turn_completed = False

        # LIDAR state
        self.is_close_to_wall = False

        # Odometry state
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0

        # State for turning
        self.target_yaw = None

    def image_callback(self, msg):
        """Process camera feed and classify signs with majority voting."""
        # self.get_logger().info("Image recieved")

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

    def odom_callback(self, msg):
        """Update the robot's current position and orientation from odometry data."""
        self.current_x = msg.x
        self.current_y = msg.y
        self.current_yaw = msg.z
        self.current_yaw = (self.current_yaw + 2 * math.pi) % (2 * math.pi)

        # if self.current_yaw < 0:
        #     self.current_yaw += 2*math.pi

    def laser_callback(self, scan_msg):
        """Detect wall proximity using LIDAR data."""
        self.ranges = np.array(scan_msg.ranges)
        self.ranges = np.where(np.isnan(self.ranges), 0.0, self.ranges)  # Filter out invalid readings (e.g., 0.0)

        right_range = scan_msg.ranges[len(scan_msg.ranges)//16]
        left_range = scan_msg.ranges[15 * len(scan_msg.ranges)//16]
        diff = right_range - left_range
        # self.get_logger().info(f"Wall detected at {max(self.ranges)}")
        if (right_range < self.wall_distance_threshold or left_range < self.wall_distance_threshold) and diff < self.wall_distance_threshold:
            # self.get_logger().info("Close to a wall. Ready to classify.")
            self.is_close_to_wall = True
        else:
            # self.get_logger().info("Far from walls. Moving forward.")
            self.is_close_to_wall = False

    def preprocess_image(self, image):
        """Preprocess the image for model input."""
        
        # self.get_logger().info("Processing image")

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

    def execute_command(self):
        """Execute the current command based on the last classified sign."""
        
        # self.get_logger().info("Excute command started.")
        # self.get_logger().info(f"Majority vote result: {self.current_command}")

        if self.current_command is None:
            self.sign_detected = False  # Clear the sign-detection flag
            self.current_command = None  # Reset the current command after execution
            return

        if self.current_command == 0:  # Empty wall: Do nothing
            self.get_logger().info("Empty wall detected. Doing nothing.")
            self.stop_robot()
            self.move_backward()
        elif self.current_command == 1:  # Turn left 90 degrees
            self.get_logger().info("Turn left 90 degrees.")
            self.perform_turn(math.radians(90))
            # self.move_forward()
        elif self.current_command == 2:  # Turn right 90 degrees
            self.get_logger().info("Turn right 90 degrees.")
            self.perform_turn(math.radians(270))
            # self.move_forward()
        elif self.current_command == 3:  # Turn around 180 degrees
            self.get_logger().info("Turn around 180 degrees.")
            self.perform_turn(math.radians(180))
            # self.move_forward()
        elif self.current_command == 4:  # Stop (turn around and stop)
            self.get_logger().info("Stop.")
            self.perform_turn(math.radians(180))
            # self.move_forward()
        elif self.current_command == 5:  # Goal reached
            self.get_logger().info("Goal reached. Stopping robot.")
            self.stop_robot()
        else:
            self.get_logger().info("Nothing there")
            self.move_forward()

    def move_forward(self):
        """Move the robot forward."""
        msg = Twist()
        msg.linear.x = self.speed
        msg.angular.z = 0.0
        self.cmd_vel_pub.publish(msg)
        # self.get_logger().info("Moving forward: Publishing Twist message.")

    def move_backward(self):
            """Move the robot forward."""
            msg = Twist()
            msg.linear.x = -self.speed/2
            msg.angular.z = 0.0
            self.cmd_vel_pub.publish(msg)
            # self.get_logger().info("Moving forward: Publishing Twist message.")

    def perform_turn(self, target_angle):
        """Turn the robot to a specific target angle (relative to the current yaw)."""
        # Update the target yaw
        if target_angle != 0:
            self.target_yaw = (self.current_yaw + target_angle) % (2 * math.pi)  # Normalize to [0, 2π)
        
        self.get_logger().info(f"Current yaw: {self.current_yaw:.2f}, Target angle: {math.degrees(target_angle):.2f}, Target yaw: {self.target_yaw:.2f}.")
        
        # Calculate yaw error
        yaw_error = (self.target_yaw - self.current_yaw + math.pi) % (2 * math.pi) - math.pi  # Normalize to [-π, π)

        if abs(yaw_error) > self.yaw_tolerance:
            msg = Twist()
            msg.angular.z = self.turn_speed if yaw_error > 0 else -self.turn_speed
            self.cmd_vel_pub.publish(msg)
            self.get_logger().info(f"Turning to target yaw: {math.degrees(self.target_yaw):.2f} degrees, Yaw error: {math.degrees(yaw_error):.2f}.")
        else:
            self.stop_robot()
            self.target_yaw = None  # Clear target yaw after turn is done
            self.turn_completed = True
            self.get_logger().info("Turn completed.")

    def stop_robot(self):
        """Stop the robot."""
        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = 0.0
        self.cmd_vel_pub.publish(msg)
 
    def control_loop(self):
        """Main control logic executed by a timer."""
        self.get_logger().info(f"Control loop started. turn_completed: {self.turn_completed}, "
                                f"target_yaw: {self.target_yaw}, is_close_to_wall: {self.is_close_to_wall}, "
                                f"sign_detected: {self.sign_detected}")

        if self.current_command == 5:
            self.stop_robot()
            self.get_logger().info("Goal reached. Stopping robot.")
        elif self.target_yaw is not None:  # A turn is in progress
            yaw_error = (self.target_yaw - self.current_yaw + math.pi) % (2 * math.pi) - math.pi
            if abs(yaw_error) > self.yaw_tolerance:
                self.perform_turn(0)  # Continue turning
            else:
                self.get_logger().info("Turn completed. Proceeding to the next action.")
                self.stop_robot()
                self.target_yaw = None  # Clear target yaw
                self.turn_completed = True
        elif self.turn_completed:  # Move forward after turn completion
            self.move_forward()
            self.turn_completed = False  # Reset turn status
        elif self.is_close_to_wall:
            self.get_logger().info("Close to wall detected.")
            if not self.sign_detected:
                # Stop and classify a new sign when a wall is detected
                self.stop_robot()
                self.get_logger().info("Wall detected. Preparing for sign classification...")
                self.sign_detected = True
            elif not self.turn_completed:
                self.get_logger().info("Executing command after detecting a sign.")
                self.execute_command()
        else:
            self.get_logger().info("No conditions met. Moving forward.")
            self.move_forward()


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
        "green": ((40, 100, 50), (120, 210, 255)),
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
    # node.control_loop()  # Start the navigation process
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()