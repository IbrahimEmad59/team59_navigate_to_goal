import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point
from sensor_msgs.msg import LaserScan
from rclpy.qos import qos_profile_sensor_data
import numpy as np  # For NaN filtering

class Bug2Controller(Node):
    def __init__(self):
        super().__init__('bug_2_with_waypoints')

        # Subscribers
        self.odom_subscriber = self.create_subscription(
            Point,
            '/fixed_odom',  # Odometry data from your transformation node
            self.odom_callback,
            10)
        
        self.scan_subscriber = self.create_subscription(
            LaserScan,
            '/scan',  # Laser scan for obstacle detection
            self.scan_callback,
            qos_profile=qos_profile_sensor_data)

        # Publisher
        self.velocity_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        # Variables for obstacle avoidance
        self.left_dist = float('inf')
        self.front_dist = float('inf')
        self.right_dist = float('inf')
        self.leftfront_dist = float('inf')
        self.rightfront_dist = float('inf')
        self.dist_thresh_obs = 0.25  # Threshold to trigger wall following
        self.forward_speed = 0.1  # Speed when moving forward
        self.turning_speed = 0.3  # Speed when turning
        self.wall_following_dist = 0.45  # Distance to maintain while following the wall
        self.dist_too_close_to_wall = 0.15  # Minimum safe distance from the wall
        self.goal_threshold = 0.15  # Distance threshold to consider waypoint reached

        # Variables for odometry and goal seeking
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.goal_x = None
        self.goal_y = None

        # Start and goal line (BUG2 specific)
        self.start_x = None
        self.start_y = None
        self.start_goal_line_calculated = False
        self.start_goal_slope = None
        self.start_goal_intercept = None

        # Waypoints (goal positions)
        self.waypoints = [(1.5, 0.0), (1.5, 1.4), (0.0, 1.4)]  # Add your waypoints here
        self.current_waypoint_index = 0

        # Modes
        self.robot_mode = "go to goal mode"  # Can be "go to goal mode" or "wall following mode"

    def odom_callback(self, msg):
        """
        Callback to update the robot's current position and orientation from odometry data.
        """
        self.current_x = msg.x
        self.current_y = msg.y
        self.current_yaw = msg.z

        # Calculate the start-to-goal line if not already done
        if not self.start_goal_line_calculated:
            self.calculate_start_goal_line()

    def calculate_start_goal_line(self):
        """
        Calculate the slope and intercept of the line connecting the start position and the goal.
        This is used in the BUG2 algorithm to determine when the robot can return to the go-to-goal mode.
        """
        if not self.waypoints:
            return

        # Set the start point as the robot's initial position
        self.start_x = self.current_x
        self.start_y = self.current_y

        # Set the goal point as the first waypoint
        self.goal_x, self.goal_y = self.waypoints[self.current_waypoint_index]

        # Calculate the slope and intercept of the line
        if self.goal_x != self.start_x:
            self.start_goal_slope = (self.goal_y - self.start_y) / (self.goal_x - self.start_x)
            self.start_goal_intercept = self.start_y - self.start_goal_slope * self.start_x
        else:
            self.start_goal_slope = float('inf')  # Vertical line

        self.start_goal_line_calculated = True

    def on_start_goal_line(self):
        """
        Check if the robot is on the start-to-goal line.
        If the line is vertical (infinite slope), check if the x-coordinate matches.
        """
        if self.start_goal_slope == float('inf'):
            return abs(self.current_x - self.start_x) < 0.05  # Allow small tolerance for floating-point error
        else:
            expected_y = self.start_goal_slope * self.current_x + self.start_goal_intercept
            return abs(self.current_y - expected_y) < 0.05  # Allow small tolerance

    def scan_callback(self, msg):
        """
        Callback to process laser scan data and handle obstacle detection.
        Filters out NaN values from the laser scan data.
        """
        # Replace NaN values in laser scan ranges with a large number (e.g., infinity)
        scan_ranges = np.array(msg.ranges)
        scan_ranges = np.where(np.isnan(scan_ranges), float('inf'), scan_ranges)  # Replace NaNs with infinity

        # LDS-02 Lidar: Adjust indices based on 360-degree field of view
        self.left_dist = scan_ranges[50]  # Left (270 degrees)
        self.front_dist = scan_ranges[0]  # Front (180 degrees)
        self.right_dist = scan_ranges[180]   # Right (90 degrees)
        self.leftfront_dist = scan_ranges[90]  # Left-front diagonal (225 degrees)
        self.rightfront_dist = scan_ranges[220]  # Right-front diagonal (135 degrees)

        # Mode switching logic
        if self.robot_mode == "go to goal mode" and self.obstacle_detected():
            # Switch to wall following mode if an obstacle is detected
            self.robot_mode = "wall following mode"
        elif self.robot_mode == "wall following mode" and not self.obstacle_detected() and self.on_start_goal_line():
            # Switch back to go to goal mode if obstacle is cleared and we are back on the start-goal line
            self.robot_mode = "go to goal mode"

        # Continue in the current mode
        if self.robot_mode == "go to goal mode":
            self.go_to_goal()
        elif self.robot_mode == "wall following mode":
            self.follow_wall()

    def obstacle_detected(self):
        """
        Return True if an obstacle is detected within the threshold distance.
        """
        return (self.front_dist < self.dist_thresh_obs or 
                self.rightfront_dist < self.dist_thresh_obs or 
                self.leftfront_dist < self.dist_thresh_obs or 
                self.right_dist < self.dist_thresh_obs or 
                self.left_dist < self.dist_thresh_obs)

    def go_to_goal(self):
        """
        Drive the robot toward the current waypoint (goal).
        """
        if self.current_waypoint_index >= len(self.waypoints):
            self.get_logger().info("All waypoints reached!")
            self.stop_robot()
            return

        # Get the current goal (waypoint)
        self.goal_x, self.goal_y = self.waypoints[self.current_waypoint_index]

        # Calculate the distance and angle to the current goal
        distance_to_goal = math.sqrt((self.goal_x - self.current_x) ** 2 + (self.goal_y - self.current_y) ** 2)
        angle_to_goal = math.atan2(self.goal_y - self.current_y, self.goal_x - self.current_x)
        yaw_error = angle_to_goal - self.current_yaw

        # Normalize yaw_error to range [-pi, pi]
        yaw_error = (yaw_error + math.pi) % (2 * math.pi) - math.pi

        # Create a Twist message for velocity control
        msg = Twist()

        if distance_to_goal > self.goal_threshold:
            # If yaw error is significant, rotate to face the goal
            if abs(yaw_error) > 0.1:
                msg.angular.z = self.turning_speed if yaw_error > 0 else -self.turning_speed
            else:
                # Move straight toward the goal
                msg.linear.x = self.forward_speed
        else:
            # If the goal is reached, move to the next waypoint
            self.get_logger().info(f"Waypoint {self.current_waypoint_index} reached.")
            self.current_waypoint_index += 1
            self.start_goal_line_calculated = False  # Recalculate start-goal line for the new waypoint

        self.velocity_publisher.publish(msg)

    def follow_wall(self):
        """
        Wall-following behavior to navigate around obstacles.
        """
        msg = Twist()

        if self.front_dist > self.dist_thresh_obs and self.right_dist > self.wall_following_dist:
            # No obstacle in front, follow the wall by moving forward
            msg.linear.x = self.forward_speed
        elif self.front_dist < self.dist_thresh_obs:
            # Wall detected in front, turn left
            msg.angular.z = self.turning_speed
        elif self.right_dist < self.dist_too_close_to_wall:
            # Too close to the wall, turn left
            msg.angular.z = self.turning_speed
        else:
            # Follow the wall by moving forward
            msg.linear.x = self.forward_speed

        self.velocity_publisher.publish(msg)

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
    bug2_controller = Bug2Controller()
    rclpy.spin(bug2_controller)
    bug2_controller.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()