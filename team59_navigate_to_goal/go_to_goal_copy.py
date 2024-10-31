import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point
from sensor_msgs.msg import LaserScan
from rclpy.qos import qos_profile_sensor_data
import numpy as np  # For NaN filtering
import time
from std_msgs.msg import Bool

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
        
        self.min_obstacle_sub = self.create_subscription(
            Point,
            '/min_obstacle_point',
            self.min_obstacle_callback,
            10)

        self.max_obstacle_sub = self.create_subscription(
            Point,
            '/max_obstacle_point',
            self.max_obstacle_callback,
            10)

        # Publisher
        self.velocity_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        # Variables for obstacle avoidance
        self.left_dist = float('inf')
        self.front_dist = float('inf')
        self.right_dist = float('inf')
        self.leftfront_dist = float('inf')
        self.rightfront_dist = float('inf')
        self.dist_thresh_obs = 0.35  # Threshold to trigger wall following
        self.forward_speed = 0.1  # Speed when moving forward
        self.turning_speed = 1.25  # Speed when turning
        self.wall_following_dist = 0.25  # Distance to maintain while following the wall
        self.dist_too_close_to_wall = 0.15  # Minimum safe distance from the wall
        self.goal_threshold = 0.1  # Distance threshold to consider waypoint reached

        # Variables for odometry and goal seeking
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.goal_x = None
        self.goal_y = None

        self.waypoint_x = None
        self.waypoint_y = None

        # Start and goal line (BUG2 specific)
        self.start_x = None
        self.start_y = None
        self.start_goal_line_calculated = False
        self.start_goal_slope = None
        self.start_goal_intercept = None

        self.hit_obstale_line_calculated = False
        self.hit_obstale_slope = None
        self.hit_obstale_intercept = None
        self.obstacle_x_min = None
        self.obstacle_y_min = None
        self.obstacle_x_max = None
        self.obstacle_y_max = None

        # Waypoints (goal positions)
        self.waypoints = [(1.5, 0.0), (1.5, 1.4), (0.0, 1.4)]  
        self.current_waypoint_index = 0
        
        # Waypoints (to avoide obstacle)
        self.obstacle_waypoints = []  
        self.current_obstacle_waypoint_index = 0

        self.has_obstacle = False
        self.avoiding = False
        self.first_obstacle_encountered = False
        
        # Modes
        self.robot_mode = "go to goal mode"  # Can be "go to goal mode" or "wall following mode"

    def min_obstacle_callback(self, msg):
        self.obstacle_x_min = msg.x
        self.obstacle_y_min = msg.y

    def max_obstacle_callback(self, msg):
        self.obstacle_x_max = msg.x
        self.obstacle_y_max = msg.y

    def odom_callback(self, msg):
        """
        Callback to update the robot's current position and orientation from odometry data.
        """
        self.current_x = msg.x
        self.current_y = msg.y
        self.current_yaw = msg.z

    def scan_callback(self, msg):
        """
        Callback to process laser scan data and handle obstacle detection.
        Filters out NaN values from the laser scan data.
        """
        # Replace NaN values in laser scan ranges with a large number (e.g., infinity)
        scan_ranges = np.array(msg.ranges)
        scan_ranges = np.where(np.isnan(scan_ranges), float('inf'), scan_ranges)  # Replace NaNs with infinity

        # LDS-02 Lidar: Adjust indices based on 360-degree field of view
        self.left_dist = scan_ranges[int(90/360 * len(msg.ranges))]  # Left (90 degrees)
        self.front_dist = scan_ranges[int(0/360 * len(msg.ranges))]  # Front (0 degrees)
        self.right_dist = scan_ranges[int(270/360 * len(msg.ranges))]   # Right (270 degrees)
        self.leftfront_dist = scan_ranges[int(45/360 * len(msg.ranges))]  # Left-front diagonal (45 degrees)
        self.rightfront_dist = scan_ranges[int(315/360 * len(msg.ranges))]  # Right-front diagonal (315 degrees)
        self.rightback_dist = scan_ranges[int(225/360 * len(msg.ranges))]  # Right-back diagonal (225 degrees)
        self.leftback_dist = scan_ranges[int(135/360 * len(msg.ranges))]  # Left-back diagonal (135 degrees)
        
        # self.get_logger().info(f"Distances are [{(self.left_dist,self.leftfront_dist,self.front_dist,self.rightfront_dist,self.right_dist)}]")

        # Mode switching logic
        if self.robot_mode == "go to goal mode" and self.obstacle_detected() and not self.avoiding:
            # Switch to wall following mode if an obstacle is detected
            self.robot_mode = "wall following mode"
            self.get_logger().info("Obstacle detected!")
            self.has_obstacle = True

        elif self.robot_mode == "wall following mode" and not self.obstacle_detected():
            # Switch back to go to goal mode if obstacle is cleared and we are back on the start-goal line
            self.robot_mode = "go to goal mode"
            self.has_obstacle = False

        # Continue in the current mode
        if self.robot_mode == "go to goal mode":
            self.go_to_goal()
            # self.get_logger().info(f"Robot is at {self.robot_mode}")

        elif self.robot_mode == "wall following mode":
            # Record the hit point  
            self.hit_point_x = self.current_x
            self.hit_point_y = self.current_y
            
            self.follow_wall()
            # self.get_logger().info(f"Robot is at {self.robot_mode}")


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
        
        self.get_logger().info(f"Following waypoint {self.current_waypoint_index}")

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
            self.stop_robot()
            time.sleep(5)  # Wait at the waypoint
            self.current_waypoint_index += 1
            self.start_goal_line_calculated = False  # Recalculate start-goal line for the new waypoint
            self.avoiding = False

        self.velocity_publisher.publish(msg)

    def follow_wall(self):
        """
        Wall-following behavior to navigate around obstacles.
        """
        if self.has_obstacle:
            if not self.first_obstacle_encountered:
                # Calculate the distance from the robot to the obstacle
                # obstacle_distance_min = np.sqrt((self.obstacle_x_min - self.current_x)**2 + (self.obstacle_y_min - self.current_y)**2)
                # obstacle_distance_max = np.sqrt((self.obstacle_x_max - self.current_x)**2 + (self.obstacle_y_max - self.current_y)**2)
                # obstacle_distance = min(obstacle_distance_min,obstacle_distance_max)

                self.stop_robot()
                time.sleep(2)  # Wait at the waypoint
                
                safety_margin = 0.25  # Adjust this margin as needed

                # Choose the side based on the current robot orientation and obstacle position
                if self.current_yaw < np.pi / 2 or self.current_yaw > 3 * np.pi / 2:  # Robot facing left
                    new_waypoint_x = self.obstacle_x_min - safety_margin * np.sin(self.current_yaw)
                    new_waypoint_y = self.obstacle_y_min + safety_margin * np.cos(self.current_yaw)
                else:  # Robot facing right
                    new_waypoint_x = self.obstacle_x_min + safety_margin * np.sin(self.current_yaw)
                    new_waypoint_y = self.obstacle_y_min - safety_margin * np.cos(self.current_yaw)

                self.waypoints.insert(self.current_waypoint_index, (new_waypoint_x, new_waypoint_y))
                # self.obstacle_waypoints.append((new_waypoint_x, new_waypoint_y))
                # self.robot_mode = "go_to_goal_mode"
                self.go_to_goal()
                self.get_logger().info(f"Obstacle detected, adding new waypoint at ({new_waypoint_x},{new_waypoint_y})")
                self.first_obstacle_encountered = True
                self.avoiding = True
        
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