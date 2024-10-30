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
        self.dist_thresh_obs = 0.25  # Threshold to trigger wall following
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
        
        self.has_obstacle = False
        
        # Modes
        self.robot_mode = "go to goal mode"  # Can be "go to goal mode" or "wall following mode"

    def min_obstacle_callback(self, msg):
        self.obstacle_x_min = msg.x
        self.obstacle_y_min = msg.y

    # Calculate the obstacel_tangent_line if not already done
        if not self.hit_obstale_line_calculated:
            self.calculate_obstacel_tangent_line()

    def max_obstacle_callback(self, msg):
        self.obstacle_x_max = msg.x
        self.obstacle_y_max = msg.y

    # Calculate the obstacel_tangent_line if not already done
        if not self.hit_obstale_line_calculated:
            self.calculate_obstacel_tangent_line()

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
    
    def calculate_obstacel_tangent_line(self):
        if not self.obstacle_y_max and not self.obstacle_y_min:
            return

        # Set the start point as the robot's initial position
        self.start_x = self.hit_point_x
        self.start_y = self.hit_point_y

        # Calculate the slope and intercept of the line
        if self.obstacle_x_max != self.obstacle_x_min:
            self.hit_obstale_slope = (self.obstacle_y_max - self.obstacle_y_min) / (self.obstacle_x_max - self.obstacle_x_min)
            self.hit_obstale_intercept = self.obstacle_y_min - self.start_goal_slope * self.obstacle_x_min
        else:
            self.hit_obstale_slope = float('inf')  # Vertical line

        self.start_goal_line_calculated = True

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
        self.left_dist = scan_ranges[int(90/360 * len(msg.ranges))]  # Left (90 degrees)
        self.front_dist = scan_ranges[int(0/360 * len(msg.ranges))]  # Front (0 degrees)
        self.right_dist = scan_ranges[int(270/360 * len(msg.ranges))]   # Right (270 degrees)
        self.leftfront_dist = scan_ranges[int(45/360 * len(msg.ranges))]  # Left-front diagonal (45 degrees)
        self.rightfront_dist = scan_ranges[int(315/360 * len(msg.ranges))]  # Right-front diagonal (315 degrees)
        self.rightback_dist = scan_ranges[int(225/360 * len(msg.ranges))]  # Right-back diagonal (225 degrees)
        self.leftback_dist = scan_ranges[int(135/360 * len(msg.ranges))]  # Left-back diagonal (135 degrees)
        
        self.get_logger().info(f"Distances are [{(self.left_dist,self.leftfront_dist,self.front_dist,self.rightfront_dist,self.right_dist)}]")

        # Mode switching logic
        if self.robot_mode == "go to goal mode" and self.obstacle_detected():
            # Switch to wall following mode if an obstacle is detected
            self.robot_mode = "wall following mode"
            self.get_logger().info("Obstacle detected!")
            self.has_obstacle = True

        elif self.robot_mode == "wall following mode" and not self.obstacle_detected() and self.on_start_goal_line():
            # Switch back to go to goal mode if obstacle is cleared and we are back on the start-goal line
            self.robot_mode = "go to goal mode"
            self.has_obstacle = False

        # Continue in the current mode
        if self.robot_mode == "go to goal mode":
            self.go_to_goal()
            self.get_logger().info(f"Robot is at {self.robot_mode}")

        elif self.robot_mode == "wall following mode":
            # Record the hit point  
            self.hit_point_x = self.current_x
            self.hit_point_y = self.current_y
            
            self.follow_wall()
            self.get_logger().info(f"Robot is at {self.robot_mode}")


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

        # Calculate the point on the start-goal 
        # line that is closest to the current position
        x_start_goal_line = self.current_x
        y_start_goal_line = (
            self.start_goal_slope * (
            x_start_goal_line)) + (
            self.start_goal_intercept)
                        
        # Calculate the distance between current position 
        # and the start-goal line
        distance_to_start_goal_line = math.sqrt(pow(
                    x_start_goal_line - self.current_x, 2) + pow(
                    y_start_goal_line - self.current_y, 2)) 
                            
        # If we hit the start-goal line again               
        if distance_to_start_goal_line < 0.1:
            
            # Determine if we need to leave the wall and change the mode
            # to 'go to goal'
            # Let this point be the leave point
            self.leave_point_x = self.current_x
            self.leave_point_y = self.current_y

            # Record the distance to the goal from the leave point
            self.distance_to_goal_from_leave_point = math.sqrt(
                pow(self.goal_x 
                - self.leave_point_x, 2)
                + pow(self.goal_y  
                - self.leave_point_y, 2)) 
            
            # Is the leave point closer to the goal than the hit point?
            # If yes, go to goal. 
            diff = self.distance_to_goal_from_hit_point - self.distance_to_goal_from_leave_point
            if diff > 0.25:
                        
                # Change the mode. Go to goal.
                self.robot_mode = "go to goal mode"


            # Exit this function
            return
        
        # if self.has_obstacle:
        #         new_waypoint_x = self.obstacle_x_max - safety_margin * np.sin(self.current_yaw)
        #         new_waypoint_y = self.obstacle_y_max + safety_margin * np.cos(self.current_yaw)                    
###########################################
        if self.has_obstacle:
            # Calculate the distance from the robot to the obstacle
            obstacle_distance = np.sqrt((self.obstacle_x_min - self.current_x)**2 + (self.obstacle_y_min - self.current_y)**2)

            if obstacle_distance < 0.5:  # Adjust this threshold as needed
                # Generate a new waypoint to the left or right of the obstacle
                safety_margin = 0.2  # Adjust this margin as needed

                # Choose the side based on the current robot orientation and obstacle position
                if self.current_yaw < np.pi / 2 or self.current_yaw > 3 * np.pi / 2:  # Robot facing left
                    new_waypoint_x = self.obstacle_x_min - safety_margin * np.sin(self.current_yaw)
                    new_waypoint_y = self.obstacle_y_min + safety_margin * np.cos(self.current_yaw)
                else:  # Robot facing right
                    new_waypoint_x = self.obstacle_x_min + safety_margin * np.sin(self.current_yaw)
                    new_waypoint_y = self.obstacle_y_min - safety_margin * np.cos(self.current_yaw)

                # Add the new waypoint to the waypoint list
                self.waypoints.insert(self.current_waypoint_index + 1, (new_waypoint_x, new_waypoint_y))

                # Switch to go-to-goal mode to reach the new waypoint
                self.robot_mode = "go_to_goal_mode"
                self.get_logger().info("Obstacle detected, adding new waypoint")
###############################################
        d = self.wall_following_dist
     
        if self.front_dist < d:
            self.wall_following_state = "1\ turn left"
            self.get_logger().info(f"State is {self.wall_following_state}")
            msg.angular.z = self.turning_speed
        
        elif self.right_dist < d:
            if (self.rightfront_dist < self.dist_too_close_to_wall):
                # Getting too close to the wall
                self.wall_following_state = "2\ turn left"
                # msg.linear.x = self.forward_speed
                msg.angular.z = self.turning_speed
                self.get_logger().info(f"State is {self.wall_following_state}")
      
            else:           
                # Go straight ahead
                self.wall_following_state = "1\ follow wall" 
                msg.linear.x = self.forward_speed  
                self.get_logger().info(f"State is {self.wall_following_state}")
        
        elif self.leftfront_dist > d and self.front_dist > d and self.rightfront_dist > d and self.rightback_dist > 2*d:
            self.wall_following_state = "1\ turn right"
            msg.angular.z = -self.turning_speed
            self.get_logger().info(f"State is {self.wall_following_state}")                                     
        
        else:
            pass

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