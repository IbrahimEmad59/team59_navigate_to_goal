#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
from math import atan2, sqrt, pi
import time

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0

    def compute(self, error, dt):
        """Compute the control signal based on the error and time step."""
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output

class ChaseObjectWithWaypoints(Node):
    def __init__(self):
        super().__init__('chase_object_with_waypoints')

        # Maximum velocities (linear and angular)
        self.max_linear_velocity = 0.1  # meters per second
        self.max_angular_velocity = 1.5  # radians per second

        # Initialize waypoints
        self.waypoints = [(1.5, 0.0), (1.5, 1.4), (0.0, 1.4)]
        self.current_goal_index = 0

        # Tolerance for reaching the waypoint
        self.distance_tolerance = 0.1  # meters

        # PID controllers for angular and linear control
        self.angular_pid = PIDController(kp=2.2, ki=0.0, kd=0.5)
        self.linear_pid = PIDController(kp=4.2, ki=0.0, kd=0.5)

        # Subscribers
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        # Publisher for velocity commands
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Current position and orientation
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0

        # Time tracking for PID computation
        self.prev_time = self.get_clock().now()

        self.get_logger().info("Chase Object with Waypoints Node initialized.")
        self.move_to_goal()

    def odom_callback(self, msg):
        """Update current position from odometry data."""
        self.get_logger().info("Odem data reseaveed.")
        
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        # Extract yaw from quaternion (assuming 2D plane, only z-axis rotation)
        orientation_q = msg.pose.pose.orientation
        yaw = atan2(2.0 * (orientation_q.w * orientation_q.z + orientation_q.x * orientation_q.y),
                    1.0 - 2.0 * (orientation_q.y * orientation_q.y + orientation_q.z * orientation_q.z))
        self.current_yaw = yaw
        self.get_logger().info(f"Current position: ({self.current_x}, {self.current_y}), Yaw: {self.current_yaw}")

    def move_to_goal(self):
        """Main logic to move to the next waypoint."""
        while self.current_goal_index < len(self.waypoints):
            goal_x, goal_y = self.waypoints[self.current_goal_index]
            self.get_logger().info(f"Moving to waypoint: ({goal_x}, {goal_y})")
            self.navigate_to_goal(goal_x, goal_y)

            # Proceed to the next waypoint after reaching the current one
            self.current_goal_index += 1

    def navigate_to_goal(self, goal_x, goal_y):
        """Navigate to the specific goal coordinates using PID control."""
        #while True:
        # Calculate the time delta dynamically
        current_time = self.get_clock().now()
        dt = (current_time - self.prev_time).nanoseconds / 1e9  # Convert nanoseconds to seconds
        self.prev_time = current_time

        # Calculate the distance and angle to the goal
        distance = sqrt((goal_x - self.current_x) ** 2 + (goal_y - self.current_y) ** 2)
        target_angle = atan2(goal_y - self.current_y, goal_x - self.current_x)
        angle_error = target_angle - self.current_yaw
        angle_error = (angle_error + pi) % (2 * pi) - pi  # Normalize angle error to [-pi, pi]

        #self.get_logger().info(f"Distance to goal: {distance}, Angle error: {angle_error}")

        # Calculate PID outputs
        angular_velocity = self.angular_pid.compute(angle_error, dt)
        linear_velocity = self.linear_pid.compute(distance, dt)

        # Ensure the computed velocities are within the max limits
        linear_velocity = max(min(linear_velocity, self.max_linear_velocity), -self.max_linear_velocity)
        angular_velocity = max(min(angular_velocity, self.max_angular_velocity), -self.max_angular_velocity)

        #self.get_logger().info(f"The linear velocity: {linear_velocity}")
        #self.get_logger().info(f"The angular velocity: {angular_velocity}")

        # Create Twist message with linear and angular velocity
        twist = Twist()
        twist.linear.x = linear_velocity
        twist.angular.z = angular_velocity

        self.cmd_pub.publish(twist)

        # Check if close enough to the waypoint
        if distance < self.distance_tolerance:
            self.stop()
            self.get_logger().info(f"Reached waypoint: ({goal_x}, {goal_y})")
            time.sleep(2)  # Wait at the waypoint
            #break

    def stop(self):
        """Stop the robot."""
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)
        self.get_logger().info("Robot stopped.")

def main(args=None):
    rclpy.init(args=args)
    chase_object_with_waypoints_node = ChaseObjectWithWaypoints()
    rclpy.spin(chase_object_with_waypoints_node)
    chase_object_with_waypoints_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()