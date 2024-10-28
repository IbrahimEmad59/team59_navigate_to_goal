#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
import math
import numpy as np


class PIDController:
    def __init__(self, kp, ki, kd, output_limits=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0
        self.output_limits = output_limits

    def compute(self, error, dt):
        """Compute the control signal based on the error and time step."""
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error

        # Apply output limits if set
        if self.output_limits:
            output = max(self.output_limits[0], min(output, self.output_limits[1]))

        return output


class GoToGoal(Node):
    def __init__(self):
        super().__init__('go_to_goal')

        # Maximum velocities (linear and angular)
        self.max_linear_velocity = 0.1  # meters per second
        self.max_angular_velocity = 1.5  # radians per second

        # Robot's current position
        self.current_position = Point()

        # Goal position variable
        self.goal_position = Point()

        # Initializing the waypoints
        self.waypoints = [(1.5, 0.0), (1.5, 1.4), (0.0, 1.4)]
        self.current_waypoint_idx = 0
        
        # Desired distance to the object
        self.desired_distance = 0.2  # meter

        # Tolerance settings
        self.distance_tolerance = 0.05 * self.desired_distance  # 5% of the desired distance
        self.angle_tolerance = 0.05  # 5% tolerance in radians

        # PID controllers for angular and linear control, with output limits
        self.angular_pid = PIDController(kp=2.2, ki=0.0, kd=0.5, output_limits=(-self.max_angular_velocity, self.max_angular_velocity))
        self.linear_pid = PIDController(kp=4.2, ki=0.0, kd=0.5, output_limits=(-self.max_linear_velocity, self.max_linear_velocity))

        # Subscriber to odometry
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        # Publisher for velocity commands
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Time tracking for PID computation
        self.prev_time = self.get_clock().now()

    def odom_callback(self, Odom):
        """Callback to process the odometry data."""
        position = Odom.pose.pose.position
        q = Odom.pose.pose.orientation
        orientation = np.arctan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))

        # The current potion of the robot
        self.get_logger().info(f"The current location is {(position.x, position.y)}")

        # Set goal position from waypoints
        self.goal_position.x, self.goal_position.y = self.waypoints[self.current_waypoint_idx]

        # Calculate goal distance and angle
        goal_distance = math.sqrt((self.goal_position.x - position.x)**2 + (self.goal_position.y - position.y)**2)
        goal_angle = np.arctan2(self.goal_position.y - position.y, self.goal_position.x - position.x)

        # Compute angular error
        angular_error = goal_angle - orientation
        angular_error = (angular_error + math.pi) % (2 * math.pi) - math.pi  # Normalize error

        # Compute linear error (desired distance to goal)
        linear_error = 1.5 - position.x

        # Get current time and compute time step
        current_time = self.get_clock().now()
        dt = (current_time - self.prev_time).nanoseconds / 1e9  # Convert nanoseconds to seconds
        self.prev_time = current_time

        # Control calculations
        if abs(angular_error) < self.angle_tolerance:
            angular_velocity = 0.0
        else:
            angular_velocity = self.angular_pid.compute(angular_error, dt)

        if abs(linear_error) < self.distance_tolerance:
            linear_velocity = 0.0
        else:
            linear_velocity = self.linear_pid.compute(linear_error, dt)
        
        self.get_logger().info(f"linear_velocity: {linear_velocity}")

        # Limit velocities
        linear_velocity = max(min(linear_velocity, self.max_linear_velocity), -self.max_linear_velocity)
        angular_velocity = max(min(angular_velocity, self.max_angular_velocity), -self.max_angular_velocity) * 0.0
        
        self.get_logger().info(f"linear_velocity: {linear_velocity}")

        # Publish velocity commands
        twist = Twist()
        twist.linear.x = linear_velocity
        twist.angular.z = angular_velocity
        self.cmd_pub.publish(twist)

        # Move to the next waypoint if within range
        if goal_distance < self.desired_distance and self.current_waypoint_idx < len(self.waypoints) - 1:
            self.get_logger().info(f"Waypoints {self.current_waypoint_idx} have reached!!")
            self.current_waypoint_idx += 1


def main(args=None):
    rclpy.init(args=args)

    # Create the node
    chase_object_node = GoToGoal()

    # Spin to keep the node running
    rclpy.spin(chase_object_node)

    # Cleanup when shutting down
    chase_object_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
