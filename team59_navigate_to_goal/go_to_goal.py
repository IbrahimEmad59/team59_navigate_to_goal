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
        self.output_limits = output_limits  # Min/max limits for output (if any)

    def compute(self, error, dt):
        """Compute the control signal based on the error and time step."""
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error

        return output

class GoToGoal(Node):
    def __init__(self):
        super().__init__('go_to_goal')

        # Maximum velocities (linear and angular)
        self.max_linear_velocity = 0.1  # meters per second
        self.max_angular_velocity = 1.5  # radians per second

        # Robot's current postion
        self.current_position = Point()

        # Goal postion variable
        self.goal_position = Point()

        # Initializing the waypoints postions
        self.waypoints = [(1.5, 0.0), (1.5, 1.4), (0.0, 1.4)]
        self.current_waypoint_idx = 0
        
        # Desired distance to the object 
        self.desired_distance = 0.2  # meter

        # 5% tolerance on distance and angle
        self.distance_tolerance = 0.05 * self.desired_distance  # 5% of the desired distance
        self.angle_tolerance = 0.05  # 5% tolerance in radians (adjust based on your application)

        # PID controllers for angular and linear control, with output limits
        self.angular_pid = PIDController(kp=2.2, ki=0.0, kd=0.5, output_limits=(-self.max_angular_velocity, self.max_angular_velocity))
        self.linear_pid = PIDController(kp=4.2, ki=0.0, kd=0.5, output_limits=(-self.max_linear_velocity, self.max_linear_velocity))

        # Subscriber to odem (distance and angle)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        # Publisher for velocity commands
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Time tracking for PID computation
        self.prev_time = self.get_clock().now()

    def odom_callback(self, Odom):
        """Callback to process the odem data."""
        position = Odom.pose.pose.position
        q = Odom.pose.pose.orientation
        orientation = np.arctan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))

        self.goal_position.x, self.goal_position.y = self.waypoints[self.current_waypoint_idx]

        # Extract the distance and angle from the message
        goal_distance = math.sqrt((self.goal_position.x - position.x)**2 + (self.goal_position.y - position.y)**2)
        goal_angle = np.arctan2(self.goal_position.x - position.x, self.goal_position.y - position.y)

        # Get current time and compute time step
        current_time = self.get_clock().now()
        dt = (current_time - self.prev_time).nanoseconds / 1e9  # Convert nanoseconds to seconds
        self.prev_time = current_time

        # Compute angular error (we want the angle to be 0, i.e., facing the object)
        angular_error = goal_angle
        self.get_logger().info(f"The angular_error: {angular_error}")

        # Compute linear error (we want the distance to be self.desired_distance)
        linear_error = self.desired_distance - goal_distance
        self.get_logger().info(f"The linear_error: {linear_error}")

        self.get_logger().info(f"The actual linear velocity: {linear_velocity}")
        self.get_logger().info(f"The actual angular velocity: {angular_velocity}")

        # Ensure the computed velocities are within the max limits
        linear_velocity = max(min(linear_velocity, self.max_linear_velocity), -self.max_linear_velocity)
        angular_velocity = max(min(angular_velocity, self.max_angular_velocity), -self.max_angular_velocity)

        self.get_logger().info(f"The linear velocity: {linear_velocity}")
        self.get_logger().info(f"The angular velocity: {angular_velocity}")

        # Create Twist message with linear and angular velocity
        twist = Twist()
        twist.linear.x = -linear_velocity  # Forward/backward velocity
        twist.angular.z = angular_velocity  # Rotational velocity

        # Publish the velocity commands
        self.cmd_pub.publish(twist)

        if math.sqrt((self.goal_position.x - position.x)**2 + (self.goal_position.y - position.y)**2) < 0.2:
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