#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import math

class ObstacleAvoidance(Node):
    def __init__(self):
        super().__init__('obstacle_avoidance')

        # Maximum velocities (linear and angular)
        self.max_linear_velocity = 0.22  # meters per second
        self.max_angular_velocity = 1.5  # radians per second

        # Threshold for detecting obstacles
        self.obstacle_distance_threshold = 0.5  # meters

        # Publisher for velocity commands
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Subscriber to LiDAR data
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        # Obstacle avoidance state
        self.obstacle_detected = False  # Track if an obstacle is currently detected
        self.rotated_90_degrees = False  # State to track if we've completed the 90-degree turn

    def scan_callback(self, scan_msg):
        """Callback to process the LiDAR data."""
        # Check if any obstacle is within the threshold distance in the front
        front_distances = scan_msg.ranges[0:30] + scan_msg.ranges[-30:]  # Front 60 degrees
        nearest_obstacle_distance = min(front_distances)

        if nearest_obstacle_distance < self.obstacle_distance_threshold:
            if not self.obstacle_detected:
                self.get_logger().info(f"Obstacle detected at {nearest_obstacle_distance} meters. Starting avoidance.")
            self.obstacle_detected = True
            self.follow_wall()
        else:
            if self.obstacle_detected:
                self.get_logger().info("Path cleared. Moving towards the goal.")
            self.obstacle_detected = False
            self.rotated_90_degrees = False  # Reset rotation state
            self.move_forward()

    def follow_wall(self):
        """Rotate 90 degrees and move forward to avoid the obstacle."""
        if not self.rotated_90_degrees:
            self.get_logger().info("Rotating 90 degrees to avoid obstacle.")
            self.rotate_90_degrees()
        else:
            self.get_logger().info("Moving forward after rotating 90 degrees.")
            self.move_forward()

    def rotate_90_degrees(self):
        """Rotate the robot by 90 degrees."""
        twist = Twist()
        twist.linear.x = 0.0  # No forward movement during rotation
        twist.angular.z = self.max_angular_velocity  # Rotate at max angular velocity

        # Publish rotation command
        self.cmd_pub.publish(twist)

        # Simulate 90 degrees turn (roughly 1/4 of a full rotation)
        rclpy.spin_once(self, timeout_sec=1.0)  # Spin for 1 second (adjust for your robot's speed)

        # Stop rotation after turning
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)

        # Mark the rotation as completed
        self.rotated_90_degrees = True
        self.get_logger().info("Completed 90-degree rotation.")

    def move_forward(self):
        """Move the robot forward."""
        twist = Twist()
        twist.linear.x = self.max_linear_velocity  # Move forward at max linear velocity
        twist.angular.z = 0.0  # No rotation

        self.cmd_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    obstacle_avoidance_node = ObstacleAvoidance()
    rclpy.spin(obstacle_avoidance_node)
    obstacle_avoidance_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()