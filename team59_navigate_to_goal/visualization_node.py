#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import matplotlib.pyplot as plt
import numpy as np
from math import atan2, pi, cos, sin

class VisualizationNode(Node):
    def __init__(self):
        super().__init__('visualization_node')

        # Subscriber to odometry to get the robot's position and orientation
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        # Subscriber to laser scan to get the obstacle information
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        # Set the goal waypoint (you can adjust this or subscribe to a goal topic)
        self.goal_x = 2.0  # Example goal x-coordinate
        self.goal_y = 2.0  # Example goal y-coordinate

        # Robot's current position and yaw
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0

        # Laser scan data (default to no obstacles)
        self.obstacle_distance = float('inf')
        self.obstacle_angle = 0.0

        # Initialize the plot
        self.fig, self.ax = plt.subplots()
        self.timer = self.create_timer(0.1, self.update_plot)  # Update plot every 100ms

    def odom_callback(self, msg):
        """Callback to update robot position and yaw from odometry."""
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y

        # Extract yaw from quaternion
        orientation_q = msg.pose.pose.orientation
        yaw = atan2(2.0 * (orientation_q.w * orientation_q.z + orientation_q.x * orientation_q.y),
                    1.0 - 2.0 * (orientation_q.y * orientation_q.y + orientation_q.z * orientation_q.z))
        self.robot_yaw = yaw

        self.get_logger().info(f"Robot position: ({self.robot_x}, {self.robot_y}), Yaw: {self.robot_yaw}")

    def scan_callback(self, msg):
        """Callback to process the laser scan data and find the nearest obstacle."""
        ranges = np.array(msg.ranges)
        # Filter out invalid ranges (e.g., Inf)
        valid_ranges = ranges[np.isfinite(ranges)]
        if len(valid_ranges) > 0:
            self.obstacle_distance = min(valid_ranges)
            # Get the index of the closest obstacle and its corresponding angle
            angle_index = np.argmin(valid_ranges)
            self.obstacle_angle = msg.angle_min + angle_index * msg.angle_increment
            self.get_logger().info(f"Nearest obstacle: Distance = {self.obstacle_distance}, Angle = {self.obstacle_angle}")
        else:
            self.obstacle_distance = float('inf')
            self.obstacle_angle = 0.0

    def update_plot(self):
        """Update the plot with the current goal and obstacle vectors."""
        self.ax.clear()

        # Set plot limits and labels
        self.ax.set_xlim([-3, 3])
        self.ax.set_ylim([-3, 3])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_title('Goal and Obstacle Vectors')

        # Plot the robot's position as a point
        self.ax.plot(self.robot_x, self.robot_y, 'bo', label='Robot Position')

        # Calculate the goal vector
        goal_vector_x = self.goal_x - self.robot_x
        goal_vector_y = self.goal_y - self.robot_y

        # Normalize and scale the goal vector for visualization
        goal_magnitude = np.sqrt(goal_vector_x**2 + goal_vector_y**2)
        if goal_magnitude > 0:
            goal_vector_x /= goal_magnitude
            goal_vector_y /= goal_magnitude

        # Plot the goal vector (blue)
        self.ax.arrow(self.robot_x, self.robot_y, goal_vector_x, goal_vector_y, head_width=0.1, head_length=0.2, fc='blue', ec='blue', label='Goal Vector')

        # Calculate and plot the obstacle vector (red)
        if self.obstacle_distance < float('inf'):
            # Calculate the obstacle's position relative to the robot
            obstacle_x = self.robot_x + self.obstacle_distance * cos(self.robot_yaw + self.obstacle_angle)
            obstacle_y = self.robot_y + self.obstacle_distance * sin(self.robot_yaw + self.obstacle_angle)

            # Plot the obstacle vector (red)
            obstacle_vector_x = obstacle_x - self.robot_x
            obstacle_vector_y = obstacle_y - self.robot_y
            obstacle_magnitude = np.sqrt(obstacle_vector_x**2 + obstacle_vector_y**2)
            if obstacle_magnitude > 0:
                obstacle_vector_x /= obstacle_magnitude
                obstacle_vector_y /= obstacle_magnitude
            self.ax.arrow(self.robot_x, self.robot_y, obstacle_vector_x, obstacle_vector_y, head_width=0.1, head_length=0.2, fc='red', ec='red', label='Obstacle Vector')

        # Add a legend
        self.ax.legend()

        # Draw the updated plot
        plt.draw()
        plt.pause(0.01)  # Redraw the plot in real-time

def main(args=None):
    rclpy.init(args=args)
    visualization_node = VisualizationNode()
    plt.ion()  # Enable interactive mode for real-time plotting
    rclpy.spin(visualization_node)
    visualization_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()