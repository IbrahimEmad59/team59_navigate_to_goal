#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from rclpy.qos import qos_profile_sensor_data

class LidarPlotter(Node):
    def __init__(self):
        super().__init__('lidar_plotter')
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            qos_profile=qos_profile_sensor_data
        )
        
        # Initialize Matplotlib figure
        self.fig, self.ax = plt.subplots()
        self.ax.set_title("Lidar Obstacle Plot (Distances < 2m)")
        self.ax.set_xlabel("Angle Index")
        self.ax.set_ylabel("Distance (m)")
        self.ax.set_xlim(0, 360)  # Adjust according to the total number of angle indices
        self.ax.set_ylim(0, 2)    # Only show distances up to 2 meters
        self.line, = self.ax.plot([], [], 'b.')  # Blue dots for all ranges below 2m
        self.obstacle_lines = []

    def scan_callback(self, msg):
        # Get range data and set distance threshold for filtering
        ranges = np.array(msg.ranges)
        distance_threshold = 1.0
        
        # Filter ranges for values below the distance threshold
        below_threshold_indices = np.where(ranges < distance_threshold)[0]
        below_threshold_ranges = ranges[below_threshold_indices]

        # Plot only the ranges below the threshold
        self.line.set_data(below_threshold_indices, below_threshold_ranges)

        # Clear previous obstacle lines
        for line in self.obstacle_lines:
            line.remove()
        self.obstacle_lines.clear()

        # Detect obstacles and interpolate lines for them
        start_idx = None
        for i in below_threshold_indices:
            if start_idx is None:
                start_idx = i
            elif i != start_idx + 1:
                # End of an obstacle segment; interpolate a line
                end_idx = i - 1
                self.draw_obstacle_line(ranges, start_idx, end_idx)
                start_idx = i

        # Handle the last obstacle segment if it ends at the last index
        if start_idx is not None:
            self.draw_obstacle_line(ranges, start_idx, below_threshold_indices[-1])

        # Plot the figure with updated data
        plt.draw()
        plt.pause(0.001)

    def draw_obstacle_line(self, ranges, start_idx, end_idx):
        """Interpolate and draw line for detected obstacle."""
        if start_idx == end_idx:
            return  # Skip single-point "obstacles"

        # Get angle indices and range values for the obstacle line
        x_vals = np.linspace(start_idx, end_idx, end_idx - start_idx + 1)
        y_vals = ranges[start_idx:end_idx + 1]
        
        # Plot a red line to represent the obstacle
        line = Line2D(x_vals, y_vals, color='red', linewidth=2)
        self.ax.add_line(line)
        self.obstacle_lines.append(line)


def main(args=None):
    rclpy.init(args=args)
    node = LidarPlotter()

    # To keep matplotlib window responsive, we call `plt.show(block=False)`
    plt.show(block=False)
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
