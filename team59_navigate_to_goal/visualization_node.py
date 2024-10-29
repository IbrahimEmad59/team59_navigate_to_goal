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
        self.ax.set_title("Lidar Obstacle Plot")
        self.ax.set_xlabel("Angle Index")
        self.ax.set_ylabel("Distance (m)")
        self.ax.set_xlim(0, 360)
        self.ax.set_ylim(0, 10)
        self.line, = self.ax.plot([], [], 'b.')  # Blue dots for all ranges
        self.obstacle_lines = []

    def scan_callback(self, msg):
        # Get range data
        ranges = np.array(msg.ranges)
        
        # Set the distance threshold for obstacles
        distance_threshold = 1.5
        
        # Plot the raw range data
        angle_indices = np.arange(len(ranges))
        self.line.set_data(angle_indices, ranges)

        # Clear previous obstacle lines
        for line in self.obstacle_lines:
            line.remove()
        self.obstacle_lines.clear()

        # Detect obstacles and interpolate lines for them
        start_idx = None
        for i in range(len(ranges)):
            if ranges[i] < distance_threshold:
                if start_idx is None:
                    start_idx = i
            elif start_idx is not None:
                # End of an obstacle segment; interpolate a line
                end_idx = i - 1
                self.draw_obstacle_line(ranges, start_idx, end_idx)
                start_idx = None

        # Plot the figure with updated data
        plt.draw()
        plt.pause(0.001)

    def draw_obstacle_line(self, ranges, start_idx, end_idx):
        """Interpolate and draw line for detected obstacle."""
        if start_idx == end_idx:
            return  # Skip single-point "obstacles"

        # Get angles and range values for the obstacle line
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
