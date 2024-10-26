#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Point
import numpy as np

class GetObjectRange(Node):
    def __init__(self):
        super().__init__('get_object_range')
        self.obstacle_pub = self.create_publisher(Point, '/object_range', 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.get_closest_obstacle, 10)

    def get_closest_obstacle(self, data):
        # Get ranges from LIDAR scan
        ranges = np.array(data.ranges)
        
        # Filter out invalid range readings (assuming max range of LIDAR is 3.5m)
        ranges[np.isinf(ranges)] = 3.5

        # Find the closest obstacle
        closest_idx = np.argmin(ranges)
        closest_distance = ranges[closest_idx]
        angle_to_obstacle = data.angle_min + closest_idx * data.angle_increment

        # Create a vector pointing to the obstacle in the robot's frame
        obstacle_vector = Point()
        obstacle_vector.x = closest_distance * np.cos(angle_to_obstacle)
        obstacle_vector.y = closest_distance * np.sin(angle_to_obstacle)
        obstacle_vector.z = 0.0
        
        # Publish the vector to the closest obstacle
        self.obstacle_pub.publish(obstacle_vector)

def main(args=None):
    rclpy.init(args=args)
    node = GetObjectRange()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()