#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Point
import numpy as np
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy

class GetObjectRange(Node):
    def __init__(self):
        super().__init__('get_object_range')
        
        #Set up QoS Profiles for passing images over WiFi
        image_qos_profile = QoSProfile(depth=5)
        image_qos_profile.history = QoSHistoryPolicy.KEEP_LAST
        image_qos_profile.durability = QoSDurabilityPolicy.VOLATILE 
        image_qos_profile.reliability = QoSReliabilityPolicy.BEST_EFFORT 
        
        self.obstacle_pub = self.create_publisher(Point, '/object_range', 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.get_closest_obstacle, image_qos_profile)

    def get_closest_obstacle(self, data):
        # Get ranges from LIDAR scan
        ranges = np.array(data.ranges)
        
        # Filter out invalid range readings
        ranges = [ranges.pop(i) for i in np.isnan(ranges)]
        
        self.get_logger().info(f"Original data: {ranges}")
        
        closest_idxs = [i for i in range(len(ranges)-1) if ranges[i] < 1 and ranges[i+1] < 1]
        closest_distances = [ranges[i] for i in closest_idxs]

        self.get_logger().info(f"Close indices: {closest_idxs}")
        self.get_logger().info(f"Close ranges: {closest_distances}")
        
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