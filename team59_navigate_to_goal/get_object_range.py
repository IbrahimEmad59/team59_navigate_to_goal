#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Point
import numpy as np
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from std_msgs.msg import Bool

class GetObjectRange(Node):
    def __init__(self):
        super().__init__('get_object_range')

        # Set up QoS Profiles for passing images over WiFi
        image_qos_profile = QoSProfile(depth=5)
        image_qos_profile.history = QoSHistoryPolicy.KEEP_LAST
        image_qos_profile.durability = QoSDurabilityPolicy.VOLATILE 
        image_qos_profile.reliability = QoSReliabilityPolicy.BEST_EFFORT 

        self.obstacle_pub_min = self.create_publisher(Point, '/min_obstacle_point', 10)
        self.obstacle_pub_max = self.create_publisher(Point, '/max_obstacle_point', 10)
        self.has_obstacle_pub = self.create_publisher(Bool, '/has_obstacle', 10)

        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.get_closest_obstacle, image_qos_profile)
        self.odom_subscriber = self.create_subscription(Point, '/fixed_odom', self.odom_callback, 10)

        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_theta = 0.0
        self.has_obstacle = False
    
    def odom_callback(self, msg):
        self.robot_x = msg.x
        self.robot_y = msg.y
        self.robot_theta = msg.z

    def get_closest_obstacle(self, data):
        # Get ranges from LIDAR scan
        ranges = np.array(data.ranges)

        # Filter out invalid range readings
        ranges[np.isnan(ranges)] = 5.5

        # Remap indices to a 0-360 range
        num_ranges = len(ranges)
        remapped_indices = np.linspace(0, 360, num_ranges, endpoint=False)
       
        # Check if there are obstacles
        self.has_obstacle = len(ranges) > 0

        if not self.has_obstacle:
            has_obstacle_msg = Bool()
            has_obstacle_msg.data = False
            self.has_obstacle_pub.publish(has_obstacle_msg)
            return
        # Find consecutive groups of indices corresponding to obstacles
        def group_consecutive(lst):
            result = []
            current_group = []
            for i in range(len(lst)):
                if i == 0 or lst[i] - lst[i-1] == 1:
                    current_group.append(lst[i])
                else:
                    result.append(current_group)
                    current_group = [lst[i]]
            if current_group:
                result.append(current_group)
            return result

        # Find indices of obstacles
        closest_idxs = [i for i in range(len(remapped_indices)-1) if ranges[i] < 0.5 and ranges[i+1] < 0.5]
        obstacle_groups = group_consecutive(closest_idxs)

        # Extract specific groups (adjust indices as needed)
        target_groups = []
        for group in obstacle_groups:
            if (group[0] >= 0 and group[-1] <= 50) or (group[0] >= int(310/360 * len(remapped_indices)) and group[-1] <= int(360/360 * len(remapped_indices))):
                target_groups.extend(group)

        if target_groups:
            # ... (rest of the code to find min/max angles, indices, calculate vectors, and publish)
            # Find minimum and maximum angles and indices
            min_angle = min(remapped_indices[i] for i in target_groups)
            max_angle = max(remapped_indices[i] for i in target_groups)

            min_idx = target_groups[remapped_indices[target_groups].tolist().index(min_angle)]
            max_idx = target_groups[remapped_indices[target_groups].tolist().index(max_angle)]

            # Calculate vectors to the minimum and maximum points in local frame
            min_vector = (ranges[min_idx] * np.cos(min_angle * np.pi / 180), 
                        ranges[min_idx] * np.sin(min_angle * np.pi / 180))
            max_vector = (ranges[max_idx] * np.cos(max_angle * np.pi / 180), 
                        ranges[max_idx] * np.sin(max_angle * np.pi / 180))
            self.get_logger().info(f"Local position: point1, point2 = [{min_vector},{max_vector}]")

            # Transform vectors to global frame
            rotation_matrix = np.array([[np.cos(self.robot_theta), -np.sin(self.robot_theta)],
                                        [np.sin(self.robot_theta),  np.cos(self.robot_theta)]])

            global_min_vector = np.dot(rotation_matrix, min_vector) + np.array([self.robot_x, self.robot_y])
            global_max_vector = np.dot(rotation_matrix, max_vector) + np.array([self.robot_x, self.robot_y])
            self.get_logger().info(f"Global position: point1, point2 = [{global_min_vector},{global_max_vector}]")

            # Publish obstacle information (as Point messages)
            obstacle_msg_min = Point()
            obstacle_msg_min.x = global_min_vector[0]
            obstacle_msg_min.y = global_min_vector[1]
            self.obstacle_pub_min.publish(obstacle_msg_min)

            obstacle_msg_max = Point()
            obstacle_msg_max.x = global_max_vector[0]
            obstacle_msg_max.y = global_max_vector[1]
            self.obstacle_pub_max.publish(obstacle_msg_max)

            has_obstacle_msg = Bool()
            has_obstacle_msg.data = True
            self.has_obstacle_pub.publish(has_obstacle_msg)
        else:
            # Handle the case where no target groups are found
            has_obstacle_msg = Bool()
            has_obstacle_msg.data = False
            self.has_obstacle_pub.publish(has_obstacle_msg)
            self.get_logger().info("No target obstacles found")

def main(args=None):
    rclpy.init(args=args)
    node = GetObjectRange()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()