#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Point
import numpy as np
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
import matplotlib.pyplot as plt

class GetObjectRange(Node):
    def __init__(self):
        super().__init__('get_object_range')

        # Set up QoS Profiles for passing images over WiFi
        image_qos_profile = QoSProfile(depth=5)
        image_qos_profile.history = QoSHistoryPolicy.KEEP_LAST
        image_qos_profile.durability = QoSDurabilityPolicy.VOLATILE 
        image_qos_profile.reliability = QoSReliabilityPolicy.BEST_EFFORT 

        self.obstacle_pub = self.create_publisher(Point, '/object_range', 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.get_closest_obstacle, image_qos_profile)
        self.odom_subscriber = self.create_subscription(Point, '/fixed_odom', self.odom_callback, 10)

        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_theta = 0.0

    def get_closest_obstacle(self, data):
        # Get ranges from LIDAR scan
        ranges = np.array(data.ranges)

        # Filter out invalid range readings
        ranges[np.isnan(ranges)] = 5.5
        
        # ranges = [x for x in ranges if not np.isnan(x)]

        # Remap indices to a 0-360 range
        num_ranges = len(ranges)
        remapped_indices = np.linspace(0, 360, num_ranges, endpoint=False)
        self.get_logger().info(f"remapped_indices = {remapped_indices}")

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
        closest_idxs = [i for i in range(len(remapped_indices)-1) if ranges[i] < 1 and ranges[i+1] < 1]
        obstacle_groups = group_consecutive(closest_idxs)
        self.get_logger().info(f"Obstacle groups: = {obstacle_groups}")
        
        # Extract specific groups (adjust indices as needed)
        target_groups = []
        for group in obstacle_groups:
            if (group[0] >= 0 and group[-1] <= 50) or (group[0] >= int(310/360 * len(remapped_indices)) and group[-1] <= int(360/360 * len(remapped_indices))):
                target_groups.extend(group)
        
        self.get_logger().info(f"Target group = {target_groups}")

        # angle_of_target_group = [int((i - data.angle_min) if i <= 180 else (i + (360 - data.angle_max))) for i in target_groups]
        angle_of_target_group = [int(((i + 180) % (2 * 180) - 180)) for i in target_groups] 
        self.get_logger().info(f"Angles in target group = {angle_of_target_group}")
        
        # for group in obstacle_groups:
        # # Find indices of minimum and maximum angle and distance
        # min_idx = min(group, key=lambda i: (ranges[i], i))
        # max_idx = max(group, key=lambda i: (ranges[i], i))

        min_angle = min(angle_of_target_group)
        max_angle = max(angle_of_target_group)
        self.get_logger().info(f"Max, Min Angles = ({max_angle}, {min_angle})")

        min_idx = target_groups[angle_of_target_group.index(min_angle)]
        max_idx = target_groups[angle_of_target_group.index(max_angle)]
        
        self.get_logger().info(f"Max, Min indices = ({max_idx}, {min_idx})")
        
        self.get_logger().info(f"ranges[min_idx] * np.cos(min_angle * data.angle_increment) = ({ranges[min_idx]}, {np.cos(min_angle * data.angle_increment)},{data.angle_increment})")

        # Calculate vectors to the minimum and maximum points in local frame
        min_vector = (ranges[min_idx] * np.cos(min_idx * data.angle_increment), 
                        ranges[min_idx] * np.sin(min_idx * data.angle_increment))
        max_vector = (ranges[max_idx] * np.cos(max_idx * data.angle_increment), 
                        ranges[max_idx] * np.sin(max_idx * data.angle_increment))
        self.get_logger().info(f"Local position: point1, point2 = [{min_vector},{max_vector}]")

        # Transform vectors to global frame
        rotation_matrix = np.matrix([[np.cos(self.robot_theta), -np.sin(self.robot_theta)],
                                    [np.sin(self.robot_theta),  np.cos(self.robot_theta)]])
                
        global_min_vector_x = rotation_matrix[0, 0] * min_vector[0] + rotation_matrix[0, 1] * min_vector[1] + self.robot_x
        global_min_vector_y = rotation_matrix[1, 0] * min_vector[0] + rotation_matrix[1, 1] * min_vector[1] + self.robot_y
        global_max_vector_x = rotation_matrix[0, 0] * max_vector[0] + rotation_matrix[0, 1] * max_vector[1] + self.robot_x
        global_max_vector_y = rotation_matrix[1, 0] * max_vector[0] + rotation_matrix[1, 1] * max_vector[1] + self.robot_y
        
        self.get_logger().info(f"Global position: point1, point2 = [({global_min_vector_x},{global_min_vector_y}),({global_max_vector_x},{global_max_vector_y})]")

        # Publish obstacle information (e.g., as a Point message)
        obstacle_msg = Point()
        obstacle_msg.x = global_min_vector_x
        obstacle_msg.y = global_min_vector_y
        self.obstacle_pub.publish(obstacle_msg)

        # Plot the robot and obstacle vectors
        plt.figure()
        plt.plot(self.robot_x, self.robot_y, 'bo', label='Robot')
        plt.quiver(global_min_vector_x, global_min_vector_y, 
            global_max_vector_x - global_min_vector_x, 
            global_max_vector_y - global_min_vector_y, 
            color='r')
        # plt.quiver(self.robot_x, self.robot_y, global_min_vector[0]-self.robot_x, global_min_vector[1]-self.robot_y, color='r', label='Min Vector')
        # plt.quiver(self.robot_x, self.robot_y, global_max_vector[0]-self.robot_x, global_max_vector[1]-self.robot_y, color='g', label='Max Vector')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Robot and Obstacle Vectors')
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        plt.show()

    def odom_callback(self, msg):
        self.robot_x = msg.x
        self.robot_y = msg.y
        self.robot_theta = msg.z

def main(args=None):
    rclpy.init(args=args)
    node = GetObjectRange()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()