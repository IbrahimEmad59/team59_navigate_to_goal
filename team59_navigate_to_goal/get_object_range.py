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
        ranges = [x for x in ranges if not np.isnan(x)]

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
        closest_idxs = [i for i in range(len(ranges)-1) if ranges[i] < 1 and ranges[i+1] < 1]
        obstacle_groups = group_consecutive(closest_idxs)

        for group in obstacle_groups:
            # Find indices of minimum and maximum angle and distance
            min_idx = min(group, key=lambda i: (ranges[i], i))
            max_idx = max(group, key=lambda i: (ranges[i], i))

            # Calculate vectors to the minimum and maximum points in local frame
            min_vector = (ranges[min_idx] * np.cos(min_idx * data.angle_increment), 
                          ranges[min_idx] * np.sin(min_idx * data.angle_increment))
            max_vector = (ranges[max_idx] * np.cos(max_idx * data.angle_increment), 
                          ranges[max_idx] * np.sin(max_idx * data.angle_increment))

            # Transform vectors to global frame
            rotation_matrix = np.array([[np.cos(self.robot_theta), -np.sin(self.robot_theta)],
                                        [np.sin(self.robot_theta),  np.cos(self.robot_theta)]])
            global_min_vector = np.dot(rotation_matrix, min_vector) + np.array([self.robot_x, self.robot_y])
            global_max_vector = np.dot(rotation_matrix, max_vector) + np.array([self.robot_x, self.robot_y])

            # Publish obstacle information (e.g., as a Point message)
            obstacle_msg = Point()
            obstacle_msg.x = global_min_vector[0]
            obstacle_msg.y = global_min_vector[1]
            self.obstacle_pub.publish(obstacle_msg)

            # Plot the robot and obstacle vectors
            plt.figure()
            plt.plot(self.robot_x, self.robot_y, 'bo', label='Robot')
            plt.quiver(self.robot_x, self.robot_y, global_min_vector[0]-self.robot_x, global_min_vector[1]-self.robot_y, color='r', label='Min Vector')
            plt.quiver(self.robot_x, self.robot_y, global_max_vector[0]-self.robot_x, global_max_vector[1]-self.robot_y, color='g', label='Max Vector')
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