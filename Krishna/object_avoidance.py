import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from math import atan2
import numpy as np
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

class RRTObstacleAvoidance(Node):
    def __init__(self):
        super().__init__('rrt_obstacle_avoidance_node')
        qos_profile = QoSProfile(reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,depth=10)
        self.scan_subscriber = self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_profile)
        self.position_subscriber = self.create_subscription(Odometry, '/odom', self.position_callback, 10)
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.obstacle_detected = False
        self.goal_position = None
        self.current_position = None
        self.get_logger().info("RRT Obstacle Avoidance node initialized.")

    def scan_callback(self, msg):
        self.obstacle_detected = any(distance < 0.5 for distance in msg.ranges if distance > 0)

    def position_callback(self, msg):
        self.current_position = msg.pose.pose.position
        if self.obstacle_detected and self.goal_position:
            self.avoid_obstacle()

    def avoid_obstacle(self):
        angle_to_avoid = atan2(self.goal_position.y - self.current_position.y, self.goal_position.x - self.current_position.x) + np.pi / 2
        cmd_msg = Twist()
        cmd_msg.linear.x = 0.1
        cmd_msg.angular.z = angle_to_avoid
        self.cmd_vel_publisher.publish(cmd_msg)
        self.get_logger().info(f"Avoiding obstacle with linear.x: {cmd_msg.linear.x}, angular.z: {cmd_msg.angular.z}")

def main(args=None):
    rclpy.init(args=args)
    rrt_obstacle_avoidance_node = RRTObstacleAvoidance()
    rclpy.spin(rrt_obstacle_avoidance_node)
    rrt_obstacle_avoidance_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
