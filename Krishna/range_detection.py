import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

class RangeDetectionNode(Node):
    def __init__(self):
        super().__init__('range_detection_node')
        qos_profile = QoSProfile(reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,depth=10)
        self.scan_subscriber = self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_profile)
        self.obstacle_detected = False
        self.get_logger().info("Range detection node initialized.")

    def scan_callback(self, msg):
        self.obstacle_detected = any(distance < 0.5 for distance in msg.ranges if distance > 0)
        self.get_logger().info(f"Obstacle detected: {self.obstacle_detected}")

def main(args=None):
    rclpy.init(args=args)
    range_detection_node = RangeDetectionNode()
    rclpy.spin(range_detection_node)
    range_detection_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
