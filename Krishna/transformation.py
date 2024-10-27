import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64  # Import Float64 for yaw publishing
from math import atan2
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

class TFNode(Node):
    def __init__(self):
        super().__init__('tf')
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=rclpy.qos.QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.subscriber = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            qos_profile
        )
        
        # Use Float64 message type for yaw
        self.publisher = self.create_publisher(Float64, '/yaw', 10)
        self.get_logger().info("TF Node initialized and waiting for odometry data...")

    def odom_callback(self, odom_msg):
        """Callback to handle incoming odometry data and convert quaternion to yaw."""
        orientation_q = odom_msg.pose.pose.orientation
        yaw = self.quaternion_to_yaw(orientation_q)

        # Publish the yaw value
        yaw_msg = Float64()
        yaw_msg.data = yaw
        self.publisher.publish(yaw_msg)
        self.get_logger().info(f"Published Yaw: {yaw}")

    def quaternion_to_yaw(self, orientation_q):
        """Convert quaternion to yaw angle."""
        x = orientation_q.x
        y = orientation_q.y
        z = orientation_q.z
        w = orientation_q.w

        # Calculate yaw (z-axis rotation)
        sin_yaw = 2.0 * (w * z + x * y)
        cos_yaw = 1.0 - 2.0 * (y * y + z * z)
        yaw = atan2(sin_yaw, cos_yaw)
        
        return yaw

def main(args=None):
    rclpy.init(args=args)
    tf_node = TFNode()
    rclpy.spin(tf_node)
    tf_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
