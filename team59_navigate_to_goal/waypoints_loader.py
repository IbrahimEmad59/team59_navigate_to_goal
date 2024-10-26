#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point

class WaypointsLoader(Node):
    def __init__(self):
        super().__init__('waypoints_loader')
        self.waypoints_pub = self.create_publisher(Point, '/waypoints', 10)
        self.load_and_publish_waypoints()

    def load_and_publish_waypoints(self):
        waypoints = self.load_waypoints()

        for wp in waypoints:
            point_msg = Point()
            point_msg.x = wp[0]
            point_msg.y = wp[1]
            point_msg.z = 0.0  # Assuming 2D navigation
            self.waypoints_pub.publish(point_msg)

    def load_waypoints(self):
        waypoints = []
        with open('wayPoints.txt', 'r') as f:
            for line in f:
                x, y = map(float, line.strip().split())
                waypoints.append((x, y))
        return waypoints

def main(args=None):
    rclpy.init(args=args)
    node = WaypointsLoader()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()