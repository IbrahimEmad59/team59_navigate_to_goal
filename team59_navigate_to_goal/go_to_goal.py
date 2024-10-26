#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Point
import math
import numpy as np

# FSM States
GO_TO_GOAL = 0
AVOID_OBSTACLE = 1
FOLLOW_WALL_CLOCKWISE = 2
FOLLOW_WALL_COUNTERCLOCKWISE = 3

class GoToGoal(Node):
    def __init__(self):
        super().__init__('go_to_goal')
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.obstacle_sub = self.create_subscription(Point, '/object_range', self.object_range_callback, 10)
        
        self.current_state = GO_TO_GOAL
        self.current_position = Point()
        self.goal_position = Point()
        self.obstacle_vector = Point()

        self.Init = True
        self.Init_ang = 0.0
        self.Init_pos = Point()
        self.globalPos = Point()
        self.globalAng = 0.0

        self.waypoints = [(1.5, 0.0), (1.5, 1.4), (0.0, 1.4)]
        self.current_waypoint_idx = 0
        self.goal_position = Point()

    # Odometry callback with rotation correction
    def odom_callback(self, Odom):
        position = Odom.pose.pose.position
        q = Odom.pose.pose.orientation
        orientation = np.arctan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))

        if self.Init:
            self.Init = False
            self.Init_ang = orientation
            self.globalAng = self.Init_ang
            Mrot = np.matrix([[np.cos(self.Init_ang), np.sin(self.Init_ang)], [-np.sin(self.Init_ang), np.cos(self.Init_ang)]])        
            self.Init_pos.x = Mrot.item((0, 0)) * position.x + Mrot.item((0, 1)) * position.y
            self.Init_pos.y = Mrot.item((1, 0)) * position.x + Mrot.item((1, 1)) * position.y
            self.Init_pos.z = position.z

        Mrot = np.matrix([[np.cos(self.Init_ang), np.sin(self.Init_ang)], [-np.sin(self.Init_ang), np.cos(self.Init_ang)]])        
        self.globalPos.x = Mrot.item((0, 0)) * position.x + Mrot.item((0, 1)) * position.y - self.Init_pos.x
        self.globalPos.y = Mrot.item((1, 0)) * position.x + Mrot.item((1, 1)) * position.y - self.Init_pos.y
        self.globalAng = orientation - self.Init_ang

    def object_range_callback(self, data):
        self.obstacle_vector = data

    def waypoints_callback(self, data):
        self.waypoints.append(data)

    def go_to_goal(self):
        goal_distance = math.sqrt((self.goal_position.x - self.globalPos.x)**2 + (self.goal_position.y - self.globalPos.y)**2)
        goal_angle = math.atan2(self.goal_position.y - self.globalPos.y, self.goal_position.x - self.globalPos.x)

        if goal_distance < 0.2:
            # Decide based on proximity to goal, either CW or CCW
            self.decide_wall_follow_direction()
            return

        vel_cmd = Twist()
        vel_cmd.linear.x = 0.15
        vel_cmd.angular.z = goal_angle - self.globalAng
        self.cmd_pub.publish(vel_cmd)

    def avoid_obstacle(self):
        obstacle_distance = math.sqrt(self.obstacle_vector.x**2 + self.obstacle_vector.y**2)

        if obstacle_distance > 0.5:
            self.current_state = GO_TO_GOAL
            return

        vel_cmd = Twist()
        vel_cmd.linear.x = 0.0
        vel_cmd.angular.z = 0.5
        self.cmd_pub.publish(vel_cmd)

    def follow_wall_clockwise(self):
        # Wall-following logic: Clockwise
        if math.sqrt(self.obstacle_vector.x**2 + self.obstacle_vector.y**2) > 0.5:
            self.current_state = GO_TO_GOAL
            return

        vel_cmd = Twist()
        vel_cmd.linear.x = 0.05
        vel_cmd.angular.z = -0.2  # Turn slightly right (CW)
        self.cmd_pub.publish(vel_cmd)

    def follow_wall_counterclockwise(self):
        # Wall-following logic: Counterclockwise
        if math.sqrt(self.obstacle_vector.x**2 + self.obstacle_vector.y**2) > 0.5:
            self.current_state = GO_TO_GOAL
            return

        vel_cmd = Twist()
        vel_cmd.linear.x = 0.05
        vel_cmd.angular.z = 0.2  # Turn slightly left (CCW)
        self.cmd_pub.publish(vel_cmd)

    def decide_wall_follow_direction(self):
        # Calculate the angle to the goal
        goal_angle = math.atan2(self.goal_position.y - self.globalPos.y, self.goal_position.x - self.globalPos.x)

        # Calculate the angle to the detected obstacle
        obstacle_angle = math.atan2(self.obstacle_vector.y, self.obstacle_vector.x)

        # Calculate the angle difference (in radians)
        angle_difference = goal_angle - obstacle_angle

        # Normalize the angle to the range [-pi, pi]
        angle_difference = (angle_difference + math.pi) % (2 * math.pi) - math.pi

        # Choose clockwise if the goal is to the right of the obstacle, counterclockwise otherwise
        if angle_difference < 0:
            self.current_state = FOLLOW_WALL_CLOCKWISE
            self.get_logger().info("Switching to Clockwise Wall Follow")
        else:
            self.current_state = FOLLOW_WALL_COUNTERCLOCKWISE
            self.get_logger().info("Switching to Counterclockwise Wall Follow")

    def state_machine(self):
        if self.current_waypoint_idx >= len(self.waypoints):
            self.get_logger().info("All waypoints reached!")
            return

        self.goal_position = self.waypoints[self.current_waypoint_idx]

        if self.current_state == GO_TO_GOAL:
            self.go_to_goal()
        elif self.current_state == AVOID_OBSTACLE:
            self.avoid_obstacle()
        elif self.current_state == FOLLOW_WALL_CLOCKWISE:
            self.follow_wall_clockwise()
        elif self.current_state == FOLLOW_WALL_COUNTERCLOCKWISE:
            self.follow_wall_counterclockwise()

        if math.sqrt((self.goal_position.x - self.globalPos.x)**2 + (self.goal_position.y - self.globalPos.y)**2) < 0.2:
            self.current_waypoint_idx += 1

def main(args=None):
    rclpy.init(args=args)
    node = GoToGoal()
    rate = node.create_rate(10)  # 10 Hz
    while rclpy.ok():
        rclpy.spin_once(node)
        node.state_machine()
        rate.sleep()

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()