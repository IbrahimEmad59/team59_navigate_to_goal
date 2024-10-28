import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64
from math import atan2, sqrt, pi
import time
import numpy as np

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0.0
        self.integral = 0.0

    def update(self, error, dt):
        """Calculate the PID output given the error and time delta."""
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output

class GoToGoalNode(Node):
    def __init__(self):
        super().__init__('go_to_goal')

        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        #self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.odom_sub = self.create_subscription(
            Point,
            '/fixed_odom',
            self.odom_callback,
            10)
        
        # Initialize waypoints
        self.waypoints = [(1.5, 0.0), (1.5, 1.4), (0.0, 1.4)]
        self.current_goal_index = 0

        # Movement parameters
        self.linear_velocity_max = 0.1  # Max linear velocity (m/s)
        self.angular_velocity_max = 1.5   # Max angular velocity (rad/s)

        # PID controllers for linear and angular motion
        self.linear_pid = PIDController(kp=1.0, ki=0.0, kd=0.0)  # Tune these parameters
        self.angular_pid = PIDController(kp=3.0, ki=0.0, kd=0.5)  # Tune these parameters

        # Time tracking for dynamic dt (time delta)
        self.last_time = time.time()

        self.get_logger().info("Go To Goal Node initialized and starting to move to waypoints...")
        
    def odom_callback(self, Odom):
        self.current_x = Odom.x
        self.current_y = Odom.y
        self.current_yaw = Odom.z
        self.move_to_goal()

    def move_to_goal(self):
        """Main logic to move to the next waypoint."""
        while self.current_goal_index < len(self.waypoints):
            goal_x, goal_y = self.waypoints[self.current_goal_index]
            self.navigate_to_goal(goal_x, goal_y)
            self.get_logger().info(f"Moving to waypoint: ({self.current_goal_index})")

    def navigate_to_goal(self, goal_x, goal_y):
        """Navigate to the specific goal coordinates using PID control."""
        # Calculate the time delta dynamically
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        
        self.get_logger().info(f"Current x: ({self.current_x}) the goal is {goal_x}")


        # Calculate the distance and angle to the goal
        distance = sqrt((goal_x - self.current_x) ** 2 + (goal_y - self.current_y) ** 2)
        target_angle = atan2(goal_y - self.current_y, goal_x - self.current_x)
        angle_error = target_angle - self.current_yaw
        self.get_logger().info(f"Distance to waypoint is: ({distance})")
        self.get_logger().info(f"The angle to waypoint is: ({target_angle})")

        # Check if the errors are within tolerance
        if abs(angle_error) < 0.2:
            angular_velocity = 0.0  # No need to rotate further
        else:
            angular_velocity = self.angular_pid.update(angle_error, dt)

        if abs(distance) < 0.2:
            linear_velocity = 0.0  # No need to move forward/backward further
        else:
            linear_velocity = self.linear_pid.update(distance, dt)

        # Ensure the computed velocities are within the max limits
        linear_velocity = max(min(linear_velocity, self.linear_velocity_max), -self.linear_velocity_max)
        angular_velocity = max(min(angular_velocity, self.angular_velocity_max), -self.angular_velocity_max)

        self.get_logger().info(f"The linear velocity: {linear_velocity}")
        self.get_logger().info(f"The angular velocity: {angular_velocity}")

        # Create a twist message for motion
        twist = Twist()
        twist.linear.x = linear_velocity
        twist.angular.z = angular_velocity

        self.publisher.publish(twist)
        
        # Check if close enough to the waypoint
        if distance < 0.1:  # Adjust threshold as necessary
            self.stop()
            self.get_logger().info(f"Reached waypoint: ({goal_x}, {goal_y})")
            time.sleep(5)  # Wait at the waypoint
            self.current_goal_index += 1  # Proceed to the next waypoint after reaching the current one

    def stop(self):
        """Stop the robot."""
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.publisher.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    go_to_goal_node = GoToGoalNode()
    rclpy.spin(go_to_goal_node)
    go_to_goal_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()