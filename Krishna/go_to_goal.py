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
        self.create_subscription(Float64, '/yaw', self.yaw_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        # Initialize waypoints
        self.waypoints = [(1.5, 0.0), (1.5, 1.4), (0.0, 1.4)]
        self.current_goal_index = 0

        # Movement parameters
        self.linear_velocity_max = 0.1  # Max linear velocity (m/s)
        self.angular_velocity_max = 2.4   # Max angular velocity (rad/s)
        self.globalPos.yaw = 0.0
        self.globalPos.x = 0.0
        self.globalPos.y = 0.0

        # PID controllers for linear and angular motion
        self.linear_pid = PIDController(kp=1.0, ki=0.0, kd=0.0)  # Tune these parameters
        self.angular_pid = PIDController(kp=3.0, ki=0.0, kd=0.5)  # Tune these parameters

        # Time tracking for dynamic dt (time delta)
        self.last_time = time.time()

        self.get_logger().info("Go To Goal Node initialized and starting to move to waypoints...")
        self.move_to_goal()

    def yaw_callback(self, yaw_msg):
        """Update the current yaw from the TF node."""
        self.globalPos.yaw = yaw_msg.data
        self.get_logger().info(f"The current yaw: ({self.globalPos.yaw})")


    def odom_callback(self, odom_msg):
        position = odom_msg.pose.pose.position
        q = odom_msg.pose.pose.orientation
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
        self.get_logger().info(f"Current position: ({self.globalPos.x}, {self.globalPos.y}), Heading: {self.globalAng}")    

    def move_to_goal(self):
        """Main logic to move to the next waypoint."""
        while self.current_goal_index < len(self.waypoints):
            goal_x, goal_y = self.waypoints[self.current_goal_index]
            self.navigate_to_goal(goal_x, goal_y)
            self.get_logger().info(f"Moving to waypoint: ({self.current_goal_index})")

            # Proceed to the next waypoint after reaching the current one
            self.current_goal_index += 1

    def navigate_to_goal(self, goal_x, goal_y):
        """Navigate to the specific goal coordinates using PID control."""
        while True:
            # Calculate the time delta dynamically
            current_time = time.time()
            dt = current_time - self.last_time
            self.last_time = current_time

            # Calculate the distance and angle to the goal
            distance = sqrt((goal_x - self.globalPos.x) ** 2 + (goal_y - self.globalPos.y) ** 2)
            target_angle = atan2(goal_y - self.globalPos.y, goal_x - self.globalPos.x)
            angle_error = target_angle - self.globalPos.yaw
            self.get_logger().info(f"Distance to waypoint is: ({distance})")
            self.get_logger().info(f"The angle to waypoint is: ({target_angle})")

            # Convert angle from [-pi, pi] to [0, 2pi]
            # if angle_error < 0:
            #    angle_error += 2 * np.pi
            
            # Calculate PID outputs
            angular_velocity = self.angular_pid.update(angle_error, dt)
            linear_velocity = self.linear_pid.update(distance, dt)

            # Dynamically adjust linear velocity as the robot approaches the goal
            dynamic_linear_velocity = max(0.05, min(self.linear_velocity_max, linear_velocity))

            # Create a twist message for motion
            twist = Twist()
            twist.linear.x = dynamic_linear_velocity if abs(angle_error) < 0.2 else 0.0  # Only move forward if facing the goal
            twist.angular.z = max(-self.angular_velocity_max, min(angular_velocity, self.angular_velocity_max))

            self.publisher.publish(twist)

            # Check if close enough to the waypoint
            if distance < 0.1:  # Adjust threshold as necessary
                self.stop()
                self.get_logger().info(f"Reached waypoint: ({goal_x}, {goal_y})")
                time.sleep(2)  # Wait at the waypoint
                break

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