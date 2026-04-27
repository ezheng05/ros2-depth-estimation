"""
cbf_node.py - ROS 2 CBF safety filter node

Implements Zhang et al. 2020 haptic teleoperation via CBFs.

Pipeline:
    /cmd_vel_ref + /depth/closest → CBF-QP → /cmd_vel + /haptic/force

Topics:
    Subscribes:
        /cmd_vel_ref       (Twist)         - user's desired command
        /depth/closest     (PointStamped)  - closest obstacle from depth_node
    Publishes:
        /cmd_vel           (Twist)         - CBF-filtered safe command to LIMO
        /haptic/force      (WrenchStamped) - force feedback for haptic device
        /cbf/debug         (String)        - debug info (b value, modification)

Testing (with bag files):
    Terminal 1: ros2 bag play calibration_bag --loop
    Terminal 2: ros2 run depth_estimation_pkg depth_node
    Terminal 3: ros2 run depth_estimation_pkg cbf_node
    Terminal 4: ros2 topic pub /cmd_vel_ref geometry_msgs/Twist
                  '{linear: {x: 0.2}, angular: {z: 0.0}}'
    Monitor:    ros2 topic echo /haptic/force
                ros2 topic echo /cbf/debug
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from geometry_msgs.msg import Twist, WrenchStamped, PointStamped
from std_msgs.msg import String
import numpy as np

from depth_estimation_pkg.cbf import CBFController


class CBFNode(Node):
    """
    CBF safety filter for haptic teleoperation.

    Reads user velocity commands and obstacle info,
    applies CBF constraint, publishes safe command
    and haptic force feedback.
    """

    def __init__(self):
        super().__init__('cbf_node')

        # --------------------------------------------------------
        # CBF Controller (all math lives in cbf.py)
        # --------------------------------------------------------
        self.cbf = CBFController(
            r_safe=0.4,       # safety radius (m) - tune based on LIMO size
            gamma=1.5,        # class-K gain - higher = more aggressive braking
            kf=10.0,          # haptic force gain - tune for device comfort
            w_omega=0.5,      # QP weight on angular velocity
            v_max=0.3,        # max forward velocity (m/s)
            v_min=-0.2,       # max reverse velocity (m/s)
            omega_max=1.0,    # max angular velocity (rad/s)
            f_max=2.5         # max haptic force (N) - below Touch's 3.3N limit
        )

        # --------------------------------------------------------
        # State
        # --------------------------------------------------------
        # Latest user command (from joystick/haptic/keyboard)
        self.v_ref = 0.0
        self.omega_ref = 0.0

        # Latest obstacle info (from depth_node)
        self.obstacle_depth = 5.0  # default = far away (safe)
        self.obstacle_px = 320     # default = center

        # Has valid obstacle data arrived yet?
        self.has_obstacle_data = False

        # --------------------------------------------------------
        # QoS - use BEST_EFFORT for sensor topics, RELIABLE for commands
        # --------------------------------------------------------
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # --------------------------------------------------------
        # Subscribers
        # --------------------------------------------------------
        # User's desired velocity command (from joystick/haptic device)
        self.cmd_sub = self.create_subscription(
            Twist,
            '/cmd_vel_ref',   # user input topic (NOT /cmd_vel - that goes to robot)
            self.on_cmd_vel_ref,
            10
        )

        # Closest obstacle from depth_node
        self.obstacle_sub = self.create_subscription(
            PointStamped,
            '/depth/closest',
            self.on_obstacle,
            sensor_qos
        )

        # --------------------------------------------------------
        # Publishers
        # --------------------------------------------------------
        # Safe velocity command to LIMO
        self.cmd_pub = self.create_publisher(
            Twist, '/cmd_vel', 10
        )

        # Haptic force feedback to device
        self.force_pub = self.create_publisher(
            WrenchStamped, '/haptic/force', 10
        )

        # Debug info
        self.debug_pub = self.create_publisher(
            String, '/cbf/debug', 10
        )

        # --------------------------------------------------------
        # Control loop at 20 Hz
        # Timer calls run_cbf() every 50ms
        # --------------------------------------------------------
        self.timer = self.create_timer(0.05, self.run_cbf)

        self.get_logger().info(
            f'CBF node ready | r_safe={self.cbf.r_safe}m | gamma={self.cbf.gamma}'
        )

    def on_cmd_vel_ref(self, msg):
        """
        Callback: store latest user velocity command.

        In testing: publish to /cmd_vel_ref manually with ros2 topic pub
        In production: joystick/haptic device publishes here
        """
        self.v_ref = msg.linear.x
        self.omega_ref = msg.angular.z

    def on_obstacle(self, msg):
        """
        Callback: store latest obstacle info from depth_node.

        msg.point.x = pixel x of closest obstacle
        msg.point.y = pixel y of closest obstacle
        msg.point.z = depth to closest obstacle (meters)
        """
        self.obstacle_depth = msg.point.z
        self.obstacle_px = msg.point.x
        self.has_obstacle_data = True

    def run_cbf(self):
        """
        Main control loop (20 Hz timer callback).

        1. Get latest user command and obstacle info
        2. Run CBF step (compute safe command + haptic force)
        3. Publish safe command to robot
        4. Publish haptic force to device
        """
        # Skip if no obstacle data yet
        if not self.has_obstacle_data:
            return

        # Run CBF
        v_safe, omega_safe, f_lin, f_ang, b = self.cbf.step(
            v_ref=self.v_ref,
            omega_ref=self.omega_ref,
            depth=self.obstacle_depth,
            pixel_x=self.obstacle_px
        )

        now = self.get_clock().now().to_msg()

        # --------------------------------------------------------
        # Publish safe velocity command to LIMO
        # --------------------------------------------------------
        cmd_msg = Twist()
        cmd_msg.linear.x = v_safe
        cmd_msg.angular.z = omega_safe
        self.cmd_pub.publish(cmd_msg)

        # --------------------------------------------------------
        # Publish haptic force feedback
        # --------------------------------------------------------
        # F = Kf * (u_safe - u_ref)
        # linear.x = braking force (negative = slow down)
        # angular.z = steering correction
        force_msg = WrenchStamped()
        force_msg.header.stamp = now
        force_msg.header.frame_id = 'base_link'
        force_msg.wrench.force.x = f_lin    # forward/backward
        force_msg.wrench.force.y = 0.0
        force_msg.wrench.force.z = 0.0
        force_msg.wrench.torque.z = f_ang   # steering
        self.force_pub.publish(force_msg)

        # --------------------------------------------------------
        # Publish debug info (throttled)
        # --------------------------------------------------------
        cbf_active = abs(v_safe - self.v_ref) > 0.01 or abs(omega_safe - self.omega_ref) > 0.01

        debug_msg = String()
        debug_msg.data = (
            f"b={b:.3f}m | d={self.obstacle_depth:.2f}m | "
            f"v_ref={self.v_ref:.2f}→v_safe={v_safe:.2f} | "
            f"F=({f_lin:.2f},{f_ang:.2f})N | "
            f"CBF={'ACTIVE' if cbf_active else 'idle'}"
        )
        self.debug_pub.publish(debug_msg)

        # Log when CBF is actively modifying commands
        if cbf_active:
            self.get_logger().warn(
                f"CBF ACTIVE: obstacle {self.obstacle_depth:.2f}m | "
                f"v {self.v_ref:.2f}→{v_safe:.2f} | F={f_lin:.2f}N",
                throttle_duration_sec=0.5
            )
        else:
            self.get_logger().info(
                f"Safe: b={b:.3f} d={self.obstacle_depth:.2f}m v={v_safe:.2f}",
                throttle_duration_sec=2.0
            )


def main(args=None):
    rclpy.init(args=args)
    node = CBFNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Send zero velocity on shutdown for safety
        stop_msg = Twist()
        node.cmd_pub.publish(stop_msg)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()