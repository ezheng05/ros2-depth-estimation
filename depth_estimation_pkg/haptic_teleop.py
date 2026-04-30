import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, WrenchStamped
from omni_msgs.msg import OmniState, OmniFeedback


class HapticTeleopNode(Node):
    def __init__(self):
        super().__init__('haptic_teleop_node')

        # position is in mm from driver, these scales convert to m/s
        self.declare_parameter('dz', 15.0)       # dead zone in mm
        self.declare_parameter('k_lin', 0.004)    # mm -> m/s
        self.declare_parameter('k_ang', 0.004)    # mm -> rad/s
        self.declare_parameter('f_scale', 0.05)   # cbf force -> device force
        self.declare_parameter('f_max', 0.15)     # max force to device (N)
        self.declare_parameter('f_alpha', 0.3)    # low-pass filter coeff
        self.declare_parameter('f_on', True)       # enable force feedback

        self.dz = self.get_parameter('dz').value
        self.k_lin = self.get_parameter('k_lin').value
        self.k_ang = self.get_parameter('k_ang').value
        self.f_scale = self.get_parameter('f_scale').value
        self.f_max = self.get_parameter('f_max').value
        self.f_alpha = self.get_parameter('f_alpha').value
        self.f_on = self.get_parameter('f_on').value

        self.fx = 0.0
        self.fy = 0.0

        self.vel_pub = self.create_publisher(Twist, '/cmd_vel_ref', 10)
        self.force_pub = self.create_publisher(OmniFeedback, '/phantom/force_feedback', 10)

        self.create_subscription(OmniState, '/phantom/state', self.on_state, 10)
        self.create_subscription(WrenchStamped, '/haptic/force', self.on_force, 10)

        self.get_logger().info(
            f"ready | dz={self.dz}mm k_lin={self.k_lin} k_ang={self.k_ang} "
            f"f_scale={self.f_scale} f_max={self.f_max}N f_on={self.f_on}")

    def deadzone(self, val, dz, k):
        if abs(val) < dz:
            return 0.0
        sign = 1.0 if val > 0 else -1.0
        return (abs(val) - dz) * k * sign

    def on_state(self, msg):
        # position from omni_state driver is in mm
        x = msg.pose.position.x
        y = msg.pose.position.y

        cmd = Twist()
        cmd.linear.x = max(-0.2, min(0.3, self.deadzone(x, self.dz, self.k_lin)))
        cmd.angular.z = max(-1.0, min(1.0, self.deadzone(y, self.dz, self.k_ang)))
        self.vel_pub.publish(cmd)

    def on_force(self, msg):
        out = OmniFeedback()
        if not self.f_on:
            self.force_pub.publish(out)
            return

        rx = msg.wrench.force.x * self.f_scale
        ry = msg.wrench.torque.z * self.f_scale

        # clamp magnitude
        mag = math.sqrt(rx*rx + ry*ry)
        if mag > self.f_max:
            s = self.f_max / mag
            rx *= s
            ry *= s

        # low-pass filter to prevent jitter
        a = self.f_alpha
        self.fx = a * rx + (1.0 - a) * self.fx
        self.fy = a * ry + (1.0 - a) * self.fy

        out.force.x = self.fx
        out.force.y = self.fy
        out.force.z = 0.0
        self.force_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = HapticTeleopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
