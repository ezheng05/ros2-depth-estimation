import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist, WrenchStamped
# Note: Adjust the import paths based on your specific omni_msgs package structure
from omni_msgs.msg import OmniState, OmniFeedback 

class HapticTeleopNode(Node):
    def __init__(self):
        super().__init__('haptic_teleop_node')
        
        # --- Parameters ---
        self.declare_parameter('dead_zone', 0.02)
        self.declare_parameter('scale_linear', 5.0)
        self.declare_parameter('scale_angular', 5.0)
        
        self.dead_zone = self.get_parameter('dead_zone').value
        self.scale_linear = self.get_parameter('scale_linear').value
        self.scale_angular = self.get_parameter('scale_angular').value

        # --- Publishers & Subscribers ---
        # 1. Publish user's command to the CBF node
        self.vel_pub = self.create_publisher(Twist, '/cmd_vel_ref', 10)
        
        # 2. Publish actual haptic forces to the hardware driver
        self.force_pub = self.create_publisher(OmniFeedback, '/phantom/force_feedback', 10)
        
        # 3. Read position from hardware driver
        self.pose_sub = self.create_subscription(
            OmniState, '/phantom/state', self.haptic_state_callback, 10)
            
        # 4. Read computed force from CBF node
        self.force_sub = self.create_subscription(
            WrenchStamped, '/haptic/force', self.cbf_force_callback, 10)

        self.get_logger().info("Haptic Teleop Bridge Ready. Translating CBF logic to Geomagic Touch.")

    def apply_dead_zone(self, value, dead_zone, scale):
        if abs(value) < dead_zone:
            return 0.0
        elif value > 0:
            return (value - dead_zone) * scale
        else:
            return (value + dead_zone) * scale

    def haptic_state_callback(self, msg):
        """ Translates physical stylus position into /cmd_vel_ref """
        x = msg.pose.position.x
        y = msg.pose.position.y
        
        cmd = Twist()
        # Stylus forward/back -> linear velocity
        cmd.linear.x = self.apply_dead_zone(x, self.dead_zone, self.scale_linear)
        # Stylus left/right -> angular velocity
        cmd.angular.z = self.apply_dead_zone(y, self.dead_zone, self.scale_angular)
        
        self.vel_pub.publish(cmd)

    def cbf_force_callback(self, msg):
        """ Translates CBF Wrench forces into hardware OmniFeedback forces """
        force_msg = OmniFeedback()
        
        # cbf.py publishes braking force on linear.x and steering torque on angular.z
        f_linear = msg.wrench.force.x
        f_angular = msg.wrench.torque.z
        
        # Map to physical hardware axes
        # X pushes the stylus forward/backward into the user's hand
        # Y pushes the stylus left/right
        force_msg.force.x = f_linear 
        force_msg.force.y = f_angular 
        force_msg.force.z = 0.0
        
        self.force_pub.publish(force_msg)

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