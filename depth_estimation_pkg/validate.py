"""
validate.py - Compare ZoeDepth estimates with hardware depth sensor

Displays side-by-side:
    Left: Color image with ZoeDepth estimate
    Right: Hardware depth sensor reading

Topics:
    Subscribes:
        /camera/color/image_raw
        /camera/depth/image_raw
        /depth/closest (from depth_node)
    Publishes:
        /depth/validate (side-by-side comparison image)

Testing:
    ros2 run depth_estimation_pkg validate
    rqt_image_view → /depth/validate
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import cv2
import numpy as np


class ValidateNode(Node):
    """Side-by-side comparison of ZoeDepth vs hardware depth."""

    def __init__(self):
        super().__init__('validate_node')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscribers
        self.color_sub = self.create_subscription(
            Image, '/camera/color/image_raw', self.on_color, qos
        )
        self.depth_sub = self.create_subscription(
            Image, '/camera/depth/image_raw', self.on_depth, qos
        )
        self.point_sub = self.create_subscription(
            PointStamped, '/depth/closest', self.on_point, 10
        )

        # Publisher
        self.pub = self.create_publisher(Image, '/depth/validate', 10)

        self.bridge = CvBridge()
        
        # Latest data
        self.color_img = None
        self.depth_img = None  # hardware depth (uint16, mm)
        self.zoe_point = None  # (px, py, zoe_depth)

        self.get_logger().info('Validate node ready')

    def on_color(self, msg):
        """Store latest color image."""
        try:
            self.color_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.publish_comparison()
        except Exception as e:
            self.get_logger().error(f'Color error: {e}')

    def on_depth(self, msg):
        """Store latest hardware depth."""
        try:
            # Hardware depth is uint16 in mm
            self.depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f'Depth error: {e}')

    def on_point(self, msg):
        """Store latest ZoeDepth estimate."""
        self.zoe_point = (
            int(msg.point.x),
            int(msg.point.y),
            msg.point.z  # meters
        )

    def publish_comparison(self):
        """Create and publish side-by-side comparison."""
        if self.color_img is None or self.depth_img is None or self.zoe_point is None:
            return

        px, py, zoe_depth = self.zoe_point

        # --- Left side: Color with ZoeDepth overlay ---
        left = self.color_img.copy()
        
        # Draw marker at ZoeDepth closest point
        cv2.circle(left, (px, py), 12, (0, 255, 255), 3)  # yellow circle
        cv2.putText(left, f'ZoeDepth: {zoe_depth:.2f}m', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # --- Right side: Hardware depth visualization ---
        # Normalize depth for display
        depth_display = self.depth_img.copy().astype(float)
        depth_display[depth_display == 0] = np.nan  # invalid = black
        
        d_min = np.nanmin(depth_display)
        d_max = np.nanmax(depth_display)
        depth_norm = (depth_display - d_min) / (d_max - d_min + 1e-6)
        depth_norm = np.nan_to_num(depth_norm, nan=0)
        
        # Apply colormap
        depth_color = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Resize depth to match color (depth is 640x400, color is 640x480)
        depth_color = cv2.resize(depth_color, (left.shape[1], left.shape[0]))
        
        # Get hardware depth at ZoeDepth's closest point
        # Scale py for different image heights
        hw_py = int(py * self.depth_img.shape[0] / self.color_img.shape[0])
        hw_px = min(px, self.depth_img.shape[1] - 1)
        hw_py = min(hw_py, self.depth_img.shape[0] - 1)
        
        hw_depth_mm = self.depth_img[hw_py, hw_px]
        hw_depth_m = hw_depth_mm / 1000.0 if hw_depth_mm > 0 else 0
        
        # Draw marker on depth image
        cv2.circle(depth_color, (px, py), 12, (255, 255, 255), 3)  # white circle
        cv2.putText(depth_color, f'Hardware: {hw_depth_m:.2f}m', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # --- Combine side by side ---
        combined = np.hstack([left, depth_color])

        # --- Add comparison stats at bottom ---
        error = abs(zoe_depth - hw_depth_m) if hw_depth_m > 0 else 0
        error_pct = (error / hw_depth_m * 100) if hw_depth_m > 0 else 0
        
        stats_text = f'ZoeDepth: {zoe_depth:.3f}m | Hardware: {hw_depth_m:.3f}m | Error: {error:.3f}m ({error_pct:.1f}%)'
        cv2.putText(combined, stats_text, (10, combined.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Publish
        out_msg = self.bridge.cv2_to_imgmsg(combined, encoding='bgr8')
        self.pub.publish(out_msg)

        # Log comparison
        self.get_logger().info(
            f'Zoe: {zoe_depth:.3f}m | HW: {hw_depth_m:.3f}m | Err: {error:.3f}m',
            throttle_duration_sec=1.0
        )


def main(args=None):
    rclpy.init(args=args)
    node = ValidateNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()