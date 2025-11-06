#!/usr/bin/env python3
"""
ROS 2 Near Node
Finds closest point in depth map and publishes both raw and smoothed data
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Vector3Stamped, PointStamped
from std_msgs.msg import Float32, Int32MultiArray
from cv_bridge import CvBridge
import numpy as np


class NearNode(Node):
    """Finds closest point and publishes raw + smoothed outputs"""
    
    def __init__(self):
        super().__init__('near_node')
        
        # Camera intrinsics
        self.declare_parameter('focal_length_x', 600.0)
        self.declare_parameter('focal_length_y', 600.0)
        self.declare_parameter('principal_point_x', 320.0)
        self.declare_parameter('principal_point_y', 240.0)
        
        # Smoothing parameters
        self.declare_parameter('smoothing_enabled', True)
        self.declare_parameter('smoothing_window', 5)
        
        self.fx = self.get_parameter('focal_length_x').value
        self.fy = self.get_parameter('focal_length_y').value
        self.cx = self.get_parameter('principal_point_x').value
        self.cy = self.get_parameter('principal_point_y').value
        
        self.smoothing_enabled = self.get_parameter('smoothing_enabled').value
        self.smoothing_window = self.get_parameter('smoothing_window').value
        
        # Smoothing buffers
        self.pixel_history = []
        self.depth_history = []
        
        self.bridge = CvBridge()
        self.latest_depth = None
        
        # Subscribe to depth
        self.depth_sub = self.create_subscription(
            Image,
            '/depth_estimation/depth',
            self.depth_callback,
            10
        )
        
        # Publishers - raw (non-averaged)
        self.direction_raw_pub = self.create_publisher(
            Vector3Stamped,
            '/direction/closest_point',
            10
        )
        
        self.depth_raw_pub = self.create_publisher(
            Float32,
            '/direction/min_depth',
            10
        )
        
        self.point_raw_pub = self.create_publisher(
            PointStamped,
            '/direction/closest_point_3d',
            10
        )
        
        # Publishers - smoothed (averaged)
        self.direction_smoothed_pub = self.create_publisher(
            Vector3Stamped,
            '/direction/closest_point_smoothed',
            10
        )
        
        self.depth_smoothed_pub = self.create_publisher(
            Float32,
            '/direction/min_depth_smoothed',
            10
        )
        
        self.point_smoothed_pub = self.create_publisher(
            PointStamped,
            '/direction/closest_point_3d_smoothed',
            10
        )
        
        # Publish pixel coordinates for visualization
        self.pixel_smoothed_pub = self.create_publisher(
            Int32MultiArray,
            '/direction/closest_pixel_smoothed',
            10
        )
        
        # Timer for processing
        self.timer = self.create_timer(0.1, self.process_callback)
        
        self.get_logger().info('Near Node initialized')
        self.get_logger().info(f'Camera intrinsics: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}')
        if self.smoothing_enabled:
            self.get_logger().info(f'Smoothing: {self.smoothing_window} frames')
    
    def depth_callback(self, msg):
        """Store latest depth map"""
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        except Exception as e:
            self.get_logger().error(f'Error processing depth: {e}')
    
    def find_closest_point(self, depth_map):
        """Find pixel with minimum depth"""
        valid_mask = (depth_map > 0) & np.isfinite(depth_map)
        
        if not np.any(valid_mask):
            return None
        
        valid_depths = depth_map[valid_mask]
        min_depth = np.min(valid_depths)
        
        min_locations = np.where((depth_map == min_depth) & valid_mask)
        v = min_locations[0][0]
        u = min_locations[1][0]
        
        return u, v, min_depth
    
    def smooth_closest_point(self, u, v, depth):
        """Apply temporal smoothing"""
        if not self.smoothing_enabled:
            return u, v, depth
        
        self.pixel_history.append((u, v))
        self.depth_history.append(depth)
        
        if len(self.pixel_history) > self.smoothing_window:
            self.pixel_history.pop(0)
            self.depth_history.pop(0)
        
        # Weighted average
        weights = np.linspace(0.5, 1.0, len(self.pixel_history))
        weights = weights / weights.sum()
        
        u_smooth = sum(w * p[0] for w, p in zip(weights, self.pixel_history))
        v_smooth = sum(w * p[1] for w, p in zip(weights, self.pixel_history))
        depth_smooth = sum(w * d for w, d in zip(weights, self.depth_history))
        
        return int(u_smooth), int(v_smooth), depth_smooth
    
    def pixel_to_3d(self, u, v, depth):
        """Convert pixel + depth to 3D point and direction"""
        x = (u - self.cx) * depth / self.fx
        y = (v - self.cy) * depth / self.fy
        z = depth
        
        point_3d = np.array([x, y, z])
        direction = point_3d / np.linalg.norm(point_3d)
        
        return direction, point_3d
    
    def process_callback(self):
        """Process depth and publish results"""
        if self.latest_depth is None:
            return
        
        result = self.find_closest_point(self.latest_depth)
        if result is None:
            return
        
        u_raw, v_raw, depth_raw = result
        
        # Get smoothed values
        u_smooth, v_smooth, depth_smooth = self.smooth_closest_point(u_raw, v_raw, depth_raw)
        
        # Convert to 3D
        direction_raw, point_raw = self.pixel_to_3d(u_raw, v_raw, depth_raw)
        direction_smooth, point_smooth = self.pixel_to_3d(u_smooth, v_smooth, depth_smooth)
        
        stamp = self.get_clock().now().to_msg()
        
        # Publish raw data
        direction_raw_msg = Vector3Stamped()
        direction_raw_msg.header.stamp = stamp
        direction_raw_msg.header.frame_id = 'camera'
        direction_raw_msg.vector.x = float(direction_raw[0])
        direction_raw_msg.vector.y = float(direction_raw[1])
        direction_raw_msg.vector.z = float(direction_raw[2])
        self.direction_raw_pub.publish(direction_raw_msg)
        
        depth_raw_msg = Float32()
        depth_raw_msg.data = float(depth_raw)
        self.depth_raw_pub.publish(depth_raw_msg)
        
        point_raw_msg = PointStamped()
        point_raw_msg.header.stamp = stamp
        point_raw_msg.header.frame_id = 'camera'
        point_raw_msg.point.x = float(point_raw[0])
        point_raw_msg.point.y = float(point_raw[1])
        point_raw_msg.point.z = float(point_raw[2])
        self.point_raw_pub.publish(point_raw_msg)
        
        # Publish smoothed data
        direction_smooth_msg = Vector3Stamped()
        direction_smooth_msg.header.stamp = stamp
        direction_smooth_msg.header.frame_id = 'camera'
        direction_smooth_msg.vector.x = float(direction_smooth[0])
        direction_smooth_msg.vector.y = float(direction_smooth[1])
        direction_smooth_msg.vector.z = float(direction_smooth[2])
        self.direction_smoothed_pub.publish(direction_smooth_msg)
        
        depth_smooth_msg = Float32()
        depth_smooth_msg.data = float(depth_smooth)
        self.depth_smoothed_pub.publish(depth_smooth_msg)
        
        point_smooth_msg = PointStamped()
        point_smooth_msg.header.stamp = stamp
        point_smooth_msg.header.frame_id = 'camera'
        point_smooth_msg.point.x = float(point_smooth[0])
        point_smooth_msg.point.y = float(point_smooth[1])
        point_smooth_msg.point.z = float(point_smooth[2])
        self.point_smoothed_pub.publish(point_smooth_msg)
        
        # Publish smoothed pixel coordinates
        pixel_msg = Int32MultiArray()
        pixel_msg.data = [u_smooth, v_smooth]
        self.pixel_smoothed_pub.publish(pixel_msg)


def main(args=None):
    rclpy.init(args=args)
    node = NearNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
