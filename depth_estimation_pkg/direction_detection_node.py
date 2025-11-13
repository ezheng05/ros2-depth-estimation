#!/usr/bin/env python3
"""
ROS 2 Direction Detection Node with Temporal Smoothing
Finds the closest point in the depth map and returns a 3D direction vector
Includes smoothing filter to reduce jitter
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Vector3Stamped
from std_msgs.msg import Float32
from cv_bridge import CvBridge
import cv2
import numpy as np


class DirectionDetectionNode(Node):
    """ROS 2 node for detecting direction to closest point with smoothing"""
    
    def __init__(self):
        super().__init__('direction_detection_node')
        
        # Camera intrinsic parameters (will be updated from calibration or params)
        self.declare_parameter('focal_length_x', 600.0)  # fx
        self.declare_parameter('focal_length_y', 600.0)  # fy
        self.declare_parameter('principal_point_x', 320.0)  # cx (image center x)
        self.declare_parameter('principal_point_y', 240.0)  # cy (image center y)
        self.declare_parameter('use_camera_info', False)  # Use CameraInfo topic if available
        
        # ROI (Region of Interest) parameters for selecting part of image
        self.declare_parameter('roi_enabled', False)
        self.declare_parameter('roi_x', 160)  # Top-left x
        self.declare_parameter('roi_y', 120)  # Top-left y
        self.declare_parameter('roi_width', 320)  # ROI width
        self.declare_parameter('roi_height', 240)  # ROI height
        
        # Smoothing parameters
        self.declare_parameter('smoothing_enabled', True)
        self.declare_parameter('smoothing_window', 5)  # Number of frames to average
        
        # Get parameters
        self.fx = self.get_parameter('focal_length_x').value
        self.fy = self.get_parameter('focal_length_y').value
        self.cx = self.get_parameter('principal_point_x').value
        self.cy = self.get_parameter('principal_point_y').value
        self.use_camera_info = self.get_parameter('use_camera_info').value
        
        self.roi_enabled = self.get_parameter('roi_enabled').value
        self.roi_x = self.get_parameter('roi_x').value
        self.roi_y = self.get_parameter('roi_y').value
        self.roi_width = self.get_parameter('roi_width').value
        self.roi_height = self.get_parameter('roi_height').value
        
        self.smoothing_enabled = self.get_parameter('smoothing_enabled').value
        self.smoothing_window = self.get_parameter('smoothing_window').value
        
        # Smoothing buffers
        self.pixel_history = []  # Store (u, v) tuples
        self.depth_history = []  # Store depth values
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # Store latest data
        self.latest_depth = None
        self.latest_image = None
        
        # Create subscribers
        self.depth_sub = self.create_subscription(
            Image,
            '/depth_estimation/depth',
            self.depth_callback,
            10
        )
        
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image',
            self.image_callback,
            10
        )
        
        if self.use_camera_info:
            self.camera_info_sub = self.create_subscription(
                CameraInfo,
                '/camera/camera_info',
                self.camera_info_callback,
                10
            )
        
        # Create publishers
        self.direction_pub = self.create_publisher(
            Vector3Stamped,
            '/direction/closest_point',
            10
        )
        
        self.min_depth_pub = self.create_publisher(
            Float32,
            '/direction/min_depth',
            10
        )
        
        self.visualization_pub = self.create_publisher(
            Image,
            '/direction/visualization',
            10
        )
        
        # Timer for processing
        self.timer = self.create_timer(0.1, self.process_callback)  # 10 Hz
        
        self.get_logger().info('Direction Detection Node initialized')
        self.get_logger().info(f'Camera intrinsics: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}')
        if self.roi_enabled:
            self.get_logger().info(f'ROI enabled: x={self.roi_x}, y={self.roi_y}, w={self.roi_width}, h={self.roi_height}')
        if self.smoothing_enabled:
            self.get_logger().info(f'Smoothing enabled: window={self.smoothing_window} frames')
    
    def camera_info_callback(self, msg):
        """Update camera parameters from CameraInfo message"""
        self.fx = msg.k[0]  # K[0,0]
        self.fy = msg.k[4]  # K[1,1]
        self.cx = msg.k[2]  # K[0,2]
        self.cy = msg.k[5]  # K[1,2]
        self.get_logger().info(f'Updated camera intrinsics from CameraInfo: fx={self.fx:.2f}, fy={self.fy:.2f}')
    
    def depth_callback(self, msg):
        """Store latest depth map"""
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        except Exception as e:
            self.get_logger().error(f'Error processing depth: {e}')
    
    def image_callback(self, msg):
        """Store latest image"""
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')
    
    def smooth_closest_point(self, u, v, depth):
        """
        Apply temporal smoothing to closest point coordinates
        
        Args:
            u, v: Current pixel coordinates
            depth: Current depth value
            
        Returns:
            Smoothed (u, v, depth)
        """
        if not self.smoothing_enabled:
            return u, v, depth
        
        # Add to history
        self.pixel_history.append((u, v))
        self.depth_history.append(depth)
        
        # Keep only last N frames
        if len(self.pixel_history) > self.smoothing_window:
            self.pixel_history.pop(0)
            self.depth_history.pop(0)
        
        # Calculate weighted average (more recent frames have higher weight)
        weights = np.linspace(0.5, 1.0, len(self.pixel_history))
        weights = weights / weights.sum()
        
        u_smooth = sum(w * p[0] for w, p in zip(weights, self.pixel_history))
        v_smooth = sum(w * p[1] for w, p in zip(weights, self.pixel_history))
        depth_smooth = sum(w * d for w, d in zip(weights, self.depth_history))
        
        return int(u_smooth), int(v_smooth), depth_smooth
    
    def pixel_to_direction(self, u, v, depth):
        """
        Convert pixel coordinates to 3D direction vector in camera frame
        
        Camera frame:
        - X: right
        - Y: down
        - Z: forward (into scene)
        
        Args:
            u: pixel x coordinate
            v: pixel y coordinate
            depth: depth value in meters
            
        Returns:
            numpy array: [X, Y, Z] direction vector (normalized)
        """
        # Convert pixel to normalized camera coordinates
        x = (u - self.cx) * depth / self.fx
        y = (v - self.cy) * depth / self.fy
        z = depth
        
        # Create 3D point
        point_3d = np.array([x, y, z])
        
        # Normalize to get direction vector
        direction = point_3d / np.linalg.norm(point_3d)
        
        return direction, point_3d
    
    def find_closest_point(self, depth_map):
        """
        Find pixel with minimum depth in the depth map (or ROI)
        
        Args:
            depth_map: numpy array of depth values
            
        Returns:
            tuple: (u, v, min_depth) or None if no valid depth
        """
        # Apply ROI if enabled
        if self.roi_enabled:
            # Ensure ROI is within bounds
            h, w = depth_map.shape
            x1 = max(0, self.roi_x)
            y1 = max(0, self.roi_y)
            x2 = min(w, self.roi_x + self.roi_width)
            y2 = min(h, self.roi_y + self.roi_height)
            
            roi_depth = depth_map[y1:y2, x1:x2]
            offset_x = x1
            offset_y = y1
        else:
            roi_depth = depth_map
            offset_x = 0
            offset_y = 0
        
        # Find minimum depth (excluding invalid values)
        valid_mask = (roi_depth > 0) & np.isfinite(roi_depth)
        
        if not np.any(valid_mask):
            return None
        
        # Find minimum depth location
        valid_depths = roi_depth[valid_mask]
        min_depth = np.min(valid_depths)
        
        # Find pixel coordinates of minimum depth
        min_locations = np.where((roi_depth == min_depth) & valid_mask)
        
        # Take first occurrence
        v_local = min_locations[0][0]
        u_local = min_locations[1][0]
        
        # Convert back to full image coordinates
        u = u_local + offset_x
        v = v_local + offset_y
        
        return u, v, min_depth
    
    def process_callback(self):
        """Process depth map and find direction to closest point"""
        if self.latest_depth is None:
            return
        
        try:
            # Find closest point
            result = self.find_closest_point(self.latest_depth)
            
            if result is None:
                self.get_logger().warn('No valid depth points found')
                return
            
            u_raw, v_raw, min_depth_raw = result
            
            # Apply smoothing
            u, v, min_depth = self.smooth_closest_point(u_raw, v_raw, min_depth_raw)
            
            # Convert to 3D direction
            direction, point_3d = self.pixel_to_direction(u, v, min_depth)
            
            # Publish direction vector
            direction_msg = Vector3Stamped()
            direction_msg.header.stamp = self.get_clock().now().to_msg()
            direction_msg.header.frame_id = 'camera'
            direction_msg.vector.x = float(direction[0])
            direction_msg.vector.y = float(direction[1])
            direction_msg.vector.z = float(direction[2])
            self.direction_pub.publish(direction_msg)
            
            # Publish minimum depth
            depth_msg = Float32()
            depth_msg.data = float(min_depth)
            self.min_depth_pub.publish(depth_msg)
            
            # Create visualization
            if self.latest_image is not None:
                vis_image = self.create_visualization(
                    self.latest_image, self.latest_depth, u, v, min_depth, direction, point_3d
                )
                vis_msg = self.bridge.cv2_to_imgmsg(vis_image, encoding='bgr8')
                vis_msg.header.stamp = self.get_clock().now().to_msg()
                self.visualization_pub.publish(vis_msg)
            
            # Log info periodically
            if hasattr(self, '_log_counter'):
                self._log_counter += 1
            else:
                self._log_counter = 0
            
            if self._log_counter % 10 == 0:
                self.get_logger().info(
                    f'Closest point: pixel=({u},{v}), depth={min_depth:.2f}m, '
                    f'direction=[{direction[0]:.2f}, {direction[1]:.2f}, {direction[2]:.2f}], '
                    f'3D point=[{point_3d[0]:.2f}, {point_3d[1]:.2f}, {point_3d[2]:.2f}]'
                )
        
        except Exception as e:
            self.get_logger().error(f'Error in processing: {e}')
    
    def create_visualization(self, image, depth_map, u, v, min_depth, direction, point_3d):
        """Create visualization showing closest point and direction"""
        vis = image.copy()
        h, w = vis.shape[:2]
        
        # Scale coordinates from depth map to camera image
        depth_h, depth_w = depth_map.shape[:2]
        scale_x = w / depth_w
        scale_y = h / depth_h
        u_scaled = int(u * scale_x)
        v_scaled = int(v * scale_y)
        
        # Draw ROI if enabled
        if self.roi_enabled:
            cv2.rectangle(
                vis,
                (self.roi_x, self.roi_y),
                (self.roi_x + self.roi_width, self.roi_y + self.roi_height),
                (255, 255, 0), 2
            )
        
        # Draw crosshair at closest point
        cv2.drawMarker(vis, (u_scaled, v_scaled), (0, 0, 255), cv2.MARKER_CROSS, 30, 3)
        
        # Draw circle around closest point
        cv2.circle(vis, (u_scaled, v_scaled), 20, (0, 0, 255), 2)
        
        # Draw direction arrow from center to closest point
        center_x, center_y = w // 2, h // 2
        cv2.arrowedLine(vis, (center_x, center_y), (u_scaled, v_scaled), (0, 255, 0), 3, tipLength=0.3)
        
        # Add text info
        info_y = 30
        cv2.putText(vis, f'Closest: ({u},{v})', (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        info_y += 25
        cv2.putText(vis, f'Depth: {min_depth:.2f}m', (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        info_y += 25
        cv2.putText(vis, f'Dir: [{direction[0]:.2f}, {direction[1]:.2f}, {direction[2]:.2f}]', 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        info_y += 25
        cv2.putText(vis, f'3D: [{point_3d[0]:.2f}, {point_3d[1]:.2f}, {point_3d[2]:.2f}]m', 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show smoothing status
        if self.smoothing_enabled:
            info_y += 25
            cv2.putText(vis, f'Smooth: {len(self.pixel_history)}/{self.smoothing_window}', 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        return vis


def main(args=None):
    rclpy.init(args=args)
    
    node = DirectionDetectionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
