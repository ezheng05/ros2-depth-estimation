#!/usr/bin/env python3
"""
ROS 2 Image Visualization Node
Shows camera, depth map, and direction overlay in OpenCV window
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float32, Int32MultiArray
from cv_bridge import CvBridge
import cv2
import numpy as np


class ImageVizNode(Node):
    """OpenCV visualization for camera, depth, and direction"""
    
    def __init__(self):
        super().__init__('image_viz_node')
        
        self.bridge = CvBridge()
        
        # Store latest data
        self.latest_image = None
        self.latest_depth = None
        self.latest_pixel = None
        self.latest_point_3d = None
        self.latest_min_depth = None
        
        # Subscribe to topics
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image',
            self.image_callback,
            10
        )
        
        self.depth_sub = self.create_subscription(
            Image,
            '/depth_estimation/depth',
            self.depth_callback,
            10
        )
        
        self.pixel_sub = self.create_subscription(
            Int32MultiArray,
            '/direction/closest_pixel_smoothed',
            self.pixel_callback,
            10
        )
        
        self.point_sub = self.create_subscription(
            PointStamped,
            '/direction/closest_point_3d_smoothed',
            self.point_callback,
            10
        )
        
        self.min_depth_sub = self.create_subscription(
            Float32,
            '/direction/min_depth_smoothed',
            self.min_depth_callback,
            10
        )
        
        # Display timer
        self.timer = self.create_timer(1.0/30.0, self.display_callback)
        self.window_created = False
        
        self.get_logger().info('Image Viz Node initialized')
    
    def image_callback(self, msg):
        """Store latest camera image"""
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')
    
    def depth_callback(self, msg):
        """Store latest depth map"""
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        except Exception as e:
            self.get_logger().error(f'Error processing depth: {e}')
    
    def pixel_callback(self, msg):
        """Store latest pixel coordinates"""
        if len(msg.data) == 2:
            self.latest_pixel = (msg.data[0], msg.data[1])
    
    def point_callback(self, msg):
        """Store latest 3D point"""
        self.latest_point_3d = (msg.point.x, msg.point.y, msg.point.z)
    
    def min_depth_callback(self, msg):
        """Store latest minimum depth"""
        self.latest_min_depth = msg.data
    
    def create_direction_overlay(self, image, pixel, point_3d, depth):
        """Create image with crosshair and arrow"""
        if image is None or pixel is None:
            return image
        
        vis = image.copy()
        h, w = vis.shape[:2]
        
        u, v = pixel
        
        # Scale pixel from depth map (64x64) to camera image (640x480)
        depth_size = 64  # Known depth map size
        scale_x = w / depth_size
        scale_y = h / depth_size
        u_scaled = int(u * scale_x)
        v_scaled = int(v * scale_y)
        
        # Clamp
        u_scaled = max(0, min(w-1, u_scaled))
        v_scaled = max(0, min(h-1, v_scaled))
        
        # Draw crosshair
        cv2.drawMarker(vis, (u_scaled, v_scaled), (0, 0, 255), cv2.MARKER_CROSS, 30, 3)
        cv2.circle(vis, (u_scaled, v_scaled), 20, (0, 0, 255), 2)
        
        # Draw arrow from center
        center_x, center_y = w // 2, h // 2
        cv2.arrowedLine(vis, (center_x, center_y), (u_scaled, v_scaled), (0, 255, 0), 3, tipLength=0.3)
        
        # Text
        cv2.putText(vis, f'Pixel: ({u},{v})', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if depth:
            cv2.putText(vis, f'Depth: {depth:.2f}m', (10, 55), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if point_3d:
            x, y, z = point_3d
            cv2.putText(vis, f'3D: [{x:.2f}, {y:.2f}, {z:.2f}]m', (10, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return vis
    
    def create_combined_view(self):
        """Combine camera, depth, and direction into one image"""
        panel_height = 360
        panel_width = 480
        
        placeholder = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
        
        # Camera panel
        if self.latest_image is not None:
            camera_panel = cv2.resize(self.latest_image, (panel_width, panel_height))
            cv2.putText(camera_panel, "Camera", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            camera_panel = placeholder.copy()
            cv2.putText(camera_panel, "Camera (waiting...)", (10, panel_height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
        
        # Depth panel
        if self.latest_depth is not None:
            depth_normalized = cv2.normalize(self.latest_depth, None, 0, 255, 
                                            cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_PLASMA)
            depth_panel = cv2.resize(depth_colored, (panel_width, panel_height))
            cv2.putText(depth_panel, "Depth Map", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            depth_panel = placeholder.copy()
            cv2.putText(depth_panel, "Depth (waiting...)", (10, panel_height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
        
        # Direction panel
        if self.latest_image is not None and self.latest_pixel is not None:
            direction_img = self.create_direction_overlay(
                self.latest_image, 
                self.latest_pixel,
                self.latest_point_3d,
                self.latest_min_depth
            )
            direction_panel = cv2.resize(direction_img, (panel_width, panel_height))
        else:
            direction_panel = placeholder.copy()
            cv2.putText(direction_panel, "Direction (waiting...)", (10, panel_height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
        
        # Combine horizontally
        combined = np.hstack([camera_panel, depth_panel, direction_panel])
        
        # Add separators
        combined[:, panel_width-1:panel_width+1, :] = [255, 255, 255]
        combined[:, 2*panel_width-1:2*panel_width+1, :] = [255, 255, 255]
        
        return combined
    
    def display_callback(self):
        """Display combined view"""
        if not self.window_created:
            cv2.namedWindow("Depth Estimation System", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Depth Estimation System", 1440, 360)
            self.window_created = True
        
        combined = self.create_combined_view()
        cv2.imshow("Depth Estimation System", combined)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.get_logger().info('Quit requested')
            rclpy.shutdown()
        elif key == ord('s'):
            import time
            filename = f"visualization_{int(time.time())}.jpg"
            cv2.imwrite(filename, combined)
            self.get_logger().info(f'Saved to {filename}')
    
    def destroy_node(self):
        """Cleanup"""
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ImageVizNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
