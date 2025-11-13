#!/usr/bin/env python3
"""
ROS 2 Combined Visualization Node
Shows camera, depth map, and direction detection in one window
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np


class CombinedVisualizationNode(Node):
    """ROS 2 node for combined visualization"""
    
    def __init__(self):
        super().__init__('visualization_node')
        
        # Create CV Bridge
        self.bridge = CvBridge()
        
        # Store latest data
        self.latest_image = None
        self.latest_depth = None
        self.latest_direction_viz = None
        
        # Create subscribers
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
        
        self.direction_viz_sub = self.create_subscription(
            Image,
            '/direction/visualization',
            self.direction_viz_callback,
            10
        )
        
        # Create timer to update display at 30 Hz
        self.timer = self.create_timer(1.0/30.0, self.display_callback)
        
        # Display window
        self.window_created = False
        
        self.get_logger().info('Combined Visualization node initialized')
        self.get_logger().info('Subscribing to:')
        self.get_logger().info('  /camera/image')
        self.get_logger().info('  /depth_estimation/depth')
        self.get_logger().info('  /direction/visualization')
        self.get_logger().info('')
        self.get_logger().info('Controls:')
        self.get_logger().info("  'q' - quit")
        self.get_logger().info("  's' - save current visualization")
    
    def image_callback(self, msg):
        """Process incoming camera images"""
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')
    
    def depth_callback(self, msg):
        """Process incoming depth maps"""
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        except Exception as e:
            self.get_logger().error(f'Error processing depth: {e}')
    
    def direction_viz_callback(self, msg):
        """Process incoming direction visualization"""
        try:
            self.latest_direction_viz = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Error processing direction viz: {e}')
    
    def create_combined_view(self):
        """Combine all three views into one image"""
        # Target size for each panel
        panel_height = 360
        panel_width = 480
        
        # Create placeholder panels
        placeholder = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
        
        # Process camera image
        if self.latest_image is not None:
            camera_panel = cv2.resize(self.latest_image, (panel_width, panel_height))
            cv2.putText(camera_panel, "Camera", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            camera_panel = placeholder.copy()
            cv2.putText(camera_panel, "Camera (waiting...)", (10, panel_height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
        
        # Process depth map
        if self.latest_depth is not None:
            # Normalize and colorize depth
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
        
        # Process direction visualization
        if self.latest_direction_viz is not None:
            direction_panel = cv2.resize(self.latest_direction_viz, (panel_width, panel_height))
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
        """Display combined view at regular interval"""
        if not self.window_created:
            cv2.namedWindow("Depth Estimation System", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Depth Estimation System", 1440, 360)
            self.window_created = True
        
        # Create and display combined view
        combined = self.create_combined_view()
        cv2.imshow("Depth Estimation System", combined)
        
        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.get_logger().info('Quit requested')
            rclpy.shutdown()
        elif key == ord('s'):
            # Save current visualization
            import time
            filename = f"combined_visualization_{int(time.time())}.jpg"
            cv2.imwrite(filename, combined)
            self.get_logger().info(f'Saved visualization to {filename}')
    
    def destroy_node(self):
        """Cleanup when node is destroyed"""
        cv2.destroyAllWindows()
        self.get_logger().info('Visualization node cleanup complete')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    node = CombinedVisualizationNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
