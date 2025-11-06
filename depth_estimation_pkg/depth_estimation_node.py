#!/usr/bin/env python3
"""
ROS 2 Depth Estimation Node - Optimized Version
Features: frame skipping, smaller images, performance monitoring
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2
import numpy as np
import time

# Import the depth estimation system
from depth_estimation_pkg.main_3d_system import DepthEstimationSystem


class DepthEstimationNode(Node):
    """ROS 2 node for depth estimation with optimizations"""
    
    def __init__(self, target_size=(64, 64), use_fast_processor=True, skip_frames=2):
        super().__init__('depth_estimation_node')
        
        # Declare parameters
        self.declare_parameter('target_size_width', target_size[0])
        self.declare_parameter('target_size_height', target_size[1])
        self.declare_parameter('use_fast_processor', use_fast_processor)
        self.declare_parameter('skip_frames', skip_frames)  # Process every Nth frame
        
        # Get parameters
        width = self.get_parameter('target_size_width').value
        height = self.get_parameter('target_size_height').value
        self.target_size = (width, height)
        self.use_fast_processor = self.get_parameter('use_fast_processor').value
        self.skip_frames = self.get_parameter('skip_frames').value
        
        # Frame skipping counter
        self.frame_counter = 0
        self.last_depth_map = None
        
        # Initialize depth estimation system
        self.get_logger().info('Initializing depth estimation system...')
        self.get_logger().info(f'Optimization: Processing every {self.skip_frames} frames')
        self.depth_system = DepthEstimationSystem(
            target_size=self.target_size,
            use_fast_processor=self.use_fast_processor
        )
        
        if not self.depth_system.load_model():
            self.get_logger().error('Failed to load depth estimation model')
            raise RuntimeError('Failed to load depth estimation model')
        
        self.get_logger().info(f'Depth system initialized with target size: {self.target_size}')
        
        # Create CV Bridge
        self.bridge = CvBridge()
        
        # Create publishers
        self.depth_pub = self.create_publisher(Image, '/depth_estimation/depth', 10)
        self.visualization_pub = self.create_publisher(Image, '/depth_estimation/visualization', 10)
        
        # Create subscriber
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image',
            self.image_callback,
            10
        )
        
        # Stats
        self.processed_count = 0
        self.start_time = time.time()
        self.processing_times = []
        
        self.get_logger().info('Depth estimation node initialized (OPTIMIZED)')
        self.get_logger().info('Subscribing to: /camera/image')
        self.get_logger().info('Publishing to: /depth_estimation/depth')
        self.get_logger().info('Publishing to: /depth_estimation/visualization')
    
    def image_callback(self, msg):
        """Process incoming camera images with frame skipping"""
        self.frame_counter += 1
        
        # Skip frames for performance
        if self.frame_counter % self.skip_frames != 0:
            # Republish last depth map if available
            if self.last_depth_map is not None:
                depth_msg = self.bridge.cv2_to_imgmsg(self.last_depth_map, encoding='32FC1')
                depth_msg.header = msg.header
                self.depth_pub.publish(depth_msg)
            return
        
        try:
            # Convert ROS Image message to OpenCV image
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Measure processing time
            start_time = time.time()
            
            # Estimate depth
            depth_map = self.depth_system.estimate_depth(image)
            
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            if depth_map is not None:
                self.last_depth_map = depth_map
                
                # Publish depth map as Image message
                depth_msg = self.bridge.cv2_to_imgmsg(depth_map, encoding='32FC1')
                depth_msg.header = msg.header
                self.depth_pub.publish(depth_msg)
                
                # Create visualization
                visualization = self.create_visualization(image, depth_map)
                vis_msg = self.bridge.cv2_to_imgmsg(visualization, encoding='bgr8')
                vis_msg.header = msg.header
                self.visualization_pub.publish(vis_msg)
                
                # Update statistics
                self.processed_count += 1
                if self.processed_count % 10 == 0:
                    elapsed = time.time() - self.start_time
                    fps = self.processed_count / elapsed
                    avg_processing_time = np.mean(self.processing_times[-10:])
                    
                    self.get_logger().info(
                        f'Processing at {fps:.1f} fps | '
                        f'Avg time: {avg_processing_time:.2f}s | '
                        f'Target size: {self.target_size[0]}x{self.target_size[1]}'
                    )
        
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')
    
    def create_visualization(self, image, depth_map):
        """Create visualization combining original image and depth map"""
        # Normalize depth for display
        depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_PLASMA)
        
        # Resize for display
        display_size = (640, 480)
        image_display = cv2.resize(image, display_size)
        depth_display = cv2.resize(depth_colored, display_size)
        
        # Combine images
        combined = np.hstack([image_display, depth_display])
        
        # Add labels
        cv2.putText(combined, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined, "Depth Map", (650, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add depth statistics
        stats = f"Min: {depth_map.min():.2f}m, Max: {depth_map.max():.2f}m"
        cv2.putText(combined, stats, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        mean_depth = f"Mean: {depth_map.mean():.2f}m"
        cv2.putText(combined, mean_depth, (10, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add optimization info
        opt_info = f"Size: {self.target_size[0]}x{self.target_size[1]} | Skip: {self.skip_frames}"
        cv2.putText(combined, opt_info, (10, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return combined


def main(args=None):
    rclpy.init(args=args)
    
    # Create node with optimized parameters
    node = DepthEstimationNode(
        target_size=(64, 64),      # Smaller = faster
        use_fast_processor=True,
        skip_frames=2               # Process every 2nd frame
    )
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
