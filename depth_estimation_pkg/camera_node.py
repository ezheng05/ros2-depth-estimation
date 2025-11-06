#!/usr/bin/env python3
"""
ROS 2 Camera Node - Real ROS 2 Implementation
Publishes camera images to /camera/image topic
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import time


class CameraNode(Node):
    """ROS 2 node for camera input"""
    
    def __init__(self, camera_id=0, fps=30):
        super().__init__('camera_node')
        
        # Declare parameters
        self.declare_parameter('camera_id', camera_id)
        self.declare_parameter('fps', fps)
        
        # Get parameters
        self.camera_id = self.get_parameter('camera_id').value
        self.fps = self.get_parameter('fps').value
        self.frame_time = 1.0 / self.fps
        
        # Initialize camera
        self.camera = cv2.VideoCapture(self.camera_id)
        if not self.camera.isOpened():
            self.get_logger().error(f'Could not open camera {self.camera_id}')
            raise RuntimeError(f'Could not open camera {self.camera_id}')
        
        # Set camera properties
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera.set(cv2.CAP_PROP_FPS, self.fps)
        
        # Create CV Bridge for converting OpenCV images to ROS messages
        self.bridge = CvBridge()
        
        # Create publisher
        self.image_pub = self.create_publisher(Image, '/camera/image', 10)
        
        # Create timer for publishing at specified FPS
        self.timer = self.create_timer(self.frame_time, self.timer_callback)
        
        # Stats
        self.frame_count = 0
        self.start_time = time.time()
        
        self.get_logger().info(f'Camera node initialized with camera {self.camera_id} at {self.fps} fps')
        self.get_logger().info('Publishing to: /camera/image')
    
    def timer_callback(self):
        """Timer callback to capture and publish images"""
        ret, frame = self.camera.read()
        
        if ret:
            # Convert OpenCV image to ROS Image message
            try:
                img_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
                img_msg.header.stamp = self.get_clock().now().to_msg()
                img_msg.header.frame_id = 'camera'
                
                # Publish image
                self.image_pub.publish(img_msg)
                
                # Update statistics
                self.frame_count += 1
                if self.frame_count % 30 == 0:
                    elapsed = time.time() - self.start_time
                    actual_fps = self.frame_count / elapsed
                    self.get_logger().info(f'Publishing at {actual_fps:.1f} fps')
            
            except Exception as e:
                self.get_logger().error(f'Error converting image: {e}')
        else:
            self.get_logger().warn('Failed to capture frame from camera')
    
    def destroy_node(self):
        """Cleanup when node is destroyed"""
        if self.camera.isOpened():
            self.camera.release()
        self.get_logger().info('Camera node cleanup complete')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    # Create node with default parameters
    node = CameraNode(camera_id=0, fps=30)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
