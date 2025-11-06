#!/usr/bin/env python3
"""
ROS 2 RViz Marker Node
Publishes 3D markers (arrow + sphere) for visualization in RViz
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, TransformStamped
from visualization_msgs.msg import Marker
from tf2_ros import TransformBroadcaster
import math


class RVizMarkerNode(Node):
    """Publishes RViz markers for closest point visualization"""
    
    def __init__(self):
        super().__init__('rviz_marker_node')
        
        # TF broadcaster for camera frame
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Subscribe to smoothed 3D point
        self.point_sub = self.create_subscription(
            PointStamped,
            '/direction/closest_point_3d_smoothed',
            self.point_callback,
            10
        )
        
        # Publish markers
        self.marker_pub = self.create_publisher(
            Marker,
            '/visualization_marker',
            10
        )
        
        # Timer to publish camera TF
        self.tf_timer = self.create_timer(0.1, self.publish_tf)
        
        self.get_logger().info('RViz Marker Node initialized')
        self.get_logger().info('Publishing to /visualization_marker')
        self.get_logger().info('Broadcasting TF: world -> camera')
    
    def publish_tf(self):
        """Publish static transform from world to camera"""
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id = 'camera'
        
        # Camera at origin, pointing forward (Z-axis)
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        
        # No rotation
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0
        
        self.tf_broadcaster.sendTransform(t)
    
    def point_callback(self, msg):
        """Receive 3D point and publish markers"""
        # Extract 3D point
        x = msg.point.x
        y = msg.point.y
        z = msg.point.z
        
        stamp = msg.header.stamp
        
        # Publish arrow from origin to point
        self.publish_arrow(x, y, z, stamp)
        
        # Publish sphere at point location
        self.publish_sphere(x, y, z, stamp)
    
    def publish_arrow(self, x, y, z, stamp):
        """Publish arrow marker from camera to closest point"""
        marker = Marker()
        marker.header.frame_id = 'camera'
        marker.header.stamp = stamp
        marker.ns = 'direction_arrow'
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        
        # Arrow starts at origin (0,0,0)
        marker.points = []
        start = marker.points
        
        # Use pose instead of points for better control
        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.0
        
        # Calculate orientation to point toward (x,y,z)
        # Arrow points along X-axis by default, so rotate to point at target
        distance = math.sqrt(x*x + y*y + z*z)
        
        if distance > 0.001:
            # Normalize
            nx, ny, nz = x/distance, y/distance, z/distance
            
            # Calculate quaternion to rotate X-axis to point at (nx,ny,nz)
            # Using axis-angle representation
            # Default is (1,0,0), target is (nx,ny,nz)
            # Rotation axis is cross product
            ax = 0.0
            ay = nz
            az = -ny
            angle = math.acos(max(-1.0, min(1.0, nx)))
            
            # Convert axis-angle to quaternion
            if abs(angle) > 0.001:
                s = math.sin(angle / 2.0)
                norm = math.sqrt(ay*ay + az*az)
                if norm > 0.001:
                    ay, az = ay/norm, az/norm
                    marker.pose.orientation.x = 0.0
                    marker.pose.orientation.y = ay * s
                    marker.pose.orientation.z = az * s
                    marker.pose.orientation.w = math.cos(angle / 2.0)
                else:
                    marker.pose.orientation.w = 1.0
            else:
                marker.pose.orientation.w = 1.0
        else:
            marker.pose.orientation.w = 1.0
        
        # Arrow scale (shaft diameter, head diameter, head length)
        marker.scale.x = distance  # Arrow length
        marker.scale.y = 0.02  # Shaft thickness
        marker.scale.z = 0.03  # Head thickness
        
        # Green arrow
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        
        marker.lifetime = rclpy.duration.Duration(seconds=0.5).to_msg()
        
        self.marker_pub.publish(marker)
    
    def publish_sphere(self, x, y, z, stamp):
        """Publish sphere marker at closest point"""
        marker = Marker()
        marker.header.frame_id = 'camera'
        marker.header.stamp = stamp
        marker.ns = 'closest_point_sphere'
        marker.id = 1
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        # Sphere position
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z
        marker.pose.orientation.w = 1.0
        
        # Sphere size (5cm diameter)
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        
        # Red sphere
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        
        marker.lifetime = rclpy.duration.Duration(seconds=0.5).to_msg()
        
        self.marker_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = RVizMarkerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
