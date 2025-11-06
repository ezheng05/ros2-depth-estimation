#!/usr/bin/env python3
# 3D depth estimation - main
# absolute depth with ZoeDepth

import cv2
import numpy as np
import torch
import time
import argparse
import os
from typing import Optional, Tuple, Union
from pathlib import Path

from transformers import AutoImageProcessor, AutoModelForDepthEstimation, infer_device
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class DepthEstimationSystem:
    
    def __init__(self, model_name: str = "Intel/zoedepth-nyu-kitti", 
                 target_size: Optional[Tuple[int, int]] = None,
                 use_fast_processor: bool = True):
        # initialize system
        # args: model name (hugging face model identifier), target processing size, use fast processor

        self.device = infer_device()
        self.model_name = model_name
        self.target_size = target_size  # (width, height) for faster processing
        self.use_fast_processor = use_fast_processor
        self.model: Optional[AutoModelForDepthEstimation] = None
        self.processor: Optional[AutoImageProcessor] = None
        self.camera: Optional[cv2.VideoCapture] = None
        self.focal_length = 500  # default focal length in pixels
        
        print(f"Device: {self.device}")
        print(f"Model: {model_name}")
        if target_size:
            print(f"target processing size: {target_size[0]}x{target_size[1]}")
        else:
            print("processing at original resolution")
    
    def load_model(self) -> bool:
        # load depth estimation model and processor
        # return: True if model loaded successfully, False otherwise
        
        print("loading depth model")
        try:
            # use fast processor if available
            processor_kwargs = {}
            if self.use_fast_processor:
                processor_kwargs['use_fast'] = True
            
            self.processor = AutoImageProcessor.from_pretrained(
                self.model_name, **processor_kwargs
            )
            self.model = AutoModelForDepthEstimation.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            print("model loaded successfully!")
            return True
        except Exception as e:
            print(f"failed to load model: {e}")
            return False
    
    def estimate_depth(self, image: Union[np.ndarray, Image.Image]) -> Optional[np.ndarray]:
        # estimate absolute depth for an image
        # args: input image as numpy array or PIL
        # return: depth map as numpy array, or None if estimation failed

        try:
            # convert to PIL image if needed
            if isinstance(image, np.ndarray):
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(image_rgb)
                else:
                    pil_image = Image.fromarray(image)
            else:
                pil_image = image
            
            # resize to target size for faster processing if specified
            if self.target_size:
                pil_image = pil_image.resize(self.target_size, Image.Resampling.LANCZOS)
            
            # prepare inputs
            inputs = self.processor(pil_image, return_tensors="pt")
            pixel_values = inputs.pixel_values.to(self.device)
            
            # run inference
            with torch.no_grad():
                outputs = self.model(pixel_values)
            
            # post-process results
            post_processed = self.processor.post_process_depth_estimation(
                outputs,
                source_sizes=[(pil_image.height, pil_image.width)],
            )
            
            depth_map = post_processed[0]["predicted_depth"]
            return depth_map.squeeze().cpu().numpy()
            
        except Exception as e:
            print(f"depth estimation error: {e}")
            return None
    
    def depth_to_3d_points(self, depth_map: np.ndarray, image: Union[np.ndarray, Image.Image], 
                          focal_length: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        # convert depth map to 3D points with absolute coordinates
        # args: depth map as numpy array, original image for color information, camera focal length
        # return: tuple of (3D points, colors) as numpy arrays
        
        if focal_length is None:
            focal_length = self.focal_length
        
        height, width = depth_map.shape
        
        # create coordinate grids
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        
        # convert to 3D coordinates using pinhole camera model
        cx, cy = width // 2, height // 2  # principal point
        
        z = depth_map
        x = (u - cx) * z / focal_length
        y = (v - cy) * z / focal_length
        
        # stack coordinates
        points_3d = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
        
        # get colors from image - resize to match depth map size if needed
        if isinstance(image, np.ndarray):
            # resize image to match depth map size if target size was used
            if self.target_size:
                image_resized = cv2.resize(image, (width, height))
                if len(image_resized.shape) == 3:
                    colors = image_resized.reshape(-1, 3) / 255.0
                else:
                    colors = np.tile(image_resized.reshape(-1, 1), (1, 3)) / 255.0
            else:
                if len(image.shape) == 3:
                    colors = image.reshape(-1, 3) / 255.0
                else:
                    colors = np.tile(image.reshape(-1, 1), (1, 3)) / 255.0
        else:
            image_array = np.array(image)
            # resize image to match depth map size if target size was used
            if self.target_size:
                image_resized = cv2.resize(image_array, (width, height))
                if len(image_resized.shape) == 3:
                    colors = image_resized.reshape(-1, 3) / 255.0
                else:
                    colors = np.tile(image_resized.reshape(-1, 1), (1, 3)) / 255.0
            else:
                if len(image_array.shape) == 3:
                    colors = image_array.reshape(-1, 3) / 255.0
                else:
                    colors = np.tile(image_array.reshape(-1, 1), (1, 3)) / 255.0
        
        # remove invalid points (keep only reasonable depth values)
        valid_mask = (np.isfinite(points_3d).all(axis=1) & 
                     (z.flatten() > 0) & 
                     (z.flatten() < 20.0))  # reasonable depth range
        
        points_3d = points_3d[valid_mask]
        colors = colors[valid_mask]
        
        return points_3d, colors
    
    def visualize_2d(self, image: np.ndarray, depth_map: np.ndarray) -> np.ndarray:
        # create comprehensive 2D visualization
        # args: original image, depth map
        # return: combined visualization image
        
        # normalize depth for display
        depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_PLASMA)
        
        # resize
        display_size = (640, 480)
        image_display = cv2.resize(image, display_size)
        depth_display = cv2.resize(depth_colored, display_size)
        
        # combine images
        combined = np.hstack([image_display, depth_display])
        
        # add labels and statistics
        cv2.putText(combined, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined, "Absolute Depth", (650, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # add depth statistics
        stats = f"Min: {depth_map.min():.2f}m, Max: {depth_map.max():.2f}m"
        cv2.putText(combined, stats, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        mean_depth = f"Mean: {depth_map.mean():.2f}m"
        cv2.putText(combined, mean_depth, (10, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("3D Absolute Depth Estimation", combined)
        return combined
    
    def visualize_3d(self, points: np.ndarray, colors: np.ndarray, 
                    title: str = "3D Point Cloud (Absolute Depth)") -> None:
        # create 3D visualization with matplotlib
        # args: 3D points array, color array for points, plot title

        if len(points) == 0:
            print("no valid 3D points to visualize")
            return
        
        # downsample for performance
        if len(points) > 10000:
            indices = np.random.choice(len(points), 10000, replace=False)
            points_vis = points[indices]
            colors_vis = colors[indices]
        else:
            points_vis = points
            colors_vis = colors
        
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # create 3D scatter plot
        scatter = ax.scatter(points_vis[:, 0], points_vis[:, 1], points_vis[:, 2], 
                           c=colors_vis, s=1, alpha=0.6)
        
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_zlabel('Z (meters)')
        ax.set_title(title)
        
        # add statistics text
        stats_text = f"""
        Points: {len(points):,}
        X range: {points[:, 0].min():.2f} to {points[:, 0].max():.2f}m
        Y range: {points[:, 1].min():.2f} to {points[:, 1].max():.2f}m
        Z range: {points[:, 2].min():.2f} to {points[:, 2].max():.2f}m
        """
        
        plt.figtext(0.02, 0.02, stats_text, fontsize=10, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    
    def save_point_cloud_ply(self, points: np.ndarray, colors: np.ndarray, 
                           filename: str = "point_cloud.ply") -> bool:
        # save point cloud in PLY format
        # args: 3D points array, color array, output filename
        # return: True if saved successfully, False otherwise
        
        if len(points) == 0:
            print("no points to save")
            return False
        
        try:
            # create PLY header
            header = f"""ply
format ascii 1.0
element vertex {len(points)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
            
            # write PLY file
            with open(filename, 'w') as f:
                f.write(header)
                for i in range(len(points)):
                    x, y, z = points[i]
                    r, g, b = (colors[i] * 255).astype(int)
                    f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")
            
            print(f"point cloud saved to {filename}")
            return True
        except Exception as e:
            print(f"save failed: {e}")
            return False
    
    def run_camera_demo(self) -> None:
        # run real-time demo
        print("\ncamera demo - absolute depth estimation")
        print("=" * 50)
        
        if not self.load_model():
            return
        
        # Initialize camera
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            print("could not open camera")
            return
        
        print("camera opened")
        print("\nControls:")
        print("  - 'q' to quit")
        print("  - 's' to save current 3D point cloud (PLY)")
        print("  - '3' to show 3D visualization")
        print("  - 't' to save as text file")
        
        try:
            while True:
                ret, frame = self.camera.read()
                if not ret:
                    break
                
                # estimate depth
                depth_map = self.estimate_depth(frame)
                if depth_map is not None:
                    # show 2D visualization
                    self.visualize_2d(frame, depth_map)
                    
                    # generate 3D points
                    points_3d, colors = self.depth_to_3d_points(depth_map, frame)
                    
                    # handle keyboard input
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s') and len(points_3d) > 0:
                        filename = f"point_cloud_{int(time.time())}.ply"
                        self.save_point_cloud_ply(points_3d, colors, filename)
                    elif key == ord('3') and len(points_3d) > 0:
                        self.visualize_3d(points_3d, colors)
                    elif key == ord('t') and len(points_3d) > 0:
                        filename = f"point_cloud_{int(time.time())}.txt"
                        np.savetxt(filename, points_3d, header="X Y Z", comments="")
                        print(f"point cloud saved to {filename}")
        
        except KeyboardInterrupt:
            print("\ninterrupted by user")
        
        finally:
            self.camera.release()
            cv2.destroyAllWindows()
    
    def run_image_demo(self, image_path: str) -> None:
        # run single image demo
        # args: path to input image

        print(f"\nimage demo: {image_path}")
        print("=" * 50)
        
        if not self.load_model():
            return
        
        # load image
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"could not load image: {image_path}")
                return
            print(f"image loaded: {image.shape}")
        except Exception as e:
            print(f"error loading image: {e}")
            return
        
        # estimate depth
        print("estimating absolute depth...")
        depth_map = self.estimate_depth(image)
        if depth_map is None:
            print("depth estimation failed")
            return
        
        print(f"depth estimated: {depth_map.shape}")
        print(f"depth range: {depth_map.min():.3f}m to {depth_map.max():.3f}m")
        print(f"mean depth: {depth_map.mean():.3f}m")
        
        # generate 3D points
        print("generating 3D points...")
        points_3d, colors = self.depth_to_3d_points(depth_map, image)
        print(f"generated {len(points_3d):,} 3D points")
        
        # show 2D visualization
        print("showing 2D visualization...")
        combined = self.visualize_2d(image, depth_map)
        cv2.waitKey(0)
        
        # show 3D visualization
        print("showing 3D visualization...")
        self.visualize_3d(points_3d, colors)
        
        # save point cloud
        base_name = Path(image_path).stem
        ply_filename = f"point_cloud_{base_name}.ply"
        txt_filename = f"point_cloud_{base_name}.txt"
        
        self.save_point_cloud_ply(points_3d, colors, ply_filename)
        np.savetxt(txt_filename, points_3d, header="X Y Z", comments="")
        print(f"text file saved to {txt_filename}")
        
        cv2.destroyAllWindows()


def main():
    # main application entry point
    parser = argparse.ArgumentParser(
        description="3D depth estimation"
    )
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--camera", action="store_true", help="Use camera instead of image")
    parser.add_argument("--model", type=str, default="Intel/zoedepth-nyu-kitti", 
                       help="Depth estimation model")
    parser.add_argument("--size", type=str, default="128x128", 
                       help="Target processing size for speed (e.g., 64x64, 128x128, 256x256)")
    parser.add_argument("--no-fast", action="store_true", 
                       help="disable fast image processor")
    
    args = parser.parse_args()
    
    # Parse target size
    try:
        width, height = map(int, args.size.split('x'))
        target_size = (width, height)
    except ValueError:
        print(f"Invalid size format: {args.size}. Use format like '256x256'")
        return
    
    # create system instance
    system = DepthEstimationSystem(
        model_name=args.model,
        target_size=target_size,
        use_fast_processor=not args.no_fast
    )
    
    if args.camera:
        system.run_camera_demo()
    elif args.image:
        if os.path.exists(args.image):
            system.run_image_demo(args.image)
        else:
            print(f"image not found: {args.image}")
    else:
        print("Usage:")
        print("  python main_3d_system.py --camera    # use camera")
        print("  python main_3d_system.py --image photo.jpg    # use image file")
        print("  python main_3d_system.py --image photo.jpg --size 64x64    # fast processing")
        print()
        print("Examples:")
        print("  python main_3d_system.py --camera --size 256x256    # fast camera demo")
        print("  python main_3d_system.py --image my_photo.jpg --size 128x128")


if __name__ == "__main__":
    main()
