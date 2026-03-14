#!/usr/bin/env python3
"""
Real-time Depth + Object Detection Pipeline with Semantic Segmentation
Automatically detects sidewalk/ground areas and processes only those regions
Useful for blind navigation - focuses on walking surface and obstacles on it
"""

import torch
import cv2
import numpy as np
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from transformers import pipeline as hf_pipeline
from ultralytics import YOLO
from PIL import Image
import time
from collections import deque
import sys

class SmartTactileVisionPipeline:
    def __init__(self, depth_model_id="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Pipeline] Using device: {self.device}")
        
        # Load depth model
        print("[Pipeline] Loading Depth Anything V2-Small...")
        self.depth_processor = AutoImageProcessor.from_pretrained(depth_model_id)
        self.depth_model = AutoModelForDepthEstimation.from_pretrained(depth_model_id).to(self.device)
        self.depth_model.eval()
        
        # Load semantic segmentation for ground/sidewalk detection
        print("[Pipeline] Loading semantic segmentation model...")
        print("  (First time download: ~800MB, patience...)")
        try:
            # SegFormer is fast and accurate
            self.segmenter = hf_pipeline(
                "image-segmentation",
                model="nvidia/segformer-b0-finetuned-ade-512-512",
                device=0 if torch.cuda.is_available() else -1
            )
            print("[Pipeline] ✓ Segmentation model loaded")
        except Exception as e:
            print(f"[WARNING] Segmentation model failed to load: {e}")
            print("[WARNING] Will use depth-based ground detection instead")
            self.segmenter = None
        
        # Load object detection
        print("[Pipeline] Loading YOLOv10-Nano...")
        self.detection_model = YOLO('yolov10n.pt')
        
        # Performance tracking
        self.frame_times = deque(maxlen=30)
        
        print(f"[Pipeline] Memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        print("[Pipeline] Ready to process frames")
        print()
    
    def get_ground_mask_semantic(self, frame):
        """
        Use semantic segmentation to identify ground/sidewalk/road areas
        Returns binary mask where 1 = ground/sidewalk, 0 = not ground
        """
        try:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            # Segment the image
            results = self.segmenter(pil_image)
            
            # Create mask for ground-like categories
            mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            
            for result in results:
                label = result['label'].lower()
                
                # Keywords that indicate ground/sidewalk/walking surface
                ground_keywords = [
                    'ground', 'sidewalk', 'pavement', 'road', 'street',
                    'path', 'walkway', 'concrete', 'asphalt', 'floor',
                    'terrain', 'grass', 'dirt', 'earth', 'surface',
                    'runway', 'parking', 'driveway', 'plaza'
                ]
                
                is_ground = any(keyword in label for keyword in ground_keywords)
                
                if is_ground:
                    # Get the segmentation mask for this category
                    seg_mask = np.array(result['mask'])
                    mask = np.maximum(mask, seg_mask.astype(np.uint8) * 255)
            
            return mask
        except Exception as e:
            print(f"[WARNING] Segmentation failed: {e}")
            return None
    
    def get_ground_mask_depth(self, depth_map):
        """
        Fallback: use depth to estimate ground plane
        Ground is typically at consistent depth with low variance
        """
        # Find the most common depth range (ground plane)
        hist, bins = np.histogram(depth_map.flatten(), bins=50)
        
        # Most common depth = ground
        ground_depth_bin = np.argmax(hist)
        ground_depth = (bins[ground_depth_bin] + bins[ground_depth_bin + 1]) / 2
        
        # Allow ±20% variance
        tolerance = (depth_map.max() - depth_map.min()) * 0.2
        
        # Mask: keep only pixels close to ground depth
        mask = np.abs(depth_map - ground_depth) < tolerance
        return (mask * 255).astype(np.uint8)
    
    def divide_into_patches(self, array, grid_size=6):
        """Divide image into 6x6 grid and return mean depth per patch"""
        if array.size == 0:
            return np.zeros((grid_size, grid_size))
        
        h, w = array.shape[:2]
        patch_height = h // grid_size
        patch_width = w // grid_size
        
        patches = np.zeros((grid_size, grid_size))
        
        for i in range(grid_size):
            for j in range(grid_size):
                y_start = i * patch_height
                y_end = (i + 1) * patch_height if i < grid_size - 1 else h
                x_start = j * patch_width
                x_end = (j + 1) * patch_width if j < grid_size - 1 else w
                
                patch = array[y_start:y_end, x_start:x_end]
                if patch.size > 0:
                    patches[i, j] = np.mean(patch)
        
        return patches
    
    def process_frame(self, frame):
        """Process frame: segment ground -> crop to ground -> compute 6x6 patches"""
        start = time.time()
        
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # --- SEMANTIC SEGMENTATION: Detect sidewalk/ground ---
        if self.segmenter is not None:
            ground_mask = self.get_ground_mask_semantic(frame)
            segmentation_time = time.time() - start
        else:
            ground_mask = None
            segmentation_time = 0
        
        # --- DEPTH ESTIMATION ---
        depth_start = time.time()
        with torch.no_grad():
            inputs = self.depth_processor(images=pil_image, return_tensors="pt").to(self.device)
            outputs = self.depth_model(**inputs)
            depth_map = outputs.predicted_depth.cpu().numpy()[0]  # [H, W]
        
        # Normalize depth
        depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-5)
        
        # If semantic segmentation didn't work, fall back to depth-based detection
        if ground_mask is None:
            ground_mask = self.get_ground_mask_depth(depth_normalized)
        
        # Resize mask to match depth map
        if ground_mask.shape != depth_normalized.shape:
            ground_mask = cv2.resize(
                ground_mask, 
                (depth_normalized.shape[1], depth_normalized.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
        
        # CROP to ground region (bounding box of non-zero mask)
        # This gives us the smallest rectangle containing all ground pixels
        ground_indices = np.where(ground_mask > 128)
        if len(ground_indices[0]) > 0:
            y_min, y_max = ground_indices[0].min(), ground_indices[0].max()
            x_min, x_max = ground_indices[1].min(), ground_indices[1].max()
            
            # Crop depth to ground region only
            depth_cropped = depth_normalized[y_min:y_max+1, x_min:x_max+1]
            ground_mask_cropped = ground_mask[y_min:y_max+1, x_min:x_max+1]
            
            # Apply mask within cropped region (to clean up edges)
            depth_cropped = depth_cropped * (ground_mask_cropped.astype(np.float32) / 255.0)
        else:
            # Fallback if no ground found
            depth_cropped = depth_normalized
            ground_mask_cropped = ground_mask
        
        # Divide the CROPPED ground region into 6x6 patches
        # The full 6x6 grid now covers just the sidewalk area
        depth_patches = self.divide_into_patches(depth_cropped, grid_size=6)
        
        # --- OBJECT DETECTION ---
        results = self.detection_model.predict(source=pil_image, device=0, verbose=False, conf=0.5)
        detections = []
        
        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            
            for box, conf, cls_id in zip(boxes, confs, classes):
                x1, y1, x2, y2 = box
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                
                # Check if detection is on ground (within segmented area)
                cx_norm = int(cx / frame.shape[1] * ground_mask.shape[1])
                cy_norm = int(cy / frame.shape[0] * ground_mask.shape[0])
                
                if (0 <= cy_norm < ground_mask.shape[0] and 
                    0 <= cx_norm < ground_mask.shape[1] and
                    ground_mask[cy_norm, cx_norm] > 128):  # On ground
                    
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': float(conf),
                        'class_id': int(cls_id),
                        'center': (cx, cy),
                        'on_ground': True
                    })
                else:
                    # Detection off ground (in sky, etc) - still include but mark
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': float(conf),
                        'class_id': int(cls_id),
                        'center': (cx, cy),
                        'on_ground': False
                    })
        
        end = time.time()
        frame_time = end - start
        self.frame_times.append(frame_time)
        
        return {
            'depth_patches': depth_patches,
            'detections': detections,
            'ground_mask': ground_mask,
            'depth_cropped': depth_cropped,  # Cropped depth region
            'frame_time_ms': frame_time * 1000,
            'fps': 1 / frame_time if frame_time > 0 else 0,
            'segmentation_time_ms': segmentation_time * 1000
        }
    
    def print_stats(self):
        if self.frame_times:
            avg_time = np.mean(self.frame_times)
            print(f"[Stats] Avg latency: {avg_time*1000:.1f} ms | FPS: {1/avg_time:.1f}")

def main():
    print("=" * 70)
    print("Smart Tactile Vision Pipeline - Semantic Segmentation + Depth")
    print("=" * 70)
    print()
    
    # Initialize pipeline
    pipeline = SmartTactileVisionPipeline()
    
    # Open camera
    print("[Camera] Opening webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[ERROR] Could not open camera")
        sys.exit(1)
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("[Camera] Ready. Press 'q' to quit, 's' to save debug image")
    print()
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame")
            break
        
        # Process frame
        result = pipeline.process_frame(frame)
        frame_count += 1
        
        # Print every 10 frames
        if frame_count % 10 == 0:
            ground_pixels = np.count_nonzero(result['ground_mask']) if result['ground_mask'] is not None else 0
            ground_percent = (ground_pixels / (result['ground_mask'].size)) * 100 if result['ground_mask'] is not None else 0
            
            on_ground_detections = sum(1 for d in result['detections'] if d.get('on_ground', False))
            
            print(f"[Frame {frame_count}] "
                  f"Latency: {result['frame_time_ms']:.1f}ms | "
                  f"FPS: {result['fps']:.1f} | "
                  f"Ground: {ground_percent:.1f}% | "
                  f"Detections (on ground): {on_ground_detections}/{len(result['detections'])}")
        
        # Visualize depth patches
        depth_visual = (result['depth_patches'] * 255).astype(np.uint8)
        depth_visual = cv2.resize(depth_visual, (384, 384), interpolation=cv2.INTER_NEAREST)
        depth_visual = cv2.applyColorMap(depth_visual, cv2.COLORMAP_VIRIDIS)
        
        # Visualize ground mask
        ground_visual = cv2.resize(result['ground_mask'], (384, 384)) if result['ground_mask'] is not None else np.zeros((384, 384), dtype=np.uint8)
        ground_visual = cv2.applyColorMap(ground_visual, cv2.COLORMAP_BONE)
        
        # Draw detections on original frame
        for det in result['detections']:
            x1, y1, x2, y2 = [int(v) for v in det['bbox']]
            conf = det['confidence']
            on_ground = det.get('on_ground', False)
            
            # Green if on ground, red if not
            color = (0, 255, 0) if on_ground else (0, 0, 255)
            thickness = 2 if on_ground else 1
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(frame, f"{conf:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)
        
        # Display
        cv2.imshow("Camera Feed", cv2.resize(frame, (480, 360)))
        cv2.imshow("Depth Patches (6x6)", depth_visual)
        cv2.imshow("Ground Mask", ground_visual)
        
        # Check for input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n[Pipeline] Shutting down...")
            break
        elif key == ord('s'):
            # Save debug images
            cv2.imwrite(f'frame_{frame_count}_camera.jpg', frame)
            cv2.imwrite(f'frame_{frame_count}_depth.jpg', depth_visual)
            cv2.imwrite(f'frame_{frame_count}_mask.jpg', ground_visual)
            print(f"[Saved] Debug images for frame {frame_count}")
    
    # Cleanup
    pipeline.print_stats()
    cap.release()
    cv2.destroyAllWindows()
    print("[Done]")

if __name__ == "__main__":
    main()
