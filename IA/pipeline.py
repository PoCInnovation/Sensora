#!/usr/bin/env python3
"""
Real-time Depth + Object Detection Pipeline for RTX 4060
Outputs: 36-patch depth grid + object detections
Ready to connect to 6x6 tactile array
"""

import torch
import cv2
import numpy as np
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from ultralytics import YOLO
from PIL import Image
import time
from collections import deque
import sys

class TactileVisionPipeline:
    def __init__(self, depth_model_id="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Pipeline] Using device: {self.device}")
        
        # Load models
        print("[Pipeline] Loading Depth Anything V2-Small...")
        self.depth_processor = AutoImageProcessor.from_pretrained(depth_model_id)
        self.depth_model = AutoModelForDepthEstimation.from_pretrained(depth_model_id).to(self.device)
        self.depth_model.eval()
        
        print("[Pipeline] Loading YOLOv10-Nano...")
        self.detection_model = YOLO('yolov10n.pt')

        # Performance tracking
        self.frame_times = deque(maxlen=30)  # Rolling average of last 30 frames

        if torch.cuda.is_available():
            print(f"[Pipeline] Memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        else:
            print("[Pipeline] Running on CPU (GPU not available)")
    
    def divide_into_patches(self, array, grid_size=6):
        """Divide image into 6x6 grid and return mean depth per patch"""
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
                patches[i, j] = np.mean(patch) if patch.size > 0 else 0
        
        return patches
    
    def process_frame(self, frame):
        """Process single frame: depth + detection"""
        start = time.time()
        
        # Convert BGR to RGB for models
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # --- DEPTH ESTIMATION ---
        with torch.no_grad():
            inputs = self.depth_processor(images=pil_image, return_tensors="pt").to(self.device)
            outputs = self.depth_model(**inputs)
            depth_map = outputs.predicted_depth.cpu().numpy()[0]  # Shape: [H, W]
        
        # Normalize depth to 0-1 range
        depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-5)
        
        # Divide into 6x6 patches
        depth_patches = self.divide_into_patches(depth_normalized, grid_size=6)
        
        # --- OBJECT DETECTION ---
        device = 0 if torch.cuda.is_available() else 'cpu'
        results = self.detection_model.predict(source=pil_image, device=device, verbose=False, conf=0.5)
        detections = []
        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            
            for box, conf, cls_id in zip(boxes, confs, classes):
                x1, y1, x2, y2 = box
                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'confidence': float(conf),
                    'class_id': int(cls_id),
                    'center': ((x1 + x2) / 2, (y1 + y2) / 2)
                })
        
        end = time.time()
        frame_time = end - start
        self.frame_times.append(frame_time)
        
        return {
            'depth_patches': depth_patches,
            'detections': detections,
            'frame_time_ms': frame_time * 1000,
            'fps': 1 / frame_time if frame_time > 0 else 0
        }
    
    def print_stats(self):
        if self.frame_times:
            avg_time = np.mean(self.frame_times)
            print(f"[Stats] Avg latency: {avg_time*1000:.1f} ms | FPS: {1/avg_time:.1f}")

def main():
    print("=" * 60)
    print("Tactile Vision Pipeline - Depth + Detection")
    print("=" * 60)
    print()
    
    # Initialize pipeline
    pipeline = TactileVisionPipeline()
    print()
    
    # Open camera
    print("[Camera] Opening webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[ERROR] Could not open camera")
        print("[HINT] Try: cap = cv2.VideoCapture(1)  # Use different camera")
        sys.exit(1)
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("[Camera] Ready. Press 'q' to quit")
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
            print(f"[Frame {frame_count}] Latency: {result['frame_time_ms']:.1f}ms | "
                  f"FPS: {result['fps']:.1f} | "
                  f"Detections: {len(result['detections'])} | "
                  f"Depth range: {result['depth_patches'].min():.2f}-{result['depth_patches'].max():.2f}")
        
        # Visualize depth map
        depth_visual = (result['depth_patches'] * 255).astype(np.uint8)
        depth_visual = cv2.resize(depth_visual, (384, 384), interpolation=cv2.INTER_NEAREST)
        depth_visual = cv2.applyColorMap(depth_visual, cv2.COLORMAP_VIRIDIS)
        
        # Draw detections on original frame
        for det in result['detections']:
            x1, y1, x2, y2 = [int(v) for v in det['bbox']]
            conf = det['confidence']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{conf:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
        
        # Display
        cv2.imshow("Camera Feed", cv2.resize(frame, (480, 360)))
        cv2.imshow("Depth Map (6x6)", depth_visual)
        
        # Check for quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n[Pipeline] Shutting down...")
            break
    
    # Cleanup
    pipeline.print_stats()
    cap.release()
    cv2.destroyAllWindows()
    print("[Done]")

if __name__ == "__main__":
    main()
