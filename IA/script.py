import cv2
import torch
import threading
import numpy as np
from transformers import AutoModelForCausalLM, AutoProcessor
from ultralytics import YOLO
from collections import deque
import time
from PIL import Image

# --- CONFIGURATION ---
DEVICE = "cpu"
MODEL_NAME = "yolo11n.pt"
CONF_THRESHOLD = 0.4
IOU_THRESHOLD = 0.45
DETECTION_FPS = 15
DEPTH_FPS = 10
VLM_FPS = 2
vlm_model_name = "apple/FastVLM-0.5B"

# Load YOLO
model_detect = YOLO(MODEL_NAME)

# Load ZoeDepth or MiDaS
print("Loading depth model...")
try:
    model_depth = torch.hub.load("isl-org/ZoeDepth", "ZoeD_N", pretrained=True)
    model_depth.to(DEVICE).eval()
    print("✓ ZoeDepth loaded")
    USE_ZOEDEPTH = True
except Exception as e:
    print(f"ZoeDepth error: {e}")
    print("Using MiDaS...")
    model_depth = torch.hub.load("intel-isl/MiDaS", "DPT_Large", pretrained=True)
    model_depth.to(DEVICE).eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.dpt_transform
    USE_ZOEDEPTH = False
    print("✓ MiDaS DPT_Large loaded")

# Load FastVLM
print(f"Loading {vlm_model_name}...")
try:
    # Load processor (required for LlavaQwen2)
    vlm_processor = AutoProcessor.from_pretrained(vlm_model_name, trust_remote_code=True)
    
    # Load model
    vlm_model = AutoModelForCausalLM.from_pretrained(
        vlm_model_name, 
        trust_remote_code=True,
        dtype="auto"
    )
    vlm_model.to(DEVICE).eval()
    print("✓ FastVLM loaded successfully")
    USE_VLM = True
    VLM_TYPE = "fastvlm"
    
except Exception as e:
    print(f"FastVLM error: {e}")
    print("Trying BLIP (fallback)...")
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        
        vlm_model_name = "Salesforce/blip-image-captioning-base"
        print(f"Loading BLIP...")
        
        vlm_processor = BlipProcessor.from_pretrained(vlm_model_name)
        vlm_model = BlipForConditionalGeneration.from_pretrained(
            vlm_model_name,
            torch_dtype=torch.float32
        )
        vlm_model.to(DEVICE).eval()
        print("✓ BLIP loaded successfully (fallback)")
        USE_VLM = True
        VLM_TYPE = "blip"
        
    except Exception as e2:
        print(f"⚠ VLM not available: {e2}")
        print("System will run without descriptions (YOLO + Depth only)")
        USE_VLM = False
        VLM_TYPE = None

# Shared variables
frame_lock = threading.Lock()
latest_frame = None
detections = None
depth_map = None
running = True
vlm_description = ""
vlm_timestamp = 0

# FPS tracking
fps_queue = deque(maxlen=30)
last_frame_time = time.time()

def thread_detection():
    """YOLO detection thread"""
    global detections
    frame_interval = 1.0 / DETECTION_FPS
    
    while running:
        start_time = time.time()
        img_local = None
        
        with frame_lock:
            if latest_frame is not None:
                img_local = latest_frame.copy()
        
        if img_local is not None:
            img_rgb = cv2.cvtColor(img_local, cv2.COLOR_BGR2RGB)
            results = model_detect.predict(
                img_rgb,
                verbose=False,
                imgsz=320,
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                max_det=20,
                device=DEVICE
            )[0]
            detections = results
        
        elapsed = time.time() - start_time
        sleep_time = max(0, frame_interval - elapsed)
        time.sleep(sleep_time)

def thread_depth():
    """Depth estimation thread"""
    global depth_map
    frame_interval = 1.0 / DEPTH_FPS
    
    while running:
        start_time = time.time()
        img_local = None
        
        with frame_lock:
            if latest_frame is not None:
                img_local = latest_frame.copy()
        
        if img_local is not None:
            h_orig, w_orig = img_local.shape[:2]
            img_rgb = cv2.cvtColor(img_local, cv2.COLOR_BGR2RGB)
            
            with torch.no_grad():
                if USE_ZOEDEPTH:
                    pil_img = Image.fromarray(img_rgb)
                    depth = model_depth.infer_pil(pil_img)
                    
                    if isinstance(depth, torch.Tensor):
                        depth_np = depth.cpu().numpy()
                    else:
                        depth_np = depth
                    
                    if depth_np.shape[:2] != (h_orig, w_orig):
                        depth_np = cv2.resize(depth_np, (w_orig, h_orig))
                else:
                    img_input = cv2.resize(img_rgb, (384, 384))
                    input_batch = transform(img_input).to(DEVICE)
                    prediction = model_depth(input_batch)
                    prediction = torch.nn.functional.interpolate(
                        prediction.unsqueeze(1),
                        size=(h_orig, w_orig),
                        mode="bicubic",
                        align_corners=False,
                    ).squeeze()
                    depth_np = prediction.cpu().numpy()
                
                depth_min, depth_max = depth_np.min(), depth_np.max()
                if depth_max - depth_min > 1e-6:
                    depth_map = (depth_np - depth_min) / (depth_max - depth_min)
                else:
                    depth_map = depth_np
        
        elapsed = time.time() - start_time
        sleep_time = max(0, frame_interval - elapsed)
        time.sleep(sleep_time)

def thread_vlm():
    """Thread for continuous description generation with FastVLM"""
    global vlm_description, vlm_timestamp
    frame_interval = 1.0 / VLM_FPS
    
    while running:
        start_time = time.time()
        img_local = None
        detections_local = None
        
        with frame_lock:
            if latest_frame is not None:
                img_local = latest_frame.copy()
                detections_local = detections
        
        if img_local is not None and USE_VLM:
            try:
                pil_img = Image.fromarray(cv2.cvtColor(img_local, cv2.COLOR_BGR2RGB))
                
                # Get detected objects
                detected_objects = []
                if detections_local is not None and len(detections_local.boxes) > 0:
                    detected_objects = [detections_local.names[int(box.cls[0])] for box in detections_local.boxes]
                
                with torch.no_grad():
                    if VLM_TYPE == "fastvlm":
                        # FastVLM uses LlavaQwen2 format
                        if detected_objects:
                            objects = ", ".join(set(detected_objects[:5]))
                            prompt = f"<|im_start|>user\n<image>\nBriefly describe this scene with {objects}.<|im_end|>\n<|im_start|>assistant\n"
                        else:
                            prompt = "<|im_start|>user\n<image>\nBriefly describe what you see in this image.<|im_end|>\n<|im_start|>assistant\n"
                        
                        # Prepare inputs
                        inputs = vlm_processor(
                            text=prompt,
                            images=pil_img,
                            return_tensors="pt"
                        )
                        
                        # Move to device
                        inputs = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v 
                                 for k, v in inputs.items()}
                        
                        # Generate
                        output_ids = vlm_model.generate(
                            **inputs,
                            max_new_tokens=60,
                            do_sample=False
                        )
                        
                        # Decode only new tokens
                        input_len = inputs['input_ids'].shape[1]
                        response_ids = output_ids[0][input_len:]
                        description = vlm_processor.decode(response_ids, skip_special_tokens=True).strip()
                    
                    elif VLM_TYPE == "blip":
                        if detected_objects:
                            objects = ", ".join(set(detected_objects[:3]))
                            prompt = f"a photo of {objects}"
                            inputs = vlm_processor(pil_img, prompt, return_tensors="pt").to(DEVICE)
                        else:
                            inputs = vlm_processor(pil_img, return_tensors="pt").to(DEVICE)
                        
                        output = vlm_model.generate(**inputs, max_length=40)
                        description = vlm_processor.decode(output[0], skip_special_tokens=True)
                    
                    else:
                        description = "VLM not available"
                    
                    vlm_description = description
                    vlm_timestamp = time.time()
                    
            except Exception as e:
                print(f"VLM error: {e}")
                import traceback
                traceback.print_exc()
        
        elapsed = time.time() - start_time
        sleep_time = max(0, frame_interval - elapsed)
        time.sleep(sleep_time)

def calculate_depth_for_box(depth_map, box):
    """Calculate depth within a bounding box"""
    if depth_map is None:
        return None
    
    x1, y1, x2, y2 = map(int, box)
    h, w = depth_map.shape
    
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    if x2 <= x1 or y2 <= y1:
        return None
    
    region = depth_map[y1:y2, x1:x2]
    return np.median(region)

def draw_enhanced_detections(frame, detections, depth_map):
    """Draw enhanced detections"""
    display = frame.copy()
    
    if detections is not None and len(detections.boxes) > 0:
        boxes = detections.boxes
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = detections.names[cls]
            
            depth = calculate_depth_for_box(depth_map, box.xyxy[0])
            if depth is not None:
                distance_relative = 1.0 - depth
                depth_meters = distance_relative * distance_scale
                depth_str = f"{depth_meters:.1f}m"
            else:
                depth_str = "N/A"
            
            color = (0, 255, 0) if conf > 0.6 else (0, 255, 255)
            
            # Thicker rectangles
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 3)
            
            text = f"{label} {conf:.2f} | {depth_str}"
            # Larger text
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(display, (x1, y1 - text_h - 15), (x1 + text_w + 10, y1), color, -1)
            cv2.putText(display, text, (x1 + 5, y1 - 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    return display

def draw_info_panel(frame, fps):
    """Draw info panel"""
    h, w = frame.shape[:2]
    num_det = len(detections.boxes) if detections is not None else 0
    
    # Main panel - larger size
    overlay = frame.copy()
    panel_height = 250 if USE_VLM else 200
    cv2.rectangle(overlay, (10, 10), (450, panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    info_lines = [
        f"FPS: {fps:.1f}",
        f"Detections: {num_det}",
        f"Model: {MODEL_NAME}",
        f"Conf. min: {CONF_THRESHOLD}",
        f"Scale: {distance_scale:.1f}m"
    ]
    
    if USE_VLM:
        vlm_age = time.time() - vlm_timestamp if vlm_timestamp > 0 else 999
        vlm_status = f"Live ({vlm_age:.1f}s)" if vlm_age < 5 else "Waiting..."
        info_lines.append(f"VLM: {vlm_status}")
    
    for i, line in enumerate(info_lines):
        cv2.putText(frame, line, (25, 50 + i * 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # VLM description panel - larger
    if USE_VLM and vlm_description:
        desc = vlm_description
        age = time.time() - vlm_timestamp
        
        # Don't display if too old (more than 10 seconds)
        if age < 10:
            # Truncate if too long
            if len(desc) > 250:
                desc = desc[:247] + "..."
            
            # Panel for description - larger
            desc_height = 140
            cv2.rectangle(overlay, (10, h - desc_height - 10), (w - 10, h - 10), (0, 50, 0), -1)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
            
            vlm_icon = "🍎" if VLM_TYPE == "fastvlm" else "📷"
            model_label = "FastVLM LIVE" if VLM_TYPE == "fastvlm" else "BLIP LIVE"
            cv2.putText(frame, f"{vlm_icon} {model_label}:", (25, h - desc_height + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
            
            # Split description into lines
            words = desc.split()
            lines = []
            current_line = []
            for word in words:
                test_line = " ".join(current_line + [word])
                (tw, _), _ = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                if tw < w - 50:
                    current_line.append(word)
                else:
                    lines.append(" ".join(current_line))
                    current_line = [word]
            if current_line:
                lines.append(" ".join(current_line))
            
            for i, line in enumerate(lines[:3]):  # Max 3 lines
                cv2.putText(frame, line, (25, h - desc_height + 65 + i * 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

# --- INITIALIZATION ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

# Start threads
t_detect = threading.Thread(target=thread_detection, daemon=True)
t_depth = threading.Thread(target=thread_depth, daemon=True)
t_detect.start()
t_depth.start()

if USE_VLM:
    t_vlm = threading.Thread(target=thread_vlm, daemon=True)
    t_vlm.start()

print("=" * 50)
model_name = "ZoeDepth" if USE_ZOEDEPTH else "MiDaS DPT-Large"
if USE_VLM:
    if VLM_TYPE == "fastvlm":
        vlm_name = "🍎 Apple FastVLM-0.5B (LIVE)"
    elif VLM_TYPE == "blip":
        vlm_name = "BLIP (LIVE fallback)"
    else:
        vlm_name = "Unknown"
else:
    vlm_name = "None"
print(f"System: YOLO + {model_name} + {vlm_name}")
print("=" * 50)
print("Commands:")
print("  'q' : Quit")
print("  's' : Screenshot")
print("  'd' : Depth map")
print("  '+/-' : Adjust distance scale")
if USE_VLM:
    print(f"  FastVLM runs continuously (~{VLM_FPS} FPS)")
print("=" * 50)

screenshot_count = 0
show_depth_map = False
distance_scale = 15.0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read camera")
        break
    
    with frame_lock:
        latest_frame = frame
    
    current_time = time.time()
    fps_queue.append(1.0 / (current_time - last_frame_time))
    last_frame_time = current_time
    fps = np.mean(fps_queue)
    
    if show_depth_map and depth_map is not None:
        depth_colored = cv2.applyColorMap((depth_map * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        display_frame = depth_colored
        cv2.putText(display_frame, f"Depth map - FPS: {fps:.1f}", 
                   (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
    else:
        display_frame = draw_enhanced_detections(frame, detections, depth_map)
        draw_info_panel(display_frame, fps)
    
    window_title = f"Detection - YOLO + {'ZoeDepth' if USE_ZOEDEPTH else 'MiDaS'} + VLM"
    cv2.imshow(window_title, display_frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        running = False
        break
    elif key == ord('s'):
        filename = f"screenshot_{screenshot_count:03d}.jpg"
        cv2.imwrite(filename, display_frame)
        print(f"✓ Screenshot: {filename}")
        screenshot_count += 1
    elif key == ord('d'):
        show_depth_map = not show_depth_map
        mode = "Depth map" if show_depth_map else "Detections"
        print(f"Mode: {mode}")
    elif key == ord('+') or key == ord('='):
        distance_scale += 1.0
        print(f"Scale: {distance_scale:.1f}m")
    elif key == ord('-') or key == ord('_'):
        distance_scale = max(1.0, distance_scale - 1.0)
        print(f"Scale: {distance_scale:.1f}m")

print("Stopping system...")
running = False
cap.release()
cv2.destroyAllWindows()
print("✓ Done")