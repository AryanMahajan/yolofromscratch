"""
YOLO Model Optimization Script
Run this to optimize your YOLO model for faster inference
"""

from ultralytics import YOLO
import torch

def optimize_yolo_model():
    """Optimize YOLO model for faster inference"""
    
    # Load your trained model
    model = YOLO('runs/detect/train/weights/best.pt')
    
    # Export to different optimized formats
    print("Exporting optimized model formats...")
    
    # 1. Export to TensorRT (if you have NVIDIA GPU)
    try:
        model.export(format='engine', half=True)  # FP16 for speed
        print("✓ TensorRT engine exported")
    except:
        print("✗ TensorRT export failed (requires NVIDIA GPU)")
    
    # 2. Export to ONNX (universal format, good speed)
    try:
        model.export(format='onnx', half=True, simplify=True)
        print("✓ ONNX model exported")
    except:
        print("✗ ONNX export failed")
    
    # 3. Export to OpenVINO (Intel optimization)
    try:
        model.export(format='openvino', half=True)
        print("✓ OpenVINO model exported")
    except:
        print("✗ OpenVINO export failed")
    
    # 4. Export to CoreML (for Apple devices)
    try:
        model.export(format='coreml', half=True)
        print("✓ CoreML model exported")
    except:
        print("✗ CoreML export failed")

def benchmark_model():
    """Benchmark different model formats"""
    import time
    import numpy as np
    
    # Create dummy input
    dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    formats_to_test = [
        ('runs/detect/train/weights/best.pt', 'PyTorch'),
        ('runs/detect/train/weights/best.onnx', 'ONNX'),
        ('runs/detect/train/weights/best.engine', 'TensorRT'),
    ]
    
    for model_path, format_name in formats_to_test:
        try:
            model = YOLO(model_path)
            
            # Warmup
            for _ in range(5):
                model(dummy_img, verbose=False)
            
            # Benchmark
            start_time = time.time()
            for _ in range(100):
                model(dummy_img, verbose=False)
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 100
            fps = 1 / avg_time
            
            print(f"{format_name}: {avg_time:.4f}s per inference, {fps:.1f} FPS")
            
        except Exception as e:
            print(f"{format_name}: Failed to load - {e}")

def create_lightweight_consumer_config():
    """Generate optimized settings for your consumer"""
    
    config = {
        "model_settings": {
            "conf": 0.5,           # Confidence threshold
            "iou": 0.45,           # IoU threshold  
            "max_det": 30,         # Max detections
            "half": True,          # Use FP16 if available
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        },
        "processing_settings": {
            "resize_width": 640,   # Resize input for consistency
            "jpeg_quality": 80,    # Balance quality vs speed
            "skip_frames": 2,      # Process every nth frame
            "max_fps": 30,         # Rate limiting
        }
    }
    
    return config

if __name__ == "__main__":
    print("Starting YOLO optimization...")
    optimize_yolo_model()
    print("\nBenchmarking models...")
    benchmark_model()
    print("\nOptimization complete!")