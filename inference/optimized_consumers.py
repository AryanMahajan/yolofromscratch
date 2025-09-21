import json
import cv2
import base64
import numpy as np
import logging
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from channels.generic.websocket import AsyncWebsocketConsumer
from ultralytics import YOLO

# Get an instance of a logger
logger = logging.getLogger(__name__)

class OptimizedVideoConsumer(AsyncWebsocketConsumer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Load model once per consumer instance
        self.model = YOLO('runs/detect/train/weights/best.pt')
        # Configure model for speed
        self.model.conf = 0.5  # Lower confidence threshold
        self.model.iou = 0.45  # Lower IoU threshold
        self.model.max_det = 50  # Limit max detections
        
        # Thread pool for CPU-intensive operations
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Frame processing control
        self.processing = False
        self.last_process_time = 0
        self.min_frame_interval = 1/30  # Max 30 FPS processing
        
        # Frame skip counter for adaptive processing
        self.frame_count = 0
        self.skip_frames = 1  # Process every nth frame initially

    async def connect(self):
        await self.accept()
        logger.info("WebSocket connected.")

    async def disconnect(self, close_code):
        logger.info(f"WebSocket disconnected with close code: {close_code}")
        # Shutdown executor
        self.executor.shutdown(wait=False)

    def correct_padding(self, s):
        return s + '=' * (-len(s) % 4)

    def process_frame_sync(self, img_data_str):
        """Synchronous frame processing to run in thread pool"""
        try:
            # Decode image
            padded_img_data_str = self.correct_padding(img_data_str)
            img_data = base64.b64decode(padded_img_data_str)
            np_arr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if img is None:
                logger.error("cv2.imdecode returned None")
                return None

            # Resize image for faster processing (optional)
            height, width = img.shape[:2]
            if width > 640:
                scale = 640 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

            # Run YOLO model with optimized settings
            results = self.model(img, verbose=False)  # Disable verbose output
            
            # Draw bounding boxes
            annotated_frame = results[0].plot()

            # Encode with lower quality for speed
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 80]  # Reduce quality for speed
            _, buffer = cv2.imencode('.jpg', annotated_frame, encode_params)
            encoded_frame = base64.b64encode(buffer).decode('utf-8')

            return encoded_frame

        except Exception as e:
            logger.exception(f"Error in frame processing: {e}")
            return None

    async def receive(self, text_data):
        # Skip frames if still processing previous frame
        if self.processing:
            return
            
        current_time = time.time()
        
        # Rate limiting - don't process frames too frequently
        if current_time - self.last_process_time < self.min_frame_interval:
            return
            
        # Adaptive frame skipping
        self.frame_count += 1
        if self.frame_count % self.skip_frames != 0:
            return

        self.processing = True
        self.last_process_time = current_time

        try:
            text_data_json = json.loads(text_data)
            frame_data = text_data_json['frame']
            img_data_str = frame_data.split(',')[1]

            # Run processing in thread pool
            loop = asyncio.get_event_loop()
            encoded_frame = await loop.run_in_executor(
                self.executor, 
                self.process_frame_sync, 
                img_data_str
            )

            if encoded_frame:
                # Send response
                await self.send(text_data=json.dumps({
                    'frame': f'data:image/jpeg;base64,{encoded_frame}'
                }))
                
                # Adaptive frame skipping based on processing time
                processing_time = time.time() - current_time
                if processing_time > 0.1:  # If processing takes > 100ms
                    self.skip_frames = min(self.skip_frames + 1, 5)
                elif processing_time < 0.05:  # If processing < 50ms
                    self.skip_frames = max(self.skip_frames - 1, 1)

        except Exception as e:
            logger.exception(f"An error occurred in receive: {e}")
        finally:
            self.processing = False


class UltraLightVideoConsumer(AsyncWebsocketConsumer):
    """Ultra-lightweight version with minimal processing"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Load model with minimal settings
        self.model = YOLO('runs/detect/train/weights/best.pt')
        # Aggressive optimization
        self.model.conf = 0.6  # Higher confidence for fewer detections
        self.model.iou = 0.5
        self.model.max_det = 20  # Very low max detections
        
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.processing = False
        self.frame_count = 0
        
    async def connect(self):
        await self.accept()
        logger.info("Ultra-light WebSocket connected.")

    async def disconnect(self, close_code):
        logger.info(f"Ultra-light WebSocket disconnected: {close_code}")
        self.executor.shutdown(wait=False)

    def ultra_fast_process(self, img_data_str):
        """Minimal processing for maximum speed"""
        try:
            # Quick decode
            img_data_str += '=' * (-len(img_data_str) % 4)
            img_data = base64.b64decode(img_data_str)
            np_arr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if img is None:
                return None

            # Aggressive resize for speed
            img = cv2.resize(img, (416, 416))  # Fixed small size

            # Fast inference
            results = self.model(img, verbose=False, stream=True)
            result = next(results)  # Get first result only
            
            # Quick annotation
            annotated_frame = result.plot()
            
            # Fast encode with low quality
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 60]
            _, buffer = cv2.imencode('.jpg', annotated_frame, encode_params)
            return base64.b64encode(buffer).decode('utf-8')

        except Exception as e:
            logger.error(f"Ultra-fast processing error: {e}")
            return None

    async def receive(self, text_data):
        # Process every 3rd frame only
        self.frame_count += 1
        if self.processing or self.frame_count % 3 != 0:
            return

        self.processing = True

        try:
            data = json.loads(text_data)
            img_data_str = data['frame'].split(',')[1]

            # Ultra-fast processing
            loop = asyncio.get_event_loop()
            encoded_frame = await loop.run_in_executor(
                self.executor,
                self.ultra_fast_process,
                img_data_str
            )

            if encoded_frame:
                await self.send(text_data=json.dumps({
                    'frame': f'data:image/jpeg;base64,{encoded_frame}'
                }))

        except Exception as e:
            logger.exception(f"Ultra-light receive error: {e}")
        finally:
            self.processing = False