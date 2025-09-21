import json
import cv2
import base64
import numpy as np
import logging
from channels.generic.websocket import AsyncWebsocketConsumer
from ultralytics import YOLO

# Get an instance of a logger
logger = logging.getLogger(__name__)

# Load the YOLO model
model = YOLO('runs/detect/train/weights/best.pt')

def correct_padding(s):
    return s + '=' * (-len(s) % 4)

class VideoConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()
        logger.info("WebSocket connected.")

    async def disconnect(self, close_code):
        logger.info(f"WebSocket disconnected with close code: {close_code}")

    async def receive(self, text_data):
        try:
            # logger.debug(f"Received data of type {type(text_data)} and length {len(text_data)}.")

            text_data_json = json.loads(text_data)
            # logger.debug(f"Successfully decoded JSON.")

            frame_data = text_data_json['frame']
            # logger.debug(f"Extracted frame data.")

            img_data_str = frame_data.split(',')[1]
            # logger.debug(f"Extracted base64/ string of length {len(img_data_str)}.")

            padded_img_data_str = correct_padding(img_data_str)
            # logger.debug(f"Padded base64 string.")

            img_data = base64.b64decode(padded_img_data_str)
            # logger.debug(f"Decoded base64 string to bytes of length {len(img_data)}.")

            np_arr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if img is None:
                logger.error("cv2.imdecode returned None. The image data is likely corrupted or in an unsupported format.")
                return
            
            # logger.debug("Successfully decoded image.")

            # Run YOLO model
            results = model(img)
            # logger.debug("Successfully ran YOLO model.")

            # Draw bounding boxes
            annotated_frame = results[0].plot()
            # logger.debug("Successfully drew bounding boxes.")

            # Encode the frame back to base64
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            encoded_frame = base64.b64encode(buffer).decode('utf-8')
            # logger.debug("Successfully encoded frame to base64.")

            # Send the processed frame back
            await self.send(text_data=json.dumps({
                'frame': 'data:image/jpeg;base64,' + encoded_frame
            }))
            logger.debug("Sent processed frame back to client.")

        except Exception as e:
            logger.exception(f"An error occurred in receive: {e}")

