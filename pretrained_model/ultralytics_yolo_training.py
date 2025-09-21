from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model.train(
    data="coco8.yaml",
    epochs=150,
    imgsz=640,
    batch=16,
)
