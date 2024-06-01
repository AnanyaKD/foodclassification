from ultralytics import YOLO
model = YOLO('yolov8n-cls.pt')
model.train(data="/Users/anushkadurg/Downloads/archive",epochs=20,batch=32,imgsz=448)