from ultralytics import YOLO

model = YOLO("yolo12s.pt")
results = model.predict(source="https://ultralytics.com/images/bus.jpg", conf=0.4, device='cpu')

r = results[0]
print(r.boxes.xyxy[:3])
print(r.boxes.conf[:3])
print(r.boxes.cls[:3])
