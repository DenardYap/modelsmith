from ultralytics import YOLO
import time 

model = YOLO("yolov8n.pt")
model = YOLO('yolov5s.pt')
model.info()
start = time.time()
results = model('cat.jpg')  
end = time.time()
print(f"Time took: {end - start}s")
print(len(results))
results[0].show() 
