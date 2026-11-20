import sys
from ultralytics import YOLO

# Usage: python check_model.py path\to\image.jpg
image_path = sys.argv[1]

model = YOLO('models/best.pt')
result = model.predict(image_path, conf=0.10, device='cpu', verbose=False)[0]

print(f"Image: {image_path}")
print(f"Total detections at conf>=0.10: {len(result.boxes)}")
for box in result.boxes:
    name = model.names[int(box.cls)]
    conf = float(box.conf)
    coords = [round(v) for v in box.xyxy[0].tolist()]
    print(f"  {name:15s} conf={conf:.2f}  box={coords}")