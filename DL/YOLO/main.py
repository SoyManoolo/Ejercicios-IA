import cv2
from ultralytics import YOLO
import supervision as sv
import matplotlib.pyplot as plt

model = YOLO("yolo11n.pt")
box_annotator = sv.BoxAnnotator()

clases = list(model.names.values())
for i in range(0, len(clases), 4):
    row = clases[i:i+4]
    print(" ".join(f"{i+j:2d}. {c:<15}" for j, c in enumerate(row)))

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

if ret:
    plt.figure(figsize=(10,6))
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

results = model(frame, conf=0.5, verbose=False) [0]

# def main(video_file_path):
#     frame_generator = sv.get_video_frames_generator()
#     for i, frame in enumarate(frame_generator):
#         result = model(frame, device="mps", verbose=False)
#         detections = sv.Detections.from_ultralytics(result)
        
#         annotated_frame = frame.copy()

#     cv2.destroyAllWindows()
