from ultralytics import YOLO
import cv2
import numpy as np
from collections import deque

# Load model
model = YOLO(r"C:\Users\satwi\Downloads\Road Sign detection\runs\detect\train4\weights\best.pt")
class_names = [str(i) for i in range(43)]

# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Store history of detections
history = deque(maxlen=10)  # Store last 10 frames

def apply_ransac_to_boxes(box_history):
    all_centers = np.array([box for frame in box_history for box in frame])
    if len(all_centers) < 5:
        return all_centers  # Not enough data

    # Apply RANSAC for spatial clustering
    import sklearn
    from sklearn.linear_model import RANSACRegressor

    x = all_centers[:, 0].reshape(-1, 1)  # x-coordinates
    y = all_centers[:, 1]  # y-coordinates

    try:
        model_ransac = RANSACRegressor().fit(x, y)
        inlier_mask = model_ransac.inlier_mask_
        return all_centers[inlier_mask]
    except Exception as e:
        print("RANSAC failed:", e)
        return all_centers

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, conf=0.4, verbose=True)
    boxes = results[0].boxes
    skips = False
    centers = []
    if boxes is not None and boxes.cls.numel() > 0:
        filtered_boxes = []
        for i, (cls, conf) in enumerate(zip(boxes.cls, boxes.conf)):
            class_id = int(cls.item())
            if class_id == 1:
                skips = True
                continue

            box = boxes.xyxy[i].cpu().numpy()
            cx = int((box[0] + box[2]) / 2)
            cy = int((box[1] + box[3]) / 2)
            centers.append([cx, cy])

            class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
            # print(f"Detected: {class_name} (ID: {class_id}), Confidence: {conf.item():.2f}")
            filtered_boxes.append(i)

        boxes = boxes[filtered_boxes]
        history.append(centers)

    else:
        # print("No detections")
        history.append([])

    # Apply RANSAC if we have enough history
    ransac_filtered_centers = apply_ransac_to_boxes(history)

    # Draw filtered points (after RANSAC)
    # for center in ransac_filtered_centers:
    #     cv2.circle(frame, (int(center[0]), int(center[1])), 5, (0, 255, 0), -1)

    # Resize and show
    
    if skips:
        plotted_frame = frame
    else:   
        plotted_frame = results[0].plot()
    resized_frame = cv2.resize(plotted_frame, (640, 740))
    cv2.imshow("YOLOv8 with RANSAC Validation", resized_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
