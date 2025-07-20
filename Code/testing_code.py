from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO(r"C:\Users\satwi\Downloads\Road Sign detection\runs\detect\train4\weights\best.pt")

# Define GTSRB class names (can be replaced with actual names)
class_names = [str(i) for i in range(43)]

# Initialize webcam/IP camera
# cap = cv2.VideoCapture("http://192.168.198.123:8080/video")
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    # Set confidence threshold to > 0.4
    results = model.predict(frame, conf=0.4, verbose=False)
    boxes = results[0].boxes

    if boxes is not None and boxes.cls.numel() > 0:
        filtered_boxes = []
        for i, (cls, conf) in enumerate(zip(boxes.cls, boxes.conf)):
            class_id = int(cls.item())
            if class_id == 1:
                continue  # Skip detections with class ID 1
            class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
            print(f"Detected: {class_name} (ID: {class_id}), Confidence: {conf.item():.2f}")
            filtered_boxes.append(i)

        # Keep only boxes not having class_id = 1
        results[0].boxes = boxes[filtered_boxes]
    else:
        print("No detections")

    # Resize the plotted frame to 640x740 for display
    plotted_frame = results[0].plot()
    resized_frame = cv2.resize(plotted_frame, (640, 740))

    cv2.imshow("YOLOv8 Real-Time", resized_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
