import cv2
import torch
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 Model (Pre-trained on COCO Dataset)
model = YOLO("yolov8n.pt")  # Use "yolov8s.pt" for better accuracy

# Load Face Detection Model (for Age & Gender)
face_net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt"
)

# Load Age & Gender Models
age_net = cv2.dnn.readNetFromCaffe(
    "age_deploy.prototxt", 
    "age_net.caffemodel"
)
gender_net = cv2.dnn.readNetFromCaffe(
    "gender_deploy.prototxt", 
    "gender_net.caffemodel"
)

# Labels
age_labels = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]
gender_labels = ["Male", "Female"]

# Open Webcam (0 for default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert BGR to RGB (YOLO expects RGB images)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run YOLOv8 detection
    results = model(frame_rgb)

    # Process detection results
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box
            confidence = float(box.conf[0])  # Extract confidence score
            class_id = int(box.cls[0])  # Extract class ID
            label = model.names[class_id]  # Get object label

            # Draw bounding box around detected objects
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # If it's a person, run Age & Gender Estimation
            if label == "person" and confidence > 0.3:
                face = frame[y1:y2, x1:x2]
                blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.426, 87.768, 114.895), swapRB=False)

                # Predict Gender
                gender_net.setInput(blob)
                gender_preds = gender_net.forward()
                gender = gender_labels[gender_preds[0].argmax()]

                # Predict Age
                age_net.setInput(blob)
                age_preds = age_net.forward()
                age = age_labels[age_preds[0].argmax()]

                label = f"{label}: {gender}, Age: {age}"

            # Display label and confidence
            text = f"{label} ({confidence:.2f})"
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show the video feed
    cv2.imshow("YOLOv8 Multi-Object + Age/Gender Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
