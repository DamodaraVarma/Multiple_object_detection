# YOLOv8 Multi-Object Detection with Age & Gender Estimation

## 📌 Overview

This project combines **YOLOv8 (You Only Look Once v8)** for real-time object detection with **deep learning models for Age and Gender classification**.
It uses a webcam to detect people and estimate their age and gender, while also detecting other objects trained on the **COCO dataset**.

---

## ⚡ Features

* Real-time **object detection** using YOLOv8 (`yolov8n.pt`).
* Detects **persons** and applies **age & gender prediction**.
* Uses **OpenCV DNN models** (`.prototxt` + `.caffemodel`) for classification.
* Displays bounding boxes, confidence scores, and age/gender labels.

---

## 🛠 Requirements

Install the following dependencies before running:

```bash
pip install ultralytics opencv-python torch numpy
```

### Pre-trained Models Required

Download and place the following files in your project directory:

1. `yolov8n.pt` → [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
2. `deploy.prototxt` → Face detection prototxt
3. `age_deploy.prototxt` and `age_net.caffemodel` → Age classification model
4. `gender_deploy.prototxt` and `gender_net.caffemodel` → Gender classification model

*(Age & gender models are usually available from OpenCV’s sample DNN models repository.)*

---

## ▶️ How to Run

1. Clone this project or copy the script.
2. Ensure all `.prototxt` and `.caffemodel` files are in the same directory.
3. Run the script:

   ```bash
   python age_gender_yolo.py
   ```
4. A webcam window will open showing detections.
5. Press **`q`** to exit.

---

## 🎯 Labels Used

* **YOLOv8** → COCO dataset (80 classes, including `person`)
* **Gender** → `Male`, `Female`
* **Age** → `(0-2), (4-6), (8-12), (15-20), (25-32), (38-43), (48-53), (60-100)`

---

## 📌 Notes

* YOLOv8 detects **all objects**, but **age & gender estimation** is only applied when a **person** is detected.
* You can replace `yolov8n.pt` with `yolov8s.pt` for better accuracy (but slightly slower performance).
* Ensure good lighting for more accurate predictions.

---

## 🚀 Future Improvements

* Improve accuracy using **deep learning-based face detectors** instead of Caffe models.
* Train custom **age/gender datasets** for more precise classification.
* Extend project to detect **mood/emotions** along with age and gender.

