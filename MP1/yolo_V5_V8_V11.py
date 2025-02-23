import os
import cv2
from ultralytics import YOLO
import torch
import csv
import warnings
import logging
import contextlib
import io
import time

# Suppress FutureWarnings and set logger levels
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("ultralytics").setLevel(logging.WARNING)

# Define paths
project_root = "SPAV_Projects"
image_folder = os.path.join("..", project_root, "Image Captures", "RenamedImages")
print("Looking for images in:", os.path.abspath(image_folder))

# Create output directories for labeled images for each model (only for images with detections)
output_dir_yolov8 = os.path.join("results_yolov8")
output_dir_yolov5 = os.path.join("results_yolov5")
output_dir_yolov11 = os.path.join("results_yolov11")
os.makedirs(output_dir_yolov8, exist_ok=True)
os.makedirs(output_dir_yolov5, exist_ok=True)
os.makedirs(output_dir_yolov11, exist_ok=True)

# List all .jpg images in the folder
image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(".jpg")]

# Load models
model_yolov8 = YOLO("yolov8n.pt")
model_yolov5 = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model_yolov11 = YOLO("yolo11n.pt")  # New YOLOv11 model

# Allowed labels for street signs (adjusted to match COCO names)
allowed_signs = ["stop sign"]

def get_expected_labels(filename):
    """
    Parses the filename to determine the expected objects.
    Returns a tuple: (expected_list, expected_flag)
      - expected_list: For images with signs, a list (["stop sign"]).
                        For images with no signs, an empty list.
      - expected_flag: "With_Signs" or "No_Signs"
    """
    basename = os.path.basename(filename)
    if basename.startswith("With_Signs"):
        return (["stop sign"], "With_Signs")
    else:
        return (["stop sign"], "No_Signs")

def draw_detections_yolov8(image, result, allowed_labels):
    """
    Draws bounding boxes for allowed detections from YOLOv8.
    Returns the labeled image, count of allowed detections, and a list of detection details.
    """
    count = 0
    detection_info = []
    names = model_yolov8.names if hasattr(model_yolov8, 'names') else {}
    for box in result[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = box.conf[0]
        cls_idx = int(box.cls[0])
        class_name = names.get(cls_idx, str(cls_idx)).lower()
        if class_name not in allowed_labels:
            continue
        count += 1
        label = f"{class_name}:{conf:.2f}"
        detection_info.append(label)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image, count, detection_info

def draw_detections_yolov5(image, result, allowed_labels):
    """
    Draws bounding boxes for allowed detections from YOLOv5.
    Returns the labeled image, count of allowed detections, and a list of detection details.
    """
    count = 0
    detection_info = []
    for det in result.xyxy[0]:
        x1, y1, x2, y2, conf, cls_idx = det.tolist()
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        class_name = result.names[int(cls_idx)].lower()
        if class_name not in allowed_labels:
            continue
        count += 1
        label = f"{class_name}:{conf:.2f}"
        detection_info.append(label)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return image, count, detection_info

def draw_detections_yolov11(image, result, allowed_labels):
    """
    Draws bounding boxes for allowed detections from YOLOv11.
    Returns the labeled image, count of allowed detections, and a list of detection details.
    Uses blue color for drawing.
    """
    count = 0
    detection_info = []
    names = model_yolov11.names if hasattr(model_yolov11, 'names') else {}
    for box in result[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = box.conf[0]
        cls_idx = int(box.cls[0])
        class_name = names.get(cls_idx, str(cls_idx)).lower()
        if class_name not in allowed_labels:
            continue
        count += 1
        label = f"{class_name}:{conf:.2f}"
        detection_info.append(label)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # blue
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return image, count, detection_info

# Prepare separate lists to store CSV rows for each model.
csv_rows_yolov8 = []
csv_rows_yolov5 = []
csv_rows_yolov11 = []
csv_header = ["image", "expected", "num_allowed_detections", "result", "processing_time"]

# Accumulators for total processing time (in seconds)
total_time_yolov8 = 0
total_time_yolov5 = 0
total_time_yolov11 = 0

for img_path in image_files:
    expected, expected_flag = get_expected_labels(img_path)
    image = cv2.imread(img_path)
    if image is None:
        print(f"Error reading {img_path}")
        continue

    print(f"\nProcessing image: {os.path.basename(img_path)} (Expected: {expected_flag})")

    # --- YOLOv8 Inference ---
    start_time = time.time()
    result_yolov8 = model_yolov8(image, verbose=False)
    end_time = time.time()
    proc_time_yolov8 = end_time - start_time
    total_time_yolov8 += proc_time_yolov8

    labeled_image_yolov8 = image.copy()
    labeled_image_yolov8, count_yolov8, info_yolov8 = draw_detections_yolov8(labeled_image_yolov8, result_yolov8, allowed_signs)
    if info_yolov8:
        print("YOLOv8 detections:", ", ".join(info_yolov8))
    else:
        print("YOLOv8 detected no allowed signs.")
    if expected_flag == "With_Signs":
        result_y8 = "TP" if count_yolov8 > 0 else "FN"
    else:
        result_y8 = "FP" if count_yolov8 > 0 else "TN"
    if count_yolov8 > 0:
        out_path_yolov8 = os.path.join(output_dir_yolov8, os.path.basename(img_path))
        cv2.imwrite(out_path_yolov8, labeled_image_yolov8)
    csv_rows_yolov8.append([os.path.basename(img_path), expected_flag, count_yolov8, result_y8, f"{proc_time_yolov8:.4f}"])

    # --- YOLOv5 Inference ---
    start_time = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        result_yolov5 = model_yolov5(img_path)
    end_time = time.time()
    proc_time_yolov5 = end_time - start_time
    total_time_yolov5 += proc_time_yolov5

    labeled_image_yolov5 = image.copy()
    labeled_image_yolov5, count_yolov5, info_yolov5 = draw_detections_yolov5(labeled_image_yolov5, result_yolov5, allowed_signs)
    if info_yolov5:
        print("YOLOv5 detections:", ", ".join(info_yolov5))
    else:
        print("YOLOv5 detected no allowed signs.")
    if expected_flag == "With_Signs":
        result_y5 = "TP" if count_yolov5 > 0 else "FN"
    else:
        result_y5 = "FP" if count_yolov5 > 0 else "TN"
    if count_yolov5 > 0:
        out_path_yolov5 = os.path.join(output_dir_yolov5, os.path.basename(img_path))
        cv2.imwrite(out_path_yolov5, labeled_image_yolov5)
    csv_rows_yolov5.append([os.path.basename(img_path), expected_flag, count_yolov5, result_y5, f"{proc_time_yolov5:.4f}"])

    # --- YOLOv11 Inference ---
    start_time = time.time()
    result_yolov11 = model_yolov11(image, verbose=False)
    end_time = time.time()
    proc_time_yolov11 = end_time - start_time
    total_time_yolov11 += proc_time_yolov11

    labeled_image_yolov11 = image.copy()
    labeled_image_yolov11, count_yolov11, info_yolov11 = draw_detections_yolov11(labeled_image_yolov11, result_yolov11, allowed_signs)
    if info_yolov11:
        print("YOLOv11 detections:", ", ".join(info_yolov11))
    else:
        print("YOLOv11 detected no allowed signs.")
    if expected_flag == "With_Signs":
        result_y11 = "TP" if count_yolov11 > 0 else "FN"
    else:
        result_y11 = "FP" if count_yolov11 > 0 else "TN"
    if count_yolov11 > 0:
        out_path_yolov11 = os.path.join(output_dir_yolov11, os.path.basename(img_path))
        cv2.imwrite(out_path_yolov11, labeled_image_yolov11)
    csv_rows_yolov11.append([os.path.basename(img_path), expected_flag, count_yolov11, result_y11, f"{proc_time_yolov11:.4f}"])

# Write separate CSV files for each model's results.
csv_file_yolov8 = "detection_results_yolov8.csv"
with open(csv_file_yolov8, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(csv_header)
    writer.writerows(csv_rows_yolov8)

csv_file_yolov5 = "detection_results_yolov5.csv"
with open(csv_file_yolov5, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(csv_header)
    writer.writerows(csv_rows_yolov5)

csv_file_yolov11 = "detection_results_yolov11.csv"
with open(csv_file_yolov11, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(csv_header)
    writer.writerows(csv_rows_yolov11)

print("\nDetection results for YOLOv8 have been saved to", csv_file_yolov8)
print("Detection results for YOLOv5 have been saved to", csv_file_yolov5)
print("Detection results for YOLOv11 have been saved to", csv_file_yolov11)

# Print final accumulated processing times for each model
print("\nFinal processing time totals (in seconds):")
print(f"YOLOv8: {total_time_yolov8:.4f} sec")
print(f"YOLOv5: {total_time_yolov5:.4f} sec")
print(f"YOLOv11: {total_time_yolov11:.4f} sec")
