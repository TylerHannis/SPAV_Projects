import os
import cv2
import time
import csv
import warnings
import logging
import contextlib
import io
import torch
from ultralytics import YOLO

# Suppress warnings and set logger levels
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("ultralytics").setLevel(logging.WARNING)

# Define paths: assume you run the script from the project root (/home/tyler/SPAV_Projects)
project_root = os.getcwd()  # Now project_root will be /home/tyler/SPAV_Projects
image_folder = os.path.join(project_root, "Image Captures", "RenamedImages")
print("Looking for images in:", os.path.abspath(image_folder))

# List all .jpg images in the folder
image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(".jpg")]

# Output directories for pre‑retraining images and annotations for YOLOv8 and YOLOv11
output_dir_yolov8 = os.path.join(project_root, "results_yolov8")
output_dir_yolov11 = os.path.join(project_root, "results_yolov11")
os.makedirs(output_dir_yolov8, exist_ok=True)
os.makedirs(output_dir_yolov11, exist_ok=True)
# (YOLOv5 output remains only for inference)

# Load original models
model_yolov8 = YOLO("yolov8n.pt")
model_yolov5 = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model_yolov11 = YOLO("yolo11n.pt")  # Assuming this file exists

# Allowed labels (using COCO naming for stop signs)
allowed_signs = ["stop sign"]

def get_expected_labels(filename):
    """
    Returns a tuple: (expected_list, expected_flag)
    For images starting with "With_Signs": (["stop sign"], "With_Signs")
    Otherwise: ([], "No_Signs")
    """
    basename = os.path.basename(filename)
    if basename.startswith("With_Signs"):
        return (["stop sign"], "With_Signs")
    else:
        return ([], "No_Signs")
    
def get_boxes(result):
    # If result is a list, assume the first element contains the detections.
    if isinstance(result, list):
        return result[0].boxes
    else:
        return result.boxes

def draw_detections_yolov8(image, result, allowed_labels):
    count = 0
    detection_info = []
    annotations = []
    names = model_yolov8.names if hasattr(model_yolov8, 'names') else {}
    h, w = image.shape[:2]
    for box in get_boxes(result):
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
        # YOLO-format annotation: class_id x_center y_center width height (normalized)
        x_center = (x1 + x2) / 2.0
        y_center = (y1 + y2) / 2.0
        box_width = x2 - x1
        box_height = y2 - y1
        annotation_line = f"0 {x_center/w:.6f} {y_center/h:.6f} {box_width/w:.6f} {box_height/h:.6f}"
        annotations.append(annotation_line)
    return image, count, detection_info, annotations

def draw_detections_yolov5(image, result, allowed_labels):
    """
    Draws bounding boxes for allowed detections from YOLOv5.
    Returns:
      - Labeled image,
      - Count of allowed detections,
      - A list of detection info strings.
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
    count = 0
    detection_info = []
    annotations = []
    names = model_yolov11.names if hasattr(model_yolov11, 'names') else {}
    h, w = image.shape[:2]
    for box in get_boxes(result):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = box.conf[0]
        cls_idx = int(box.cls[0])
        class_name = names.get(cls_idx, str(cls_idx)).lower()
        if class_name not in allowed_labels:
            continue
        count += 1
        label = f"{class_name}:{conf:.2f}"
        detection_info.append(label)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue for YOLOv11
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        x_center = (x1 + x2) / 2.0
        y_center = (y1 + y2) / 2.0
        box_width = x2 - x1
        box_height = y2 - y1
        annotation_line = f"0 {x_center/w:.6f} {y_center/h:.6f} {box_width/w:.6f} {box_height/h:.6f}"
        annotations.append(annotation_line)
    return image, count, detection_info, annotations

# Dictionaries to store pre‑retraining results (keyed by image basename)
pre_results_yolov8 = {}
pre_results_yolov5 = {}
pre_results_yolov11 = {}

print("\n--- Pre-Retraining Inference ---")
for img_path in image_files:
    expected, expected_flag = get_expected_labels(img_path)
    image = cv2.imread(img_path)
    if image is None:
        print(f"Error reading {img_path}")
        continue
    base = os.path.basename(img_path)
    print(f"\nProcessing {base} (Expected: {expected_flag})")
    
    # YOLOv8 pre‑inference
    start = time.time()
    result_v8 = model_yolov8(image, verbose=False)
    pre_time_v8 = time.time() - start
    labeled_img_v8, count_pre_v8, info_pre_v8, annotations_v8 = draw_detections_yolov8(image.copy(), result_v8, allowed_signs)
    pre_status_v8 = ("TP" if (expected_flag=="With_Signs" and count_pre_v8 > 0)
                     else ("FN" if expected_flag=="With_Signs" and count_pre_v8 == 0
                           else ("TN" if expected_flag=="No_Signs" and count_pre_v8 == 0 else "FP")))
    pre_results_yolov8[base] = {"expected": expected_flag, "pre_count": count_pre_v8,
                                "pre_status": pre_status_v8, "pre_time": pre_time_v8}
    print("YOLOv8 pre-detections:", ", ".join(info_pre_v8) if info_pre_v8 else "None")
    # Save the original (unmodified) image and corresponding annotation file for retraining
    if count_pre_v8 > 0:
        cv2.imwrite(os.path.join(output_dir_yolov8, base), image)
        txt_filename = os.path.splitext(base)[0] + ".txt"
        with open(os.path.join(output_dir_yolov8, txt_filename), "w") as f:
            for line in annotations_v8:
                f.write(line + "\n")
    
    # YOLOv5 pre‑inference (annotations not needed for retraining)
    start = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        result_v5 = model_yolov5(img_path)
    pre_time_v5 = time.time() - start
    _, count_pre_v5, info_pre_v5 = draw_detections_yolov5(image.copy(), result_v5, allowed_signs)
    pre_status_v5 = ("TP" if (expected_flag=="With_Signs" and count_pre_v5 > 0)
                     else ("FN" if expected_flag=="With_Signs" and count_pre_v5 == 0
                           else ("TN" if expected_flag=="No_Signs" and count_pre_v5 == 0 else "FP")))
    pre_results_yolov5[base] = {"expected": expected_flag, "pre_count": count_pre_v5,
                                "pre_status": pre_status_v5, "pre_time": pre_time_v5}
    print("YOLOv5 pre-detections:", ", ".join(info_pre_v5) if info_pre_v5 else "None")
    # (For YOLOv5, we do not save images or annotations for retraining.)
    
    # YOLOv11 pre‑inference
    start = time.time()
    result_v11 = model_yolov11(image, verbose=False)
    pre_time_v11 = time.time() - start
    labeled_img_v11, count_pre_v11, info_pre_v11, annotations_v11 = draw_detections_yolov11(image.copy(), result_v11, allowed_signs)
    pre_status_v11 = ("TP" if (expected_flag=="With_Signs" and count_pre_v11 > 0)
                      else ("FN" if expected_flag=="With_Signs" and count_pre_v11 == 0
                            else ("TN" if expected_flag=="No_Signs" and count_pre_v11 == 0 else "FP")))
    pre_results_yolov11[base] = {"expected": expected_flag, "pre_count": count_pre_v11,
                                 "pre_status": pre_status_v11, "pre_time": pre_time_v11}
    print("YOLOv11 pre-detections:", ", ".join(info_pre_v11) if info_pre_v11 else "None")
    if count_pre_v11 > 0:
        cv2.imwrite(os.path.join(output_dir_yolov11, base), image)
        txt_filename = os.path.splitext(base)[0] + ".txt"
        with open(os.path.join(output_dir_yolov11, txt_filename), "w") as f:
            for line in annotations_v11:
                f.write(line + "\n")

# ----- Create Separate data.yaml Files for Retraining -----
# For YOLOv8 retraining:
labeled_folder_v8 = os.path.abspath(output_dir_yolov8)
data_yaml_v8 = "data_v8.yaml"
with open(data_yaml_v8, "w") as f:
    f.write(f"train: {labeled_folder_v8}\n")
    f.write(f"val: {labeled_folder_v8}\n")
    f.write("nc: 1\n")
    f.write("names: ['stop sign']\n")
print(f"\nData YAML for YOLOv8 created at {data_yaml_v8} using images from {labeled_folder_v8}")

# For YOLOv11 retraining:
labeled_folder_v11 = os.path.abspath(output_dir_yolov11)
data_yaml_v11 = "data_v11.yaml"
with open(data_yaml_v11, "w") as f:
    f.write(f"train: {labeled_folder_v11}\n")
    f.write(f"val: {labeled_folder_v11}\n")
    f.write("nc: 1\n")
    f.write("names: ['stop sign']\n")
print(f"\nData YAML for YOLOv11 created at {data_yaml_v11} using images from {labeled_folder_v11}")

# ----- Retraining Step -----
epochs = 10  # Adjust as needed

print("\n--- Retraining Step ---")
if os.path.exists(data_yaml_v8):
    print("Retraining YOLOv8...")
    model_yolov8.train(data=data_yaml_v8, epochs=epochs, imgsz=640,
                         project=project_root, name="yolov8_retrained")
    retrained_yolov8_path = os.path.join("yolov8_retrained", "weights", "best.pt")
    if os.path.exists(retrained_yolov8_path):
        os.system(f"cp {retrained_yolov8_path} yolov8n_retrained.pt")
        print("YOLOv8 retraining complete. New weights saved as 'yolov8n_retrained.pt'")
    else:
        print("Retrained YOLOv8 weights not found!")
else:
    print("data_v8.yaml not found; skipping YOLOv8 retraining.")

if os.path.exists(data_yaml_v11):
    print("Retraining YOLOv11...")
    model_yolov11.train(data=data_yaml_v11, epochs=epochs, imgsz=640,
                          project=project_root, name="yolov11_retrained")
    retrained_yolov11_path = os.path.join("yolov11_retrained", "weights", "best.pt")
    if os.path.exists(retrained_yolov11_path):
        os.system(f"cp {retrained_yolov11_path} yolo11n_retrained.pt")
        print("YOLOv11 retraining complete. New weights saved as 'yolo11n_retrained.pt'")
    else:
        print("Retrained YOLOv11 weights not found!")
else:
    print("data_v11.yaml not found; skipping YOLOv11 retraining.")
# YOLOv5 retraining is assumed to be done externally.

# ----- Post-Retraining Inference -----
post_results_yolov8 = {}
post_results_yolov5 = {}
post_results_yolov11 = {}

print("\n--- Post-Retraining Inference ---")
# YOLOv8 post‑inference
retrained_yolov8_file = "yolov8n_retrained.pt"
if os.path.exists(retrained_yolov8_file):
    model_yolov8_rt = YOLO(retrained_yolov8_file)
    print("\nRunning YOLOv8 post‑retraining inference on all images...")
    for img_path in image_files:
        expected, expected_flag = get_expected_labels(img_path)
        image = cv2.imread(img_path)
        if image is None:
            continue
        base = os.path.basename(img_path)
        start = time.time()
        result = model_yolov8_rt(image, verbose=False)
        post_time = time.time() - start
        _, count_post, info_post, _ = draw_detections_yolov8(image.copy(), result, allowed_signs)
        post_status = ("TP" if (expected_flag=="With_Signs" and count_post > 0)
                       else ("FN" if expected_flag=="With_Signs" and count_post == 0
                             else ("TN" if expected_flag=="No_Signs" and count_post == 0 else "FP")))
        post_results_yolov8[base] = {"post_count": count_post,
                                     "post_status": post_status, "post_time": post_time}
        print(f"YOLOv8 post-detections for {base}:", ", ".join(info_post) if info_post else "None")
else:
    print("\nNo retrained YOLOv8 model found; skipping post‑inference for YOLOv8.")

# YOLOv5 post‑inference (assuming retrained weights provided externally)
retrained_yolov5_file = "yolov5s_retrained.pt"
if os.path.exists(retrained_yolov5_file):
    model_yolov5_rt = torch.hub.load('ultralytics/yolov5', 'custom', path=retrained_yolov5_file)
    print("\nRunning YOLOv5 post‑retraining inference on all images...")
    for img_path in image_files:
        expected, expected_flag = get_expected_labels(img_path)
        image = cv2.imread(img_path)
        if image is None:
            continue
        base = os.path.basename(img_path)
        start = time.time()
        with contextlib.redirect_stdout(io.StringIO()):
            result = model_yolov5_rt(img_path)
        post_time = time.time() - start
        _, count_post, info_post = draw_detections_yolov5(image.copy(), result, allowed_signs)
        post_status = ("TP" if (expected_flag=="With_Signs" and count_post > 0)
                       else ("FN" if expected_flag=="With_Signs" and count_post == 0
                             else ("TN" if expected_flag=="No_Signs" and count_post == 0 else "FP")))
        post_results_yolov5[base] = {"post_count": count_post,
                                     "post_status": post_status, "post_time": post_time}
        print(f"YOLOv5 post-detections for {base}:", ", ".join(info_post) if info_post else "None")
else:
    print("\nNo retrained YOLOv5 model found; skipping post‑inference for YOLOv5.")

# YOLOv11 post‑inference
retrained_yolov11_file = "yolo11n_retrained.pt"
if os.path.exists(retrained_yolov11_file):
    model_yolov11_rt = YOLO(retrained_yolov11_file)
    print("\nRunning YOLOv11 post‑retraining inference on all images...")
    for img_path in image_files:
        expected, expected_flag = get_expected_labels(img_path)
        image = cv2.imread(img_path)
        if image is None:
            continue
        base = os.path.basename(img_path)
        start = time.time()
        result = model_yolov11_rt(image, verbose=False)
        post_time = time.time() - start
        _, count_post, info_post, _ = draw_detections_yolov11(image.copy(), result, allowed_signs)
        post_status = ("TP" if (expected_flag=="With_Signs" and count_post > 0)
                       else ("FN" if expected_flag=="With_Signs" and count_post == 0
                             else ("TN" if expected_flag=="No_Signs" and count_post == 0 else "FP")))
        post_results_yolov11[base] = {"post_count": count_post,
                                      "post_status": post_status, "post_time": post_time}
        print(f"YOLOv11 post-detections for {base}:", ", ".join(info_post) if info_post else "None")
else:
    print("\nNo retrained YOLOv11 model found; skipping post‑inference for YOLOv11.")

# ----- Create CSV Comparison Files -----
csv_header = ["image", "expected", "pre_num_allowed_detections", "pre_result", "pre_processing_time",
              "post_num_allowed_detections", "post_result", "post_processing_time",
              "difference_flag", "improvement_status"]

def write_csv(csv_filename, pre_results, post_results):
    rows = []
    for img, pre_data in pre_results.items():
        expected = pre_data["expected"]
        pre_count = pre_data["pre_count"]
        pre_status = pre_data["pre_status"]
        pre_time = f"{pre_data['pre_time']:.4f}"
        if img in post_results:
            post_count = post_results[img]["post_count"]
            post_status = post_results[img]["post_status"]
            post_time = f"{post_results[img]['post_time']:.4f}"
        else:
            post_count = 0
            post_status = ""
            post_time = ""
        diff = post_count - pre_count
        diff_flag = "Yes" if diff != 0 else "No"
        if expected == "With_Signs":
            if diff > 0:
                imp = "Improved"
            elif diff < 0:
                imp = "Worsened"
            else:
                imp = "No change"
        else:
            if diff < 0:
                imp = "Improved"
            elif diff > 0:
                imp = "Worsened"
            else:
                imp = "No change"
        rows.append([img, expected, pre_count, pre_status, pre_time,
                     post_count, post_status, post_time, diff_flag, imp])
    with open(csv_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        writer.writerows(rows)

write_csv("detection_results_yolov8_comparison.csv", pre_results_yolov8, post_results_yolov8)
write_csv("detection_results_yolov5_comparison.csv", pre_results_yolov5, post_results_yolov5)
write_csv("detection_results_yolov11_comparison.csv", pre_results_yolov11, post_results_yolov11)

print("\nCSV files comparing pre- and post-retraining results have been saved:")
print(" - detection_results_yolov8_comparison.csv")
print(" - detection_results_yolov5_comparison.csv")
print(" - detection_results_yolov11_comparison.csv")
