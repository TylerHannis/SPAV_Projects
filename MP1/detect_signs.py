import os
import csv
import re
import sys

# If "yolo_inference.py" is in C:\Dev\YOLO-V3, add that path to sys.path:
sys.path.append(r"C:\Dev\YOLO-V3")

from yolo_inference import YOLODetector  # the class we just created

def parse_image_filename(filename):
    """
    Example filenames:
      'No_Signs_time(16)_fog(25)_rain(0)_202.jpg'
      'With_Signs_time(8)_fog(90)_rain(0)_164.jpg'
    """
    base, _ = os.path.splitext(filename)

    info = {
        'expected_label': None,
        'time': None,
        'fog': None,
        'rain': None,
        'extra': None
    }

    # Check if it starts with "With_Signs_" or "No_Signs_"
    if base.startswith("With_Signs_"):
        info['expected_label'] = "With_Signs"
        rest = base[len("With_Signs_"):]
    elif base.startswith("No_Signs_"):
        info['expected_label'] = "No_Signs"
        rest = base[len("No_Signs_"):]
    else:
        info['expected_label'] = "UNKNOWN"
        rest = base

    # Parse environment data
    param_pattern = re.compile(r'(\w+)\(([^)]*)\)')
    parts = rest.split("_")
    for chunk in parts:
        m = param_pattern.match(chunk)
        if m:
            key, val = m.group(1).lower(), m.group(2)
            if key in ['time', 'fog', 'rain']:
                info[key] = val
            else:
                info['extra'] = chunk
        else:
            info['extra'] = chunk

    return info

def main():
    # 1. Where are your images?
    images_dir = r"C:\Dev\SPAV_Projects\Image Captures\RenamedImages"
    # 2. Where to save detection images + CSV?
    output_dir = r"C:\Dev\SPAV_Projects\Detections"
    os.makedirs(output_dir, exist_ok=True)

    # 3. Gather images
    all_images = [
        os.path.join(images_dir, f)
        for f in os.listdir(images_dir)
        if f.lower().endswith('.jpg')
    ]
    print(f"[INFO] Found {len(all_images)} images in {images_dir}")

    # 4. Initialize YOLO once
    detector = YOLODetector(
        weights=r"C:\Dev\YOLO-V3\weights\yolov3.tf",
        classes=r"C:\Dev\YOLO-V3\data\labels\coco.names",
        tiny=False,
        size=416,
        num_classes=80
    )

    # 5. Run detection on all images
    results = detector.detect(all_images, output_dir=output_dir, save_images=True)

    # 6. Compare detection results to "No_Signs"/"With_Signs" & record CSV
    csv_path = os.path.join(output_dir, "sign_detections.csv")
    # add any relevant sign classes here:
    relevant_signs = {"stop sign", "traffic light"}  

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "filename", "expected_label", "time", "fog", "rain",
            "found_sign", "best_conf", "matched_expectation"
        ])
        
        for (img_path, detections) in results:
            filename = os.path.basename(img_path)
            info = parse_image_filename(filename)
            expected_label = info["expected_label"]  # "With_Signs", "No_Signs", or "UNKNOWN"

            # Build a string listing ALL detections, e.g. "person(0.81), stop sign(0.56)"
            detection_strs = [f"{cls_name}({conf:.2f})" for (cls_name, conf, box) in detections]
            all_detections_str = ", ".join(detection_strs)

            # See if we found a relevant sign
            found_sign = False
            best_conf = 0.0
            for (cls_name, conf, box) in detections:
                if cls_name in relevant_signs:
                    found_sign = True
                    if conf > best_conf:
                        best_conf = conf

            # Check if we matched the expectation
            matched = False
            if expected_label == "With_Signs" and found_sign:
                matched = True
            elif expected_label == "No_Signs" and not found_sign:
                matched = True

            writer.writerow([
                filename,
                expected_label,
                info["time"],
                info["fog"],
                info["rain"],
                found_sign,
                f"{best_conf:.2f}",
                matched
            ])

            # Print terminal output showing all detections, found_sign, best_conf, matched
            print(
                f"Processed {filename}, "
                f"detections=[{all_detections_str}], "
                f"found_sign={found_sign}, "
                f"best_conf={best_conf:.2f}, "
                f"matched={matched}"
            )

    print(f"[INFO] Done! CSV results at {csv_path}")

if __name__ == "__main__":
    main()
