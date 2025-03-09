import os
import argparse
import torch
from ultralytics import YOLO
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Function to parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="YOLO Inference for Traffic Sign Detection")
    parser.add_argument("--with_signs_dir", type=str, required=True, help="Path to folder containing "
                "images with traffic signs")
    parser.add_argument("--without_signs_dir", type=str, required=True, help="Path to folder "
                "containing images without traffic signs")
    parser.add_argument("--yolo_version", type=str, choices=["yolov5", "yolov8", "yolov11", 
                                                             "retrained"], 
                        required=True, help="Specify YOLO version (yolov5, yolov8, yolov11)")
    
    return parser.parse_args()

# Main function
def main():
    # Parse arguments
    args = parse_arguments()

    # Map YOLO version to appropriate model weight file
    model_map = {
        "yolov5": "yolov5s.pt",  # Small YOLOv5 model
        "yolov8": "yolov8n.pt",  # Nano YOLOv8 model
        "yolov11": "yolo11n.pt",
        "retrained": "runs/detect/train/weights/best.pt"  # YOLOv11 model
    }

    # Load the selected YOLO model
    model_path = model_map[args.yolo_version]
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    # Function to classify images
    def classify_images(folder):
        predictions = []
        for image_file in os.listdir(folder):
            image_path = os.path.join(folder, image_file)
            results = model(image_path)  # Run inference
            has_sign = any(result.boxes.shape[0] > 0 for result in results)  # Binary classification
            predictions.append(int(has_sign))
        return predictions

    # Count total images
    with_signs_count = len(os.listdir(args.with_signs_dir))
    without_signs_count = len(os.listdir(args.without_signs_dir))

    # Run inference on both categories
    y_pred_with = classify_images(args.with_signs_dir)
    y_pred_without = classify_images(args.without_signs_dir)

    # Ground truth labels
    y_true = [1] * with_signs_count + [0] * without_signs_count
    y_pred = y_pred_with + y_pred_without

    # Compute classification metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # Report results
    print(f"Total images with signs: {with_signs_count}")
    print(f"Total images without signs: {without_signs_count}")
    print(f"Inferenced images with signs: {sum(y_pred)}")
    print(f"Inferenced images without signs: {len(y_pred) - sum(y_pred)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

# Run the script
if __name__ == "__main__":
    main()
