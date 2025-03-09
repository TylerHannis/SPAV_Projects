import os
import argparse
from ultralytics import YOLO

# Function to parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Train YOLOv11 on Stop and Yield Sign Dataset")
    parser.add_argument("--train_dir", type=str, required=True, help="Path to training "
                        "dataset (images and labels)")
    parser.add_argument("--val_dir", type=str, required=True, help="Path to validation "
                        "dataset (images and labels)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--img_size", type=int, default=640, help="Image size for training")

    return parser.parse_args()

# Function to create data.yaml
def create_data_yaml(train_path, val_path):
    yaml_content = f"""
train: {train_path}/images
val: {val_path}/images

nc: 2
names: ["stop_sign", "Unyielding"]
"""
    yaml_path = os.path.join(train_path, "data.yaml")
    with open(yaml_path, "w") as file:
        file.write(yaml_content)
    
    return yaml_path

# Main function
def main():
    args = parse_arguments()

    # Define YOLOv11 model path
    model_path = "yolo11n.pt"

    # Create data.yaml
    yaml_path = create_data_yaml(args.train_dir, args.val_dir)

    # Load YOLOv11 model
    print(f"Loading YOLOv11 model: {model_path}")
    model = YOLO(model_path)  # Load YOLOv11 model

    # Train the model
    print("Starting YOLOv11 training...")
    model.train(
        data=yaml_path,
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.img_size
    )

    print("Training completed successfully!")

# Run the script
if __name__ == "__main__":
    main()
