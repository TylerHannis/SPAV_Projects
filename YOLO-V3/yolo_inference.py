import os
import time
import tensorflow as tf
import cv2
import numpy as np

from yolov3.models import YoloV3, YoloV3Tiny
from yolov3.dataset import transform_images
from yolov3.utils import draw_outputs


class YOLODetector:
    """
    A class to load a YoloV3 (or YoloV3Tiny) model once and run inference
    on multiple images, avoiding repeated weight loading. It will save 
    annotated images only when there are actual detections, and maintain 
    the original filename.
    """

    def __init__(
        self,
        weights="./weights/yolov3.tf",
        classes="./data/labels/coco.names",
        tiny=False,
        size=416,
        num_classes=80,
        gpu_memory_growth=True
    ):
        """
        :param weights: Path to the *.tf weights file
        :param classes: Path to the *.names file listing class labels
        :param tiny: True for YoloV3Tiny, False for full YoloV3
        :param size: Image size (416 is standard for YOLOv3)
        :param num_classes: Number of classes in the model (80 for COCO)
        :param gpu_memory_growth: If True, sets memory growth on the first GPU
        """
        self.size = size
        self.num_classes = num_classes

        # Optionally enable GPU memory growth if a GPU is present
        if gpu_memory_growth:
            physical_devices = tf.config.experimental.list_physical_devices('GPU')
            if len(physical_devices) > 0:
                tf.config.experimental.set_memory_growth(physical_devices[0], True)

        # Build the model
        if tiny:
            self.yolo = YoloV3Tiny(classes=num_classes)
        else:
            self.yolo = YoloV3(classes=num_classes)

        # Load the weights
        self.yolo.load_weights(weights).expect_partial()
        print(f"[INFO] Weights loaded from {weights}")

        # Load class names
        with open(classes, 'r') as f:
            self.class_names = [c.strip() for c in f.readlines()]
        print(f"[INFO] Classes loaded from {classes}")

    def detect(self, image_paths, output_dir="./detections", save_images=True):
        """
        Runs YOLO inference on all images in image_paths.

        :param image_paths: List of absolute paths to images (*.jpg, etc.)
        :param output_dir: Folder where annotated images will be saved
        :param save_images: If True, only save images that have >=1 detections 
                            (keeping original filenames).
        :return: A list of (image_path, detections) where:
                 detections is a list of (class_name, confidence, [ymin, xmin, ymax, xmax]).
        """
        os.makedirs(output_dir, exist_ok=True)

        results = []
        for img_path in image_paths:
            # Load the image file
            raw_img = tf.image.decode_image(open(img_path, 'rb').read(), channels=3)

            # Preprocess for YOLO
            img_input = tf.expand_dims(raw_img, 0)
            img_input = transform_images(img_input, self.size)

            # Run inference
            t1 = time.time()
            boxes, scores, classes, nums = self.yolo(img_input)
            t2 = time.time()
            print(f"[INFO] Processed {img_path} in {t2 - t1:.2f}s")

            # Collect detection info
            detection_list = []
            for i in range(nums[0]):  # YOLO returns up to 'nums[0]' detections
                cls_index = int(classes[0][i])
                cls_name = self.class_names[cls_index]
                confidence = float(scores[0][i])
                # boxes are [ymin, xmin, ymax, xmax] in normalized coords
                box = [float(x) for x in boxes[0][i]]
                detection_list.append((cls_name, confidence, box))

            # Store results (even if empty)
            results.append((img_path, detection_list))

            # If we have at least one detection and user wants to save images:
            if save_images and len(detection_list) > 0:
                # Keep the same filename
                original_filename = os.path.basename(img_path)
                out_img_path = os.path.join(output_dir, original_filename)

                # Convert from TensorFlow's RGB to OpenCV's BGR
                bgr_img = cv2.cvtColor(raw_img.numpy(), cv2.COLOR_RGB2BGR)

                # Draw bounding boxes on the image
                bgr_img = draw_outputs(bgr_img, (boxes, scores, classes, nums), self.class_names)

                # Save the annotated image
                cv2.imwrite(out_img_path, bgr_img)
                print(f"[INFO] Saved detection image: {out_img_path}")

        return results
