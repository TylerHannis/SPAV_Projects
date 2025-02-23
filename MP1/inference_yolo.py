from ultralytics import YOLO
import os

def main():
    # load the v3 version of YOLO's model weights
    # TODO: add an argument that can adjust the yolo version
    

    
    model = YOLO("yolov3u.pt")

    model.info()

    # low light, part of a sign, just a test
    # we need a method for iterating through the directory of images
    # running the results on them
    print(os.getcwd())
    results = model("Trovinger/With Signs/time(6)_fog(0)_rain(0)/819_image.jpg")
    

if __name__ == "__main__":
    main()