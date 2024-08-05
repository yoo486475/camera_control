import config
import cv2
import requests
from ultralytics import YOLO
import time

# Load YOLO models
model_count_people = YOLO("./yolov8n.pt")
model_hitter_side = YOLO("./yolov8n.pt")

# Read the video
video_path = "./video/101943--av-1.mp4"
camVideo = cv2.VideoCapture(video_path)

if not camVideo.isOpened():
    print(f"Error: Could not open video file {video_path}")
else:
    people_count = 0  
    has_run = False 

    while camVideo.isOpened():
        ret, CFImage = camVideo.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        cf_results = model_count_people(CFImage, show = True)[0]
        class_names = cf_results.names
        detected_people_count = 0 

        for box in cf_results.boxes:
            class_id = int(box.cls[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            if class_names[class_id] == 'person':
                if (x1 > config.bigBoxLeft and y1 > config.bigBoxTop and x2 < config.bigBoxRight and y2 < config.bigBoxBottom):
                    detected_people_count += 1
                    print("detected_people_count : ", detected_people_count)


        if has_run == True and detected_people_count == 3 or has_run == True and detected_people_count == 2: # 如果在執行過has_run後偵測人數為2或3, reset
            people_count = 0
            has_run = False  
            print("reset")
        elif detected_people_count == 5:
            pass
        elif detected_people_count == 4:
            people_count = 4
            if not has_run:
                print("Four people")
                hitter_sides = model_hitter_side(CFImage, show=True)[0]
                for hitter_box in hitter_sides.boxes:
                    hx1, hy1, hx2, hy2 = hitter_box.xyxy[0].tolist()
                    if class_names[class_id] == 'person':
                        if (hx1 > config.LeftHitterLeftside and hx2 < config.LeftHitterRightside):
                            print("Left hitter")
                            time.sleep(1)
                            # requests.get(f"http://127.0.0.1:8088/api/?Function=QuickPlay&Input=3")
                            # time.sleep(3)
                            # requests.get(f"http://127.0.0.1:8088/api/?Function=QuickPlay&Input=1")
                            break
                        elif (hx1 > config.RightHitterLeftside and hx2 < config.RightHitterRightside):
                            print("Right hitter")
                            time.sleep(1)
                            # requests.get(f"http://127.0.0.1:8088/api/?Function=QuickPlay&Input=2")
                            # time.sleep(3)
                            # requests.get(f"http://127.0.0.1:8088/api/?Function=QuickPlay&Input=1")
                            break
                has_run = True  # Set the flag to indicate the block has been executed

    camVideo.release()
    cv2.destroyAllWindows()
