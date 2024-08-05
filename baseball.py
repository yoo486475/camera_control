import config  # Your config file
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
    detected_count = {'count_time': 0, 'count_reset': 0, 'count_left': 0, 'count_right': 0}
    has_run = False

    while camVideo.isOpened():
        ret, CFImage = camVideo.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        cf_results = model_count_people(CFImage, show=True)[0]
        class_names = cf_results.names
        detected_people_count = 0

        for box in cf_results.boxes:
            class_id = int(box.cls[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            if class_names[class_id] == 'person':
                if (x1 > config.bigBoxLeft and y1 > config.bigBoxTop and x2 < config.bigBoxRight and y2 < config.bigBoxBottom):
                    detected_people_count += 1
                    print("detected_people_count:", detected_people_count)

        if detected_people_count < 4:  # If the block has run and detected people count is less than 4, reset
            detected_count['count_time'] = 0
            if has_run == True:
                detected_count['count_reset'] += 1
                if detected_count['count_reset'] >= 10:  # If count_reset is 5 or more, reset
                    has_run = False
                    detected_count['count_reset'] = 0
                    print("reset")

        if detected_people_count >= 4:
            if detected_people_count == 4:
                detected_count['count_reset'] = 0
                if has_run == False:
                    detected_count['count_time'] += 1
                    print("count_time is:", detected_count['count_time'])
                    if detected_count['count_time'] == 3:
                        print("Four people")
                        hitter_sides = model_hitter_side(CFImage, show=True)[0]
                        for hitter_box in hitter_sides.boxes:
                            hx1, hy1, hx2, hy2 = hitter_box.xyxy[0].tolist()
                            if class_names[class_id] == 'person':
                                if (hx1 > config.LeftHitterLeftside and hx2 < config.LeftHitterRightside):
                                    detected_count['count_left'] += 1
                                    detected_count['count_right'] = 0
                                    if detected_count['count_left'] == 2:
                                        print("Left hitter")
                                        time.sleep(1)
                                        # requests.get(f"http://127.0.0.1:8088/api/?Function=QuickPlay&Input=3")
                                        print("change to camera 3")
                                        time.sleep(3)
                                        # requests.get(f"http://127.0.0.1:8088/api/?Function=QuickPlay&Input=1")
                                        print("change to camera 1")
                                        detected_count['count_left' and 'count_time'] = 0
                                        has_run = True  # Set the flag to indicate the block has been executed
                                        print("count_time is reset")
                                        print("has_run is change to True")
                                    break
                                if (hx1 > config.RightHitterLeftside and hx2 < config.RightHitterRightside):
                                    detected_count['count_right'] += 1
                                    detected_count['count_left'] = 0
                                    if detected_count['count_right'] == 2:
                                        print("Right hitter")
                                        time.sleep(1)
                                        # requests.get(f"http://127.0.0.1:8088/api/?Function=QuickPlay&Input=2")
                                        print("change to camera 2")
                                        time.sleep(3)
                                        # requests.get(f"http://127.0.0.1:8088/api/?Function=QuickPlay&Input=1")
                                        print("change to camera 1")
                                        detected_count['count_right' and 'count_time'] = 0
                                        has_run = True  # Set the flag to indicate the block has been executed
                                        print("count_time is reset")
                                        print("has_run is change to True")
                                    break                
    camVideo.release()
    cv2.destroyAllWindows()