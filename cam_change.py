import cv2
import requests
from ultralytics import YOLO
import time

class YOLOv8vMixController:
    def __init__(self, person_number_model_path, Left_Right_model_path, vmix_server_path):
        self.person_number_model = YOLO(person_number_model_path)
        self.Left_Right_model = YOLO(Left_Right_model_path)
        self.vmix_server = vmix_server_path
        self.person_detection_count = 0
    
    def detect_person_number(self, frame):
        results = self.person_number_model(frame)
        detected_objects = 0

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                num = result.names[int(box.cls[0])]
                confidence = box.conf[0]
                
                if confidence > 0.2:
                    detected_objects += 1
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, num, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    self.reset(num)

        self.detection_count = detected_objects
        if self.detection_count > 2:
            self.reset()

        return frame
    
    def left_right_player(self, apple):
        bananas = self.Left_Right_model(apple)
        for banana in bananas:
            for box in banana.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                direction = banana.names[int(box.cls[0])]
                confidence = box.conf[0]
                
                if confidence > 0.5:
                    cv2.rectangle(apple, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(apple, direction, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    self.LR_camera_control(direction)

    def LR_camera_control(self, direction):
        if direction == 'Left':
            requests.get(f"http://{self.vmix_server}:8088/api/?Function=QuickPlay&Input=3")
            time.sleep(3)
            requests.get(f"http://{self.vmix_server}:8088/api/?Function=QuickPlay&Input=1")
        elif direction == 'Right':
            requests.get(f"http://{self.vmix_server}:8088/api/?Function=QuickPlay&Input=2")
            time.sleep(3)
            requests.get(f"http://{self.vmix_server}:8088/api/?Function=QuickPlay&Input=1")



    def reset(self, num):
        if num <= 2:            
            # requests.get(f"http://{self.vmix_server}:8088/api/?Function=QuickPlay&Input=4")
            # time.sleep(3)
            # requests.get(f"http://{self.vmix_server}:8088/api/?Function=QuickPlay&Input=1")
            self.detection_count = 0

def main():
    Final= YOLOv8vMixController(
    person_number_model_path = "./playernumber/best.pt",
    Left_Right_model_path = "./number/v1/best.pt",
    vmix_server_path = "127.0.0.1"
    )

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = Final.left_right_player(frame)
        frame = Final.detect_person_number(frame)
        cv2.imshow("YOLOv8 Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
