#請正常運作不要出問題
#Please don't have any problems
#何も問題はありませんのでご安心ください

import os
import cv2
import glob
import math
import shutil
import numpy as np
import json as js
import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO
from skimage.color import rgb2lab
from skimage.color import deltaE_cie76
from sklearn.cluster import KMeans
from gdAPI import *


def clean(Folder): #清空資料夾
    shutil.rmtree(Folder)
    os.mkdir(Folder)

def pick_folder(): #選取資料夾
    root = tk.Tk()
    root.withdraw()
    video_file_path = filedialog.askdirectory()
    print("影片資料夾位置:",video_file_path)
    return video_file_path

def rebuild_folder(): #清空重建classified_videos資料夾
    pathpath = "C:/Users/yoo48/Documents/code/bbyolo"
    outermost_path = os.path.join(pathpath, "classified_videos") #建最外層目錄
    if not os.path.isdir(outermost_path):
        os.mkdir(outermost_path)
    else:
        clean(outermost_path)

class ColorClassification(): #將背號依顏色分為兩類用(效果不好，沒用進去)
    def __init__(self):
        pass

    def calculate_luminance(self, color): #計算背號亮度
        r, g, b = color
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        return luminance
    
    def compare_lab_colors(self, color_list_1, color_list_2, threshold): #Lab顏色空間

        color_A1, color_A2 = color_list_1
        color_B1, color_B2 = color_list_2

        lab_color_A1 = rgb2lab([[[(color_A1[0] / 255, color_A1[1] / 255, color_A1[2] / 255)]]])[0][0]
        lab_color_A2 = rgb2lab([[[(color_A2[0] / 255, color_A2[1] / 255, color_A2[2] / 255)]]])[0][0]

        lab_color_B1 = rgb2lab([[[(color_B1[0] / 255, color_B1[1] / 255, color_B1[2] / 255)]]])[0][0]
        lab_color_B2 = rgb2lab([[[(color_B2[0] / 255, color_B2[1] / 255, color_B2[2] / 255)]]])[0][0]

        color_difference_1 = deltaE_cie76(lab_color_A1, lab_color_B1)
        color_difference_2 = deltaE_cie76(lab_color_A2, lab_color_B2)
        if color_difference_1 <= threshold and color_difference_2 <= threshold:
            return True  
        else:
            return False

    def color_gap(self,f_color1, f_color2, s_color1, s_color2): #計算顏色差距
        f_r1, f_g1, f_b1 = f_color1
        f_r2, f_g2, f_b2 = f_color2
        s_r1, s_g1, s_b1 = s_color1
        s_r2, s_g2, s_b2 = s_color2

        root1 = math.sqrt(((f_r1 - s_r1)**2) + ((f_g1 - s_g1)**2) + ((f_b1 - s_b1)**2) + ((f_r2 - s_r2)**2) + ((f_g2 - s_g2)**2) + ((f_b2 - s_b2)**2))
        root2 = math.sqrt(((f_r1 - s_r2)**2) + ((f_g1 - s_g2)**2) + ((f_b1 - s_b2)**2) + ((f_r2 - s_r1)**2) + ((f_g2 - s_g1)**2) + ((f_b2 - s_b1)**2))
        min(root1,root2)
    
    def find_uniform_color(self, img_path): #找出球衣顏色
        img = cv2.imread(img_path)

        pixels = img.reshape(-1, 3)
        k = 2
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(pixels)
        labels = kmeans.labels_
        cluster_centers = kmeans.cluster_centers_.astype(int)
        segmented_image = cluster_centers[labels].reshape(img.shape)
        target_depth = np.uint8
        converted_image = cv2.convertScaleAbs(segmented_image, alpha=1.0, beta=0)

        image_rgb = cv2.cvtColor(converted_image, cv2.COLOR_BGR2RGB)
        height, width, _ = converted_image.shape
        unique_colors = set()
        for x in range(width):
            for y in range(height):
                r, g, b = image_rgb[y, x]
                unique_colors.add((r, g, b))
        color1, color2 = unique_colors

        # return unique_colors #回傳圖片中的球衣及背號顏色

        luminance1 = self.calculate_luminance(color1)
        luminance2 = self.calculate_luminance(color2)

        allcolors = []

        if  luminance1 > luminance2:
            allcolors.append(color1)
            allcolors.append(color2)
            # uniform_color = color1
            # number_color = color2
        else:
            allcolors.append(color2)
            allcolors.append(color1)
            # uniform_color = color2
            # number_color = color1

        return allcolors

    def find_home_team_uniform(self, images_path): #找到有主隊球衣的圖片
        images_name = glob.glob(images_path + "/*.jpg")
        home_team_color = self.find_uniform_color(images_name[0]) #第一張圖片的球衣顏色為主隊顏色

        team_1_images = [images_name[0]]
        team_2_images = []
        # home_team_color_sure = False #主隊顏色未確定

        for img_path in images_name[1:]:
            uncategorized_color = self.find_uniform_color(img_path)
            similar = self.compare_lab_colors(home_team_color,uncategorized_color,10)

            if similar == True:
                team_1_images.append(img_path)
            else:
                team_2_images.append(img_path)

        return team_1_images, team_2_images

class Detector_and_upload(): #辨認背號數字並傳到雲端上
    def __init__(self, player_model_path, number_model_path, video_folder, json_output_path, folder_path, folder2):
        self.player_model = YOLO(player_model_path)
        self.number_model = YOLO(number_model_path)
        self.video_folder = video_folder
        self.json_output_path = json_output_path
        self.folder_path = folder_path
        self.folder2 = folder2
        
    def detect_playernumber(self, video_path):
        cap = cv2.VideoCapture(video_path)
        outputVideoBuffer = []
        max_frames = 180
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 編碼器：XVID
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        backNumber = ""
        lastNumber = [-1, 0]  # 記錄檢測到的背號和出現次數
        numberBuffer = -1
        frameRead = 0
        out = None
        mismatch_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            frameRead += 1
            if not ret:
                break
            if frameRead % 5 != 0:
                continue

            results = self.player_model(frame, show=True, save=False)
            if lastNumber[1] >= 5:
                outputVideoBuffer.append(frame)  # 如果已經檢測到背號，則添加幀到緩存

            for result in results:
                if result is None:
                    continue
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    if x1 >= 3 and y1 >= 3 and x2 >= 3 and y2 >= 3:
                        number_img = frame[y1-2:y2+2, x1-2:x2+2]  # 裁剪背號區域
                    else:
                        print("x1 x2 y1 y2 were wrong")
                        continue
                    backNumber = self.detect_number(number_img)  # 辨認背號數字
                    if backNumber != "":
                        if backNumber == lastNumber[0]:
                            mismatch_count = 0
                            lastNumber[1] += 1
                            print("lastNumber : ", lastNumber)
                            if lastNumber[1] >= 10: # 這個背號出現了連續5幀
                                if lastNumber[1] == 10:  # 第一次創建新文件
                                    if backNumber != numberBuffer:  # 跟 buffer 裡的背號不同,開新檔案
                                        print(lastNumber[0])
                                        outputFilename = "./classified_videos/" + f"{lastNumber[0]}_output.mp4"
                                        print("output filename : ",outputFilename)
                                        out = cv2.VideoWriter(outputFilename, fourcc, fps, frame_size)
                                        if len(outputVideoBuffer) > 50:
                                            for buffered_frame in outputVideoBuffer:
                                                out.write(buffered_frame)
                                            print("File written: ", outputFilename)
                                        else:
                                            print("outputVideoBuffer too short, len: ", len(outputVideoBuffer))
                                        outputVideoBuffer = []  # 清空緩存
                                        numberBuffer = backNumber  # 記錄背號
                                if out:
                                    out.write(frame)
                        else:  # 如果背號不同於上一幀
                            mismatch_count += 1
                            if mismatch_count >= 3:                                                                
                                lastNumber = [backNumber, 1]
                                mismatch_count = 0
                    break
            # if frameRead >= max_frames:
            #     break
        cap.release()
        if out:
            out.release()
        return lastNumber[0], outputVideoBuffer

    def detect_number(self, number_img): #辨認背號的數字
        results = self.number_model(number_img, show=True, save=False)
        labels = [] #儲存號碼
        coordinate = [] #儲存座標
        real_order = [] #儲存正確的背號順序
        real_playernumber = "" #真正背號

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int,box.xyxy[0])
                class_id = int(box.cls[0])
                label = self.number_model.names[class_id]
                if x1 > 0:
                    labels.append(label)
                    coordinate.append([x1, y1, x2, y2])
                    print("Find the number:", label)

        for i in range(len(coordinate)): #確定背號的排列順序
            x1, y1, x2, y2 = coordinate[i]
            label = labels[i]
            if len(coordinate) > 1:
                if coordinate[i][0] > coordinate[i-1][0]:
                    real_order.append(label)
                else:
                    real_order.insert(0, label)
            elif len(coordinate) == 1:
                real_order.append(label)

        if len(real_order) != 0: #把正確背號組合
            for n in range(len(real_order)):
                real_playernumber += str(real_order[n])

        print(real_playernumber)
        return real_playernumber
    
    def process_videos(self):  # 記下影片id對應的背號
        video_paths = glob.glob(f"{self.video_folder}/*.mp4")
        with open(self.json_output_path, "w") as jsonfile:
            for video_path in video_paths:
                outputVideoBuffer, detections = self.detect_playernumber(video_path)
                print("playernumber : ", playernumber)
                file_name, extension = os.path.splitext(video_path)
                video_id = file_name.split("\\")[-1]
                if outputVideoBuffer:
                    data = {
                        'video_id': video_id,
                        'detections': []
                    }
                    for detection in detections:
                        playernumber, detection_info = self.detect_number(detection['frame'])
                        data['detections'].append({
                            'playernumber': playernumber,
                            'detection_info': detection_info
                        })
                    js.dump(data, jsonfile)
                    jsonfile.write('\n')

    def assign_video(self): #將影片分到對應的資料夾
        video_paths = glob.glob(f"{self.folder_path}/*.mp4")
        with open(self.json_output_path, "r") as jsonfile:
            data = []
            for line in jsonfile:
                data.append(js.loads(line))
        for item in data:  # 读取 JSON 档案中的数据
            videoID = item["video_id"]
            detections = item["detections"]
            for detection in detections:
                playernumber = detection["playernumber"]
                detection_info = detection["detection_info"] 

                numdir_path = os.path.join(self.folder_path, playernumber)
                if not os.path.isdir(numdir_path):
                    os.mkdir(numdir_path)
                for folder_path in video_paths:
                    file_name, extension = os.path.splitext(folder_path)
                    videoid = os.path.basename(file_name)
                # print("numdir_path:", numdir_path)
                # cv2.waitKey(0)
                # camdir_path = numdir_path + "/" + videocam #分攝影機用，用不到就先標記起來
                # if not os.path.isdir(camdir_path):
                #     os.mkdir(camdir_path)
                if videoID == videoid:
                    old_video_path = os.path.join(self.folder_path, videoid + ".mp4")
                    # new_vodeo_path = os.path.join(camdir_path, videoid + ".mp4")
                    new_video_path = os.path.join(numdir_path, videoid + ".mp4")
                    print(f"Copying {old_video_path} to {new_video_path}") # 增加日誌
                    shutil.copyfile(old_video_path, new_video_path)

    def upload_video(self): #影片上傳雲端
        print("開始上傳影片")
        video_paths = glob.glob(f"{self.folder_path}/*.mp4")
        outermost_foldername = os.path.basename(self.folder_path)
        outermost_folderID = createFolder(outermost_foldername)
        with open(self.json_output_path, "r") as jsonfile:
            data = []
            for line in jsonfile:
                data.append(js.loads(line))
        for item in data:
            videoID = f"{item['playernumber']}_output"
            print("videoID :", videoID)
            playernumber = item["playernumber"]
            print("playernumber :", playernumber)
            number_folderID = createFolderInFolder(playernumber, outermost_folderID)
            for folder_path in video_paths: 
                file_name, extension = os.path.splitext(folder_path)
                videoid = os.path.basename(file_name)
                print("videoid :", videoid)
                if videoID == videoid:
                    retry_count = 3  # 設定重試次數
                    while retry_count > 0:
                        try: 
                            print(f"Uploading {folder_path} to cloud folder {number_folderID}") # 增加日誌
                            uploadFile(number_folderID, videoid, folder_path, 'video/x-msvideo')
                            break  # 成功上傳後跳出重試循環
                        except Exception as e:
                            print(f"Failed to upload {folder_path}: {e}")
                            retry_count -= 1  # 減少重試次數
                            if retry_count == 0:
                                print(f"Giving up on uploading {folder_path} after 3 attempts")
                            else:
                                print(f"Retrying upload for {folder_path}, {retry_count} attempts left")
        print("影片上傳完畢")

rebuild = rebuild_folder()

detector = Detector_and_upload(
    player_model_path = "./playernumber/best.pt",
    number_model_path = "./number/v1/best.pt",
    # video_folder = "GameVideo", #這個測試用比較快
    video_folder = pick_folder(), #最後要改成這種選取資料夾的形式
    json_output_path = "./playernumber_save.json",
    folder_path = "./classified_videos",
    folder2 = "C:/Users/yoo48/Documents/code/bbyolo"
)

detector.process_videos() #辨識號碼
detector.assign_video() #分類號碼
detector.upload_video() #上傳影片