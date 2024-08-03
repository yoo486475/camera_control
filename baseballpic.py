# 導入yolo
from ultralytics import YOLO 


# 使用預先訓練好的模型
model = YOLO("yolov8n.pt")

#  設定檢索物品大小，並進行檢索
result = model.predict(source="./test.png",\
    model="predict",
    save_txt=True,
    save=True)