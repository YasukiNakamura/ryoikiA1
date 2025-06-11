import cv2
import numpy as np
from ultralytics import YOLO
import torch

# モデルの読み込み
model = YOLO("yolov8x.pt")

# 画像読み込みと物体検出
results = model.predict("ex3.jpg", conf=0.2)
img = results[0].orig_img
boxes = results[0].boxes
names = model.names
class_ids = boxes.cls.cpu().numpy()

# 見やすさのためリサイズ
scale = 900 / img.shape[1]
img = cv2.resize(img, (900, int(img.shape[0] * scale)))

# チーム判定と描画
for i, box in enumerate(boxes):
    if names[int(class_ids[i])] != "person":
        continue

    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
    x1, y1, x2, y2 = [int(v * scale) for v in (x1, y1, x2, y2)]
    # 上半身だけ切り出す（中央30〜50％の高さ）
    person_h = y2 - y1
    person_w = x2 - x1

    # 胸あたりの縦領域（例：上から30〜60%）
    roi_y1 = y1 + int(person_h * 0.3)
    roi_y2 = y1 + int(person_h * 0.5)

    # 幅は中央寄せにする（任意）
    roi_x1 = x1 + int(person_w * 0.2)
    roi_x2 = x2 - int(person_w * 0.2)

    # 領域切り出し（ユニフォーム色判定用）
    roi = img[roi_y1:roi_y2, roi_x1:roi_x2]
   
    if roi.size == 0:
        continue

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # 色ごとのマスク作成
    yellow_mask = cv2.inRange(hsv, (20, 80, 50), (40, 255, 255))
    blue_mask = cv2.inRange(hsv, (120, 40, 60), (255, 255, 255))
    red_mask = cv2.inRange(hsv, (100, 40, 230), (255, 255, 255))

    # 色の割合
    area = roi.shape[0] * roi.shape[1]
    yellow_ratio = cv2.countNonZero(yellow_mask) / area
    blue_ratio = cv2.countNonZero(blue_mask) / area
    red_ratio = cv2.countNonZero(red_mask) / area

    # 判定と描画
    if yellow_ratio > 0.27:
        color = (0, 255, 255)  # 黄色チーム（ドルトムント）
    elif blue_ratio > 0.03 or red_ratio > 0.03:
        color = (255, 0, 255)  # 青赤チーム（バルセロナなど）
    else:
        continue  # GKや審判などは除外

    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

# 表示
cv2.imshow("Team Players", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
