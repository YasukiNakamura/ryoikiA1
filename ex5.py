import cv2
import torch
from ultralytics import YOLO
import numpy as np

# YOLOモデルの読み込み
model = YOLO("yolov8x.pt")

# 推論
results = model.predict("ex2.jpg", conf=0.1)

# 入力画像
img = results[0].orig_img

# 認識された物体のボックス
boxes = results[0].boxes
class_ids = results[0].boxes.cls.cpu().numpy()  # クラスID取得
names = model.names  # クラス名辞書

for i, box in enumerate(boxes):
    # クラス名が "person" のみ対象
    if names[int(class_ids[i])] != "person":
        continue

    # ボックスの座標
    xy1 = box.data[0][0:2].to(torch.int).tolist()
    xy2 = box.data[0][2:4].to(torch.int).tolist()

    # 領域を切り出し
    x1, y1 = xy1
    x2, y2 = xy2
    person_roi = img[y1:y2, x1:x2]

    # 色空間をBGR→HSVへ変換
    hsv = cv2.cvtColor(person_roi, cv2.COLOR_BGR2HSV)

    # 青色の範囲（HSVで定義）
    lower_blue = np.array([100, 100, 50])
    upper_blue = np.array([130, 255, 255])

    # 青色領域をマスク
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # 青色が一定以上あれば人物を日本代表とみなす
    blue_pixels = cv2.countNonZero(mask)
    total_pixels = mask.shape[0] * mask.shape[1]

    if blue_pixels / total_pixels > 0.01:  # 青色が5%以上
        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),  # 赤枠
            thickness=3
        )

# 表示
cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
