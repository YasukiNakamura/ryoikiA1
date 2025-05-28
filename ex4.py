# https://ultralytics.com/images/bus.jpg 中の物体を抽出し，その領域を元の画像に描画せよ
import cv2
import torch
from ultralytics import YOLO

model = YOLO("yolov8x.pt")

results = model.predict("ex2.jpg", conf=0.1)

# 入力画像
img = results[0].orig_img

# 認識した物体領域を取得する．
boxes = results[0].boxes

max_area = 0
maxxy1 = None
maxxy2 = None

for box in boxes:
    # 物体領域の始点xy座標を得る．
    xy1 = box.data[0][0:2]
    # 物体領域の終点xy座標を得る．
    xy2 = box.data[0][2:4]
    width_height = xy2 - xy1
    area = width_height[0] * width_height[1]
    if area > max_area:
        max_area = area
        maxxy1 = xy1
        maxxy2 = xy2
    
cv2.rectangle(
    img,
    maxxy1.to(torch.int).tolist(),
    maxxy2.to(torch.int).tolist(),
    (0, 0, 255),
    thickness=3,
)

cv2.imshow("", img)
cv2.waitKey(0)
cv2.destroyAllWindows()