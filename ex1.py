from ultralytics import YOLO
import cv2
import torch

model = YOLO("yolo11x-pose.pt")

results = model("ex1.jpg")

nodes = results[0].keypoints.data[0][:, :2]
keypoints = results[0].keypoints



#骨格のリンク
links = [
    [5, 7],
    [6, 8],
    [7, 9],
    [8, 10],
    [11, 13],
    [12, 14],
    [13, 15],
    [14, 16],
    [5, 11],
    [6, 12],
    [5, 6],
    [11, 12]
]

print(keypoints.data)

# 入力画像
img = results[0].orig_img




for n1, n2 in links:
    # 誤認識のリンクを描画しない．
    if nodes[n1][0] * nodes[n1][1] * nodes[n2][0] * nodes[n2][1] == 0:
        continue

    cv2.line(
        img,
        # 2つの座標を整数化し，テンソルからリストにする．
        nodes[n1].to(torch.int).tolist(),
        nodes[n2].to(torch.int).tolist(),
        (0, 0, 255),
        thickness=4,
    )

for point in keypoints.data[0][5:]:   
    x,y,conf = point
    cv2.circle(
        img,
        (int(x),int(y)),
        3,
        (0,255,255),
        -1
    )

cv2.imshow("", img)
cv2.waitKey(0)
cv2.destroyAllWindows()