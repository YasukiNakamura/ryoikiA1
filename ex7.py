from ultralytics import YOLO
import cv2
from collections import Counter

# 画像とモデルのパス
image_path = "ex4.jpg"
model_path = "othello.pt"

# モデル読み込み
model = YOLO(model_path)


# 推論
results = model(image_path)

# 元画像の読み込み
image = cv2.imread(image_path)
height, width, _ = image.shape

# ----------------------------
# 盤面の領域を定義（経験則）
# ※この領域にある石だけを数える
# ----------------------------
board_x_min = int(width * 0.05)
board_x_max = int(width * 0.95)
board_y_min = int(height * 0.10)
board_y_max = int(height * 0.95)

# 検出結果取得
boxes = results[0].boxes
labels = []
class_names = model.names

# 各検出に対して、盤上かどうかチェック
for box, cls_id in zip(boxes.xyxy, boxes.cls):
    x1, y1, x2, y2 = box.tolist()
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    if board_x_min <= cx <= board_x_max and board_y_min <= cy <= board_y_max:
        label = class_names[int(cls_id)]
        labels.append(label)

# カウント
counts = Counter(labels)

# 結果表示
print(f"盤上の白の石の数: {counts.get('white', 0)}")
print(f"盤上の黒の石の数: {counts.get('black', 0)}")
