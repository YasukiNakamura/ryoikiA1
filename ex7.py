from ultralytics import YOLO
import cv2

# モデル読み込み
model = YOLO('othello.pt')

# 画像読み込み
image_path = 'ex4.jpg'
conf_threshold = 0.99  # 盤外の誤検出を防ぐため信頼度を上げる（例: 0.99）

# 推論実行
results = model(image_path, conf=conf_threshold)
detections = results[0]

# カウント用変数
white_count = 0
black_count = 0

# 各検出結果に対してラベルを判定
for box in detections.boxes:
    cls_id = int(box.cls)
    conf = float(box.conf)
    if conf < conf_threshold:
        continue  # 信頼度が低い検出は無視

    if cls_id == 0:
        white_count += 1
    elif cls_id == 1:
        black_count += 1

# 結果出力
print(f"白石の数: {white_count}")
print(f"黒石の数: {black_count}")
