from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# モデル読み込み
model = YOLO('soccer2.pt')

# 画像読み込みと推論
img_path = 'ex3.jpg'
results = model(img_path, conf=0.3)[0]

# 結果情報取得
boxes = results.boxes
class_ids = boxes.cls.cpu().numpy().astype(int)
confidences = boxes.conf.cpu().numpy()
xyxy = boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]

# 枠色定義（チームごとに）
colors = {
    0: (255, 0, 0),    # player_A：青枠
    1: (0, 0, 255),    # player_B：赤枠
}

# 元画像読み込みとリサイズ（見やすくする）
img = cv2.imread(img_path)
scale = 800 / max(img.shape[:2])
img = cv2.resize(img, None, fx=scale, fy=scale)
resized_shape = img.shape[:2]

# 座標もリサイズに合わせて調整
xyxy *= scale

# クラスラベル
class_names = ['redblue', 'yellow', 'GK', 'ref']

# 描画
for box, class_id, conf in zip(xyxy, class_ids, confidences):
    if class_id in [2, 3]:  # GK, referee はスキップ
        continue
    x1, y1, x2, y2 = map(int, box)
    color = colors.get(class_id, (0, 255, 0))  # 不明クラスは緑
    label = f"{class_names[class_id]} ({conf:.2f})"
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

# 結果画像保存または表示
cv2.imwrite('result.jpg', img)
cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
