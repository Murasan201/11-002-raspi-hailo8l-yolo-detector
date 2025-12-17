# ライブラリAPI リファレンス

**raspi_hailo8l_yolo** モジュールを外部プロジェクトからライブラリとして使用するためのAPIドキュメントです。

## インストール

環境構築については [SETUP_GUIDE.md](./SETUP_GUIDE.md) を参照してください。

```bash
# リポジトリをクローン
git clone https://github.com/Murasan201/11-002-raspi-hailo8l-yolo-detector.git

# 仮想環境の作成と有効化
python3 -m venv .venv --system-site-packages
source .venv/bin/activate

# 依存パッケージのインストール
pip install -r requirements.txt
```

## クイックスタート

```python
from raspi_hailo8l_yolo import (
    YOLODetector, CameraManager, draw_detections,
    is_hailo_available, COCO_CLASSES
)
import cv2

# Hailo デバイスの確認
if not is_hailo_available():
    print("Hailo-8L が利用できません")
    exit(1)

# 検出器の初期化
detector = YOLODetector("models/yolov8s_h8l.hef")

# 画像から物体検出
image = cv2.imread("test.jpg")
detections = detector.detect(image)

# 結果の描画
result = draw_detections(image, detections)
cv2.imwrite("result.jpg", result)

# 検出結果の表示
for det in detections:
    print(f"{det['class_name']}: {det['confidence']:.2f}")
```

---

## 公開API一覧

### クラス
| クラス名 | 説明 |
|---------|------|
| `YOLODetector` | YOLO物体検出器（Hailo-8L対応） |
| `CameraManager` | カメラ入力管理（Picamera2/USB両対応） |
| `DetectionLogger` | 検出結果のCSVログ管理 |

### ユーティリティ関数
| 関数名 | 説明 |
|-------|------|
| `draw_detections()` | 画像に検出結果を描画 |
| `draw_info()` | 画像にFPS等の情報を描画 |
| `parse_resolution()` | 解像度文字列をパース |

### 可用性チェック関数
| 関数名 | 説明 |
|-------|------|
| `is_hailo_available()` | HailoRT SDKが利用可能か確認 |
| `is_picamera_available()` | Picamera2が利用可能か確認 |

### ロガー設定
| 関数名 | 説明 |
|-------|------|
| `setup_logging()` | ロギングの設定 |
| `get_logger()` | ロガーインスタンスの取得 |

### 定数
| 定数名 | 説明 |
|-------|------|
| `COCO_CLASSES` | COCO 80クラス名リスト |

---

## クラス詳細

### YOLODetector

YOLO物体検出器クラス。Hailo-8L AIアクセラレータを使用した高速推論を実行します。

#### コンストラクタ

```python
YOLODetector(
    model_path: str,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    target_classes: Optional[List[str]] = None
)
```

| パラメータ | 型 | 説明 |
|-----------|-----|------|
| `model_path` | str | HEFモデルファイルのパス |
| `conf_threshold` | float | 信頼度閾値（0.0-1.0） |
| `iou_threshold` | float | IoU閾値（NMS用） |
| `target_classes` | List[str] | 検出対象クラス（Noneで全クラス） |

#### 使用例

```python
# 基本的な使用（全クラス検出）
detector = YOLODetector("models/yolov8s_h8l.hef")

# 特定クラスのみ検出
detector = YOLODetector(
    "models/yolov8s_h8l.hef",
    conf_threshold=0.5,
    target_classes=['person', 'car', 'dog']
)

# 画像から検出
image = cv2.imread("photo.jpg")
detections = detector.detect(image)
```

#### メソッド

##### detect(image)
画像から物体を検出します。

```python
detections = detector.detect(image)
```

**引数**:
- `image` (np.ndarray): 入力画像（BGRフォーマット）

**戻り値**: List[Dict] - 検出結果のリスト
```python
[
    {
        'bbox': [x1, y1, x2, y2],  # バウンディングボックス座標
        'confidence': 0.85,        # 信頼度（0.0-1.0）
        'class_id': 0,             # クラスID
        'class_name': 'person'     # クラス名
    },
    ...
]
```

##### set_target_classes(class_names)
検出対象クラスを動的に変更します。

```python
# 人と車のみ検出
detector.set_target_classes(['person', 'car'])

# 全クラス検出に戻す
detector.set_target_classes(None)
```

##### get_available_classes()
使用可能な全クラス名を取得します。

```python
classes = detector.get_available_classes()
# ['person', 'bicycle', 'car', ...]
```

##### is_class_targeted(class_name)
指定クラスが検出対象かを確認します。

```python
if detector.is_class_targeted('person'):
    print("person は検出対象です")
```

##### is_initialized
Hailoデバイスが初期化済みかを確認するプロパティ。

```python
if detector.is_initialized:
    print("Hailo デバイス初期化済み")
```

---

### CameraManager

カメラ入力を管理するクラス。Picamera2（Camera Module V3）とUSB Webカメラの両方に対応。

#### コンストラクタ

```python
CameraManager(
    resolution: Tuple[int, int] = (1280, 720),
    device_id: int = 0,
    flip_vertical: bool = False,
    use_picamera: Optional[bool] = None
)
```

| パラメータ | 型 | 説明 |
|-----------|-----|------|
| `resolution` | Tuple[int, int] | 解像度 (width, height) |
| `device_id` | int | カメラデバイスID |
| `flip_vertical` | bool | 上下反転 |
| `use_picamera` | bool | Picamera2使用（Noneで自動検出） |

#### 使用例

```python
# 自動検出（Picamera2優先）
camera = CameraManager(resolution=(1280, 720))

# USB Webカメラを強制使用
camera = CameraManager(use_picamera=False)

# 上下反転
camera = CameraManager(flip_vertical=True)

# フレーム取得
frame = camera.read_frame()

# リソース解放
camera.release()
```

#### コンテキストマネージャー対応

```python
with CameraManager(resolution=(1280, 720)) as camera:
    while True:
        frame = camera.read_frame()
        if frame is None:
            break
        # フレーム処理...
# 自動的にリソース解放
```

#### メソッド

##### read_frame()
カメラからフレームを取得します。

```python
frame = camera.read_frame()  # np.ndarray (BGR) or None
```

##### release()
カメラリソースを解放します。

```python
camera.release()
```

---

### DetectionLogger

検出結果をCSVファイルに記録するクラス。

#### コンストラクタ

```python
DetectionLogger(log_dir: str = "logs")
```

#### 使用例

```python
logger = DetectionLogger("logs")

# 検出結果を記録
logger.log_detections(frame_id=0, detections=detections)
```

#### 出力フォーマット

```csv
timestamp,frame_id,class_name,confidence,x1,y1,x2,y2
2024-12-17T14:30:52.123456,1,person,0.85,100,150,300,400
```

---

## ユーティリティ関数

### draw_detections()

画像に検出結果（バウンディングボックス、ラベル）を描画します。

```python
result = draw_detections(
    image: np.ndarray,
    detections: List[Dict],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray
```

#### 使用例

```python
result = draw_detections(image, detections)
cv2.imshow("Result", result)

# 色と線の太さを変更
result = draw_detections(image, detections, color=(255, 0, 0), thickness=3)
```

---

### draw_info()

画像にパフォーマンス情報（FPS、解像度、推論時間）を描画します。

```python
result = draw_info(
    image: np.ndarray,
    fps: float,
    resolution: Tuple[int, int],
    inference_time: float
) -> np.ndarray
```

#### 使用例

```python
result = draw_info(image, fps=25.5, resolution=(1280, 720), inference_time=45.2)
```

---

### is_hailo_available()

HailoRT SDKが利用可能かを確認します。

```python
if is_hailo_available():
    detector = YOLODetector("model.hef")
else:
    print("Hailo-8L が利用できません")
```

---

### is_picamera_available()

Picamera2が利用可能かを確認します。

```python
if is_picamera_available():
    camera = CameraManager(use_picamera=True)
else:
    camera = CameraManager(use_picamera=False)  # USB Webカメラ
```

---

### setup_logging()

ログ出力を有効化します。ライブラリ使用時はデフォルトでログ出力が無効です。

```python
import logging
from raspi_hailo8l_yolo import setup_logging

# 基本的な使用
setup_logging()

# デバッグレベルで詳細出力
setup_logging(level=logging.DEBUG)

# カスタムフォーマット
setup_logging(format_string='%(levelname)s: %(message)s')
```

---

### get_logger()

モジュールのロガーインスタンスを取得します。

```python
logger = get_logger()
logger.addHandler(my_custom_handler)
```

---

## 定数

### COCO_CLASSES

COCO データセットの80クラス名リスト。

```python
from raspi_hailo8l_yolo import COCO_CLASSES

print(COCO_CLASSES)
# ['person', 'bicycle', 'car', 'motorcycle', 'airplane', ...]

print(len(COCO_CLASSES))  # 80
```

---

## 実用例

### 例1: 基本的な物体検出

```python
from raspi_hailo8l_yolo import YOLODetector, draw_detections
import cv2

detector = YOLODetector("models/yolov8s_h8l.hef")

image = cv2.imread("photo.jpg")
detections = detector.detect(image)

result = draw_detections(image, detections)
cv2.imwrite("result.jpg", result)

print(f"検出数: {len(detections)}")
for det in detections:
    print(f"  {det['class_name']}: {det['confidence']:.2%}")
```

### 例2: リアルタイムカメラ処理

```python
from raspi_hailo8l_yolo import (
    YOLODetector, CameraManager, draw_detections, draw_info
)
import cv2
import time

detector = YOLODetector("models/yolov8s_h8l.hef", target_classes=['person'])

with CameraManager(resolution=(1280, 720)) as camera:
    fps = 0
    prev_time = time.time()

    while True:
        frame = camera.read_frame()
        if frame is None:
            break

        # 物体検出
        start = time.time()
        detections = detector.detect(frame)
        inference_time = (time.time() - start) * 1000

        # FPS計算
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        # 描画
        result = draw_detections(frame, detections)
        result = draw_info(result, fps, (1280, 720), inference_time)

        cv2.imshow("Detection", result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
```

### 例3: 特定クラスのフィルタリング

```python
from raspi_hailo8l_yolo import YOLODetector, COCO_CLASSES

# 動物のみ検出
animals = ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']
detector = YOLODetector("models/yolov8s_h8l.hef", target_classes=animals)

# 検出実行...

# 検出対象を動的に変更（乗り物のみ）
vehicles = ['bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat']
detector.set_target_classes(vehicles)
```

### 例4: 環境チェック付き初期化

```python
from raspi_hailo8l_yolo import (
    YOLODetector, CameraManager,
    is_hailo_available, is_picamera_available,
    setup_logging
)
import logging

# ログ出力を有効化
setup_logging(level=logging.INFO)

# 環境チェック
if not is_hailo_available():
    print("エラー: Hailo-8L が検出されません")
    print("確認: hailortcli fw-control identify")
    exit(1)

print(f"Picamera2: {'利用可能' if is_picamera_available() else '利用不可'}")

# 初期化
detector = YOLODetector("models/yolov8s_h8l.hef")
camera = CameraManager()

print("初期化完了")
```

---

## エラーハンドリング

### 想定される例外

| 例外 | 発生条件 | 対処方法 |
|-----|---------|---------|
| `FileNotFoundError` | モデルファイルが見つからない | パスを確認、シンボリックリンクを作成 |
| `RuntimeError` | Hailoデバイス初期化失敗 | `hailortcli fw-control identify` で確認 |
| `ValueError` | 無効なクラス名 | `COCO_CLASSES` で利用可能なクラスを確認 |

### 例外処理の例

```python
from raspi_hailo8l_yolo import YOLODetector, is_hailo_available

try:
    if not is_hailo_available():
        raise RuntimeError("Hailo-8L が利用できません")

    detector = YOLODetector("models/yolov8s_h8l.hef", target_classes=['invalid_class'])

except FileNotFoundError as e:
    print(f"モデルエラー: {e}")
except ValueError as e:
    print(f"クラス名エラー: {e}")
except RuntimeError as e:
    print(f"デバイスエラー: {e}")
```

---

## 関連ドキュメント

- [SETUP_GUIDE.md](./SETUP_GUIDE.md) - 環境構築ガイド
- [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) - トラブルシューティング
- [11_002_raspi_hailo_8_l_yolo_detector.md](./11_002_raspi_hailo_8_l_yolo_detector.md) - 要件定義書
