# raspi_hailo8l_yolo.py 機能仕様書

## 概要
本ドキュメントは `raspi_hailo8l_yolo.py` の機能仕様を定義する。

## ドキュメント情報
| 項目 | 内容 |
|-----|------|
| 対象ファイル | `raspi_hailo8l_yolo.py` |
| プロジェクト | 11-002-raspi-hailo8l-yolo-detector |
| 作成日 | 2025-12-29 |
| ステータス | 作成中 |

---

## 外部参照情報

本モジュールは外部プロジェクトからライブラリとして参照されている。
API変更時は後方互換性を維持すること。

### 参照プロジェクト一覧

| プロジェクト | 参照元ファイル |
|------------|--------------|
| 12-002-pet-monitoring-yolov8 | `camera_tracker.py` |

### 使用されているAPI

#### 12-002-pet-monitoring-yolov8 / camera_tracker.py

**インポート文:**
```python
# ファイル先頭（37行目）
from raspi_hailo8l_yolo import YOLODetector, CameraManager, draw_detections

# 関数内での遅延インポート（1327行目）
from raspi_hailo8l_yolo import COCO_CLASSES
```

**使用API詳細:**

| API | 種別 | 用途 |
|-----|------|------|
| `YOLODetector` | クラス | ペット（犬・猫）の物体検出 |
| `CameraManager` | クラス | カメラフレームの取得・管理 |
| `draw_detections` | 関数 | 検出結果のバウンディングボックス描画 |
| `COCO_CLASSES` | 定数 | 利用可能なクラス一覧の表示 |

**YOLODetector の使用例:**
```python
# インスタンス生成（432行目）
self.detector = YOLODetector(
    model_path=model_path,
    conf_threshold=conf_threshold,
    target_classes=target_classes
)

# 検出実行（876行目、1117行目）
detections = self.detector.detect(frame)
```

**CameraManager の使用例:**
```python
# インスタンス生成（446行目）
self.camera = CameraManager(
    resolution=(frame_width, frame_height),
    device_id=camera_index,
    flip_vertical=flip_vertical
)

# フレーム読み込み（645行目、779行目、1111行目）
frame = self.camera.read_frame()

# カメラ終了処理（1215行目）
self.camera.release()
```

**draw_detections の使用例:**
```python
# バウンディングボックス描画（1017行目、1120行目）
display_frame = draw_detections(frame, detections)
```

**COCO_CLASSES の使用例:**
```python
# クラス一覧表示（1330行目）
for i, name in enumerate(COCO_CLASSES):
    print(f"  {i:2d}: {name}")
```

---

## MVP化前の機能一覧（Full版: 1230行）

### ファイル情報
| 項目 | 内容 |
|-----|------|
| 総行数 | 1230行 |
| クラス数 | 3 |
| 関数数 | 10 |
| コマンドライン引数 | 10個 |

---

### 1. クラス一覧

#### 1.1 YOLODetector クラス (243-703行)
**役割**: YOLO物体検出器（Hailo-8L対応）

| メソッド | 行番号 | 概要 |
|---------|--------|------|
| `__init__` | 275-316 | 初期化 |
| `is_initialized` | 318-325 | プロパティ、初期化状態の確認 |
| `target_classes` | 328-338 | プロパティ、現在の検出対象クラス取得 |
| `set_target_classes` | 340-393 | 検出対象クラスの設定 |
| `get_available_classes` | 394-405 | 使用可能な全クラス名取得 |
| `is_class_targeted` | 407-424 | 指定クラスが検出対象か確認 |
| `_initialize_hailo` | 426-481 | Hailoデバイスとモデルの初期化 |
| `preprocess_image` | 483-510 | 画像の前処理（リサイズ、RGB変換） |
| `postprocess_detections` | 512-599 | 推論結果の後処理（座標変換） |
| `_apply_nms` | 601-625 | Non-Maximum Suppression適用 |
| `detect` | 627-679 | 画像から物体検出を実行 |
| `_dummy_detect` | 681-702 | CPUモード用のダミー検出 |

#### 1.2 CameraManager クラス (705-862行)
**役割**: カメラ管理（Camera Module V3またはUSB Webカメラ）

| メソッド | 行番号 | 概要 |
|---------|--------|------|
| `__init__` | 725-759 | 初期化 |
| `_initialize_camera` | 761-784 | Picamera2の初期化 |
| `_initialize_webcam` | 786-806 | USB Webカメラの初期化 |
| `read_frame` | 808-839 | カメラからフレーム読み取り |
| `release` | 841-853 | カメラリソースの解放 |
| `__enter__` | 855-857 | コンテキストマネージャーエントリー |
| `__exit__` | 859-862 | コンテキストマネージャー終了 |

#### 1.3 DetectionLogger クラス (865-918行)
**役割**: 検出結果ログ管理

| メソッド | 行番号 | 概要 |
|---------|--------|------|
| `__init__` | 877-896 | ログ管理の初期化 |
| `log_detections` | 898-917 | 検出結果のCSVログ保存 |

---

### 2. 関数一覧

| 関数名 | 行番号 | 概要 |
|--------|--------|------|
| `setup_logging` | 87-117 | ロギングの設定 |
| `get_logger` | 120-132 | モジュールのロガーインスタンス取得 |
| `_check_picamera2` | 155-171 | Picamera2の可用性チェック |
| `_check_hailo` | 174-205 | HailoRT SDKの可用性チェック |
| `is_hailo_available` | 208-221 | HailoRT SDK利用可能性確認（公開API） |
| `is_picamera_available` | 224-237 | Picamera2利用可能性確認（公開API） |
| `draw_detections` | 923-982 | 画像に検出結果を描画 |
| `draw_info` | 985-1020 | 画像にパフォーマンス情報を描画 |
| `parse_resolution` | 1023-1041 | 解像度文字列のパース |
| `main` | 1047-1225 | CLIアプリケーションのメイン関数 |

---

### 3. 定数一覧

| 定数名 | 行番号 | 概要 |
|--------|--------|------|
| `__all__` | 41-58 | ライブラリ公開API定義リスト |
| `COCO_CLASSES` | 65-78 | COCOデータセットのクラス名（80クラス） |
| `_PICAMERA2_AVAILABLE` | 139 | Picamera2可用性フラグ |
| `_HAILO_AVAILABLE` | 140 | Hailo可用性フラグ |
| `_logger` | 84 | モジュール専用ロガー |

---

### 4. コマンドライン引数

| 引数 | デフォルト値 | 説明 |
|------|------------|------|
| `--model` | `models/yolov8s_h8l.hef` | HEFモデルファイルのパス |
| `--res` | `1280x720` | カメラ解像度 |
| `--conf` | `0.25` | 信頼度閾値 |
| `--iou` | `0.45` | IoU閾値（NMS用） |
| `--device` | `0` | カメラデバイスID |
| `--save` | - | 動画を保存する（フラグ） |
| `--log` | - | 検出結果をCSVログに保存（フラグ） |
| `--flip` | - | カメラ映像を上下反転（フラグ） |
| `--classes` | `None` | 検出対象のクラス名 |
| `--list-classes` | - | クラス名一覧を表示して終了（フラグ） |

---

### 5. 主要機能

#### 5.1 カメラ関連機能

| 機能名 | 概要 | 行番号 |
|--------|------|--------|
| Picamera2初期化 | Camera Module V3を初期化 | 761-784 |
| USB Webカメラ初期化 | OpenCV VideoCapture初期化 | 786-806 |
| フレーム取得 | BGRフォーマットでフレーム取得 | 808-839 |
| 自動フォールバック | Picamera2不可時にUSBカメラへ切替 | 749-784 |
| 画像上下反転 | カメラ映像の上下反転処理 | 832-833 |

#### 5.2 検出関連機能

| 機能名 | 概要 | 行番号 |
|--------|------|--------|
| Hailo-8L初期化 | HEFファイル読み込み、デバイス設定 | 426-481 |
| 画像前処理 | リサイズ、BGR→RGB変換 | 483-510 |
| 物体検出推論 | Hailo-8Lで推論実行 | 627-679 |
| 検出結果後処理 | 座標変換（正規化→ピクセル） | 512-599 |
| NMS | Non-Maximum Suppression適用 | 601-625 |
| クラスフィルタリング | 特定クラスのみ検出 | 340-393, 552-554 |
| ダミー検出 | CPUモード用テスト検出 | 681-702 |

#### 5.3 表示・描画関連機能

| 機能名 | 概要 | 行番号 |
|--------|------|--------|
| バウンディングボックス描画 | 検出結果の描画 | 923-982 |
| ラベル位置調整 | 画面端でのラベル位置最適化 | 959-969 |
| パフォーマンス情報描画 | FPS、解像度、推論時間表示 | 985-1020 |
| リアルタイム表示 | OpenCVウィンドウ表示 | 1185 |

#### 5.4 ログ・記録関連機能

| 機能名 | 概要 | 行番号 |
|--------|------|--------|
| CSVログ | 検出結果をCSVに記録 | 898-917 |
| ログファイル自動生成 | タイムスタンプ付きファイル作成 | 877-896 |
| 動画保存 | MP4形式で動画保存 | 1139-1148, 1181-1182 |
| 出力ディレクトリ自動作成 | output/, logs/の自動作成 | 885, 1140-1141 |

#### 5.5 その他の機能

| 機能名 | 概要 | 行番号 |
|--------|------|--------|
| 遅延インポート | Picamera2/HailoRTの遅延読み込み | 155-205 |
| FPS計測 | 10フレームごとにFPS計算 | 1150-1154, 1187-1192 |
| 推論時間計測 | 処理時間をミリ秒で測定 | 1165-1168 |
| 解像度パース | "1280x720"形式のパース | 1023-1041 |
| クラス一覧表示 | COCO 80クラスの一覧表示 | 1096-1105 |
| キーボード入力処理 | q/ESCキーで終了 | 1196-1200 |
| リソース解放 | カメラ・動画の適切な解放 | 1216-1225 |
| コンテキストマネージャー | with文対応 | 855-862 |

---

## MVP化検討

### 維持必須のAPI（外部参照あり）

| API | 種別 | 理由 |
|-----|------|------|
| `YOLODetector` | クラス | camera_tracker.pyで使用 |
| `CameraManager` | クラス | camera_tracker.pyで使用 |
| `draw_detections` | 関数 | camera_tracker.pyで使用 |
| `COCO_CLASSES` | 定数 | camera_tracker.pyで使用 |

### 維持必須のメソッド・引数（外部参照あり）

#### YOLODetector
| メソッド/引数 | 使用箇所 |
|--------------|---------|
| `__init__(model_path, conf_threshold, target_classes)` | camera_tracker.py:432 |
| `detect(frame)` | camera_tracker.py:876, 1117 |

#### CameraManager
| メソッド/引数 | 使用箇所 |
|--------------|---------|
| `__init__(resolution, device_id, flip_vertical)` | camera_tracker.py:446 |
| `read_frame()` | camera_tracker.py:645, 779, 1111 |
| `release()` | camera_tracker.py:1215 |

---

### 削減可能な機能一覧

#### クラス

| 対象 | 削減理由 | 影響 |
|-----|---------|------|
| `DetectionLogger` | 外部参照なし、CLIアプリ専用機能 | なし |

#### YOLODetectorのメソッド

| 対象 | 削減理由 | 影響 |
|-----|---------|------|
| `is_initialized` プロパティ | 外部参照なし | なし |
| `target_classes` プロパティ | 外部参照なし | なし |
| `set_target_classes()` | 外部参照なし（__init__で設定可能） | なし |
| `get_available_classes()` | 外部参照なし | なし |
| `is_class_targeted()` | 外部参照なし | なし |
| `_dummy_detect()` | テスト用、本番不要 | なし |
| `iou_threshold` 引数 | 外部参照なし、デフォルト値で固定可 | なし |

#### CameraManagerのメソッド

| 対象 | 削減理由 | 影響 |
|-----|---------|------|
| `_initialize_webcam()` | Camera Module V3専用にすれば不要 | USB Webカメラ非対応になる |
| `__enter__` / `__exit__` | 外部参照なし（with文未使用） | なし |
| `use_picamera` 引数 | Camera Module V3専用にすれば不要 | なし |

#### 関数

| 対象 | 削減理由 | 影響 |
|-----|---------|------|
| `setup_logging()` | 外部参照なし、print文で代替可 | なし |
| `get_logger()` | 外部参照なし | なし |
| `_check_picamera2()` | 簡素化可能 | なし |
| `_check_hailo()` | 簡素化可能 | なし |
| `is_hailo_available()` | 外部参照なし | なし |
| `is_picamera_available()` | 外部参照なし | なし |
| `draw_info()` | 外部参照なし、CLIアプリ専用 | なし |
| `parse_resolution()` | 外部参照なし、CLIアプリ専用 | なし |

#### コマンドライン引数

| 対象 | 削減理由 | 影響 |
|-----|---------|------|
| `--save` | 動画保存は高度機能 | 動画保存不可 |
| `--log` | CSVログは高度機能 | ログ保存不可 |
| `--iou` | デフォルト値で固定可 | IoU調整不可 |
| `--list-classes` | クラス一覧は別途参照可 | 一覧表示不可 |

#### 機能

| 対象 | 削減理由 | 影響 |
|-----|---------|------|
| USB Webカメラ対応 | Camera Module V3専用で十分 | USBカメラ非対応 |
| 自動フォールバック | Camera Module V3専用で十分 | 自動切替なし |
| 動画保存機能 | 高度機能、書籍では省略可 | 動画保存不可 |
| CSVログ機能 | 高度機能、書籍では省略可 | ログ保存不可 |
| ダミー検出機能 | テスト用、本番不要 | Hailo必須 |
| 遅延インポート | 簡素化のため即時インポートに | 起動時間微増 |
| ラベル位置調整 | 簡素化可能、固定位置で十分 | 画面端で見切れる可能性 |
| コンテキストマネージャー | 外部参照なし | with文使用不可 |
| ロギング機能 | print文で代替可 | ログレベル制御不可 |

---

### コメント削減方針

コメントは `COMMENT_STYLE_GUIDE.md` のガイドラインを守りつつ、以下の方針で削減する。

#### 削減対象のコメント

| 対象 | 例 | 削減理由 |
|-----|-----|---------|
| 冗長な仕様説明 | 長文で処理の詳細を説明 | コードで自明な内容は不要 |
| 自明なコメント | `# 変数を初期化` `# ループ開始` | コードを読めばわかる |
| 重複説明 | docstringと同内容のインラインコメント | 二重管理になる |
| 過剰な補足 | 初心者向けの詳細すぎる説明 | 書籍本文で補足可能 |
| セクション区切り | `# ========` などの装飾コメント | 行数削減 |

#### 維持すべきコメント

| 対象 | 理由 |
|-----|------|
| ファイルヘッダー | プロジェクト情報、簡潔な概要 |
| クラスdocstring | 役割と設計意図（簡潔に） |
| メソッドdocstring | Args, Returns（型情報含む、簡潔に） |
| 「なぜ」を説明するコメント | 設計判断、ワークアラウンドの理由 |
| 警告・注意事項 | エラーハンドリングの意図 |

#### 削減例

**Before（冗長）:**
```python
def detect(self, image):
    """
    画像から物体検出を実行する。

    この関数は入力画像を受け取り、Hailo-8Lアクセラレータを使用して
    YOLOモデルによる物体検出を実行します。検出処理は以下の手順で行われます：
    1. 入力画像の前処理（リサイズ、色空間変換）
    2. Hailoデバイスへの推論リクエスト送信
    3. 推論結果の後処理（座標変換、NMS適用）
    4. 検出結果のフィルタリング（信頼度閾値、対象クラス）

    Args:
        image (numpy.ndarray): 入力画像。BGR形式のnumpy配列。
            形状は(height, width, 3)で、データ型はuint8。

    Returns:
        list: 検出結果のリスト。各要素は以下の形式の辞書：
            - 'bbox': [x1, y1, x2, y2] バウンディングボックス座標
            - 'class_id': int クラスID
            - 'class_name': str クラス名
            - 'confidence': float 信頼度スコア
    """
```

**After（簡潔）:**
```python
def detect(self, image):
    """
    画像から物体検出を実行する。

    Args:
        image: BGR形式の入力画像 (numpy.ndarray)

    Returns:
        list: 検出結果 [{'bbox', 'class_id', 'class_name', 'confidence'}, ...]
    """
```

---

### MVP版の構成案

#### 維持する構成

| 項目 | Full版 | MVP版 |
|-----|--------|-------|
| クラス数 | 3 | 2（YOLODetector, CameraManager） |
| 関数数 | 10 | 2（draw_detections, main） |
| コマンドライン引数 | 10 | 5（--model, --res, --conf, --device, --classes） |
| 推定行数 | 1230 | 400-500 |

#### MVP版で維持する機能

**カメラ関連:**
- Picamera2初期化（Camera Module V3専用）
- フレーム取得
- 画像上下反転（--flipは維持検討）

**検出関連:**
- Hailo-8L初期化
- 画像前処理
- 物体検出推論
- 検出結果後処理
- NMS
- クラスフィルタリング（外部参照で使用）

**表示・描画関連:**
- バウンディングボックス描画（draw_detections）
- リアルタイム表示

**その他:**
- COCO_CLASSES定数
- キーボード入力処理（q/ESCで終了）
- リソース解放

---

## 機能仕様（MVP版）

### 実施結果

| 項目 | Full版 | MVP版 | 削減率 |
|-----|--------|-------|--------|
| 総行数 | 1230行 | 402行 | **67%** |
| クラス数 | 3 | 2 | 33% |
| 関数数 | 10 | 2 | 80% |
| CLI引数 | 10 | 6 | 40% |

### クラス構成

#### YOLODetector

```python
class YOLODetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.25,
                 target_classes: Optional[List[str]] = None)
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]
```

| メソッド | 概要 |
|---------|------|
| `__init__` | 初期化（Hailo-8Lデバイス設定） |
| `_set_target_classes` | 検出対象クラス設定（内部メソッド） |
| `_initialize_hailo` | Hailoデバイス初期化（内部メソッド） |
| `detect` | 物体検出実行（公開API） |
| `_postprocess` | 後処理（内部メソッド） |

#### CameraManager

```python
class CameraManager:
    def __init__(self, resolution: Tuple[int, int] = (1280, 720),
                 device_id: int = 0, flip_vertical: bool = False)
    def read_frame(self) -> Optional[np.ndarray]
    def release(self) -> None
```

| メソッド | 概要 |
|---------|------|
| `__init__` | 初期化（Picamera2設定） |
| `_initialize_camera` | カメラ初期化（内部メソッド） |
| `read_frame` | フレーム取得（公開API） |
| `release` | リソース解放（公開API） |

### 関数構成

```python
def draw_detections(image: np.ndarray, detections: List[Dict[str, Any]],
                   color: Tuple[int, int, int] = (0, 255, 0),
                   thickness: int = 2) -> np.ndarray

def main() -> None
```

### 定数

```python
COCO_CLASSES: List[str]  # 80クラス
```

### コマンドライン引数

| 引数 | デフォルト | 説明 |
|------|------------|------|
| `--model` | `models/yolov8s_h8l.hef` | HEFモデルファイル |
| `--res` | `1280x720` | カメラ解像度 |
| `--conf` | `0.25` | 信頼度閾値 |
| `--device` | `0` | カメラデバイスID |
| `--flip` | - | 上下反転 |
| `--classes` | - | 検出対象クラス |

### 外部参照互換性

以下のAPIシグネチャは外部参照プロジェクトとの互換性を維持：

| API | シグネチャ | 互換性 |
|-----|-----------|--------|
| `YOLODetector.__init__` | `(model_path, conf_threshold=0.25, target_classes=None)` | ✓ |
| `YOLODetector.detect` | `(image) -> List[Dict]` | ✓ |
| `CameraManager.__init__` | `(resolution, device_id, flip_vertical)` | ✓ |
| `CameraManager.read_frame` | `() -> Optional[np.ndarray]` | ✓ |
| `CameraManager.release` | `() -> None` | ✓ |
| `draw_detections` | `(image, detections, color, thickness)` | ✓ |
| `COCO_CLASSES` | `List[str]` | ✓ |

---

## ライブラリAPI仕様

本モジュールをライブラリとして使用する際のAPI仕様を定義する。

### インポート

```python
from raspi_hailo8l_yolo import YOLODetector, CameraManager, draw_detections, COCO_CLASSES
```

---

### YOLODetector

YOLO物体検出器クラス（Hailo-8L対応）

#### コンストラクタ

```python
YOLODetector(model_path, conf_threshold=0.25, target_classes=None)
```

| パラメータ | 型 | デフォルト | 必須 | 説明 |
|-----------|-----|-----------|------|------|
| `model_path` | `str` | - | ✓ | HEFモデルファイルのパス（例: `"models/yolov8s_h8l.hef"`） |
| `conf_threshold` | `float` | `0.25` | - | 信頼度閾値（0.0〜1.0）。この値以上の検出結果のみ返す |
| `target_classes` | `List[str]` or `None` | `None` | - | 検出対象のクラス名リスト。`None`で全80クラス検出 |

**例外:**
- `FileNotFoundError`: モデルファイルが見つからない場合
- `RuntimeError`: Hailoデバイスの初期化に失敗した場合
- `ValueError`: 無効なクラス名が指定された場合

#### detect メソッド

```python
detect(image) -> List[Dict[str, Any]]
```

| パラメータ | 型 | 説明 |
|-----------|-----|------|
| `image` | `numpy.ndarray` | BGR形式の入力画像（shape: `(height, width, 3)`、dtype: `uint8`） |

**戻り値:** `List[Dict[str, Any]]`

検出結果のリスト。各要素は以下のキーを持つ辞書：

| キー | 型 | 説明 |
|------|-----|------|
| `bbox` | `List[int]` | バウンディングボックス座標 `[x1, y1, x2, y2]`（ピクセル単位） |
| `class_id` | `int` | クラスID（0〜79、COCOデータセット） |
| `class_name` | `str` | クラス名（例: `"person"`, `"car"`） |
| `confidence` | `float` | 信頼度スコア（0.0〜1.0） |

---

### CameraManager

カメラ管理クラス（Camera Module V3用）

#### コンストラクタ

```python
CameraManager(resolution=(1280, 720), device_id=0, flip_vertical=False)
```

| パラメータ | 型 | デフォルト | 必須 | 説明 |
|-----------|-----|-----------|------|------|
| `resolution` | `Tuple[int, int]` | `(1280, 720)` | - | カメラ解像度 `(width, height)` |
| `device_id` | `int` | `0` | - | カメラデバイスID |
| `flip_vertical` | `bool` | `False` | - | `True`で画像を上下反転（カメラを逆さまに設置した場合） |

**対応解像度:**
- `(640, 480)` - VGA
- `(1280, 720)` - HD 720p（推奨）
- `(1920, 1080)` - Full HD 1080p

**例外:**
- `RuntimeError`: Picamera2が利用できない場合、またはカメラの初期化に失敗した場合

#### read_frame メソッド

```python
read_frame() -> Optional[numpy.ndarray]
```

| 戻り値 | 型 | 説明 |
|--------|-----|------|
| フレーム | `numpy.ndarray` | BGR形式の画像（shape: `(height, width, 3)`） |
| 失敗時 | `None` | フレーム取得に失敗した場合 |

#### release メソッド

```python
release() -> None
```

カメラリソースを解放する。使用後は必ず呼び出すこと。

---

### draw_detections

画像に検出結果を描画する関数

```python
draw_detections(image, detections, color=(0, 255, 0), thickness=2) -> numpy.ndarray
```

| パラメータ | 型 | デフォルト | 必須 | 説明 |
|-----------|-----|-----------|------|------|
| `image` | `numpy.ndarray` | - | ✓ | BGR形式の入力画像 |
| `detections` | `List[Dict]` | - | ✓ | `YOLODetector.detect()`の戻り値 |
| `color` | `Tuple[int, int, int]` | `(0, 255, 0)` | - | バウンディングボックスの色（BGR形式、緑） |
| `thickness` | `int` | `2` | - | 線の太さ（ピクセル） |

**戻り値:** `numpy.ndarray` - バウンディングボックスとラベルが描画された画像

---

### COCO_CLASSES

COCOデータセットの80クラス名リスト

```python
COCO_CLASSES: List[str]
```

| インデックス | クラス名 | インデックス | クラス名 |
|-------------|---------|-------------|---------|
| 0 | person | 40 | wine glass |
| 1 | bicycle | 41 | cup |
| 2 | car | 42 | fork |
| 3 | motorcycle | 43 | knife |
| 4 | airplane | 44 | spoon |
| 5 | bus | 45 | bowl |
| 6 | train | 46 | banana |
| 7 | truck | 47 | apple |
| 14 | bird | 48 | sandwich |
| 15 | cat | 56 | chair |
| 16 | dog | 57 | couch |
| 17 | horse | 62 | tv |
| ... | ... | ... | ... |

※全80クラスは `COCO_CLASSES` を参照

---

### 使用例

```python
from raspi_hailo8l_yolo import YOLODetector, CameraManager, draw_detections, COCO_CLASSES
import cv2

# 初期化
detector = YOLODetector(
    model_path="models/yolov8s_h8l.hef",
    conf_threshold=0.3,
    target_classes=['person', 'car']  # 人と車のみ検出
)

camera = CameraManager(
    resolution=(1280, 720),
    flip_vertical=False
)

try:
    while True:
        # フレーム取得
        frame = camera.read_frame()
        if frame is None:
            break

        # 物体検出
        detections = detector.detect(frame)

        # 結果処理
        for det in detections:
            print(f"{det['class_name']}: {det['confidence']:.2f}")

        # 描画
        result = draw_detections(frame, detections)
        cv2.imshow('Detection', result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
```

---

## 変更履歴

| 日付 | 内容 |
|-----|------|
| 2025-12-29 | 初版作成（外部参照情報を記載） |
| 2025-12-29 | MVP化前の機能一覧を追加 |
| 2025-12-29 | MVP化検討（削減可能な機能一覧）を追加 |
| 2025-12-29 | コメント削減方針を追加 |
| 2025-12-29 | MVP版実施結果を追加 |
| 2025-12-29 | ライブラリAPI仕様を追加 |
