# Troubleshooting Guide - Raspberry Pi Hailo-8L YOLO Detector

本ドキュメントは、環境構築および運用中に発生した問題とその解決策を記録したナレッジベースです。

> **更新日**: 2024-12-17
> **対象環境**: Raspberry Pi 5 + Hailo-8L AI Kit + Camera Module V3

---

## 目次

1. [環境情報](#環境情報)
2. [カメラ関連の問題](#カメラ関連の問題)
3. [Hailo-8L 関連の問題](#hailo-8l-関連の問題)
4. [Python 環境の問題](#python-環境の問題)
5. [アプリケーション実行時の問題](#アプリケーション実行時の問題)
6. [パフォーマンスの問題](#パフォーマンスの問題)

---

## 環境情報

環境構築時のシステム情報：

```
OS: Debian 13 (trixie)
Kernel: 6.12.47+rpt-rpi-2712
Python: 3.13.5
Hardware: Raspberry Pi 5 (8GB RAM)
Storage: 28GB microSD (19GB 空き)
```

---

## カメラ関連の問題

### 問題 1: libcamera-hello コマンドが見つからない

**発生日**: 2024-12-17

**エラーメッセージ**:
```
/bin/bash: libcamera-hello: コマンドが見つかりません
```

**原因**:
- libcamera-apps パッケージがインストールされていない
- **重要**: Debian 13 (trixie) / Raspberry Pi OS 最新版では、コマンド名が変更されている
  - 旧: `libcamera-hello`, `libcamera-still`, `libcamera-vid`
  - 新: `rpicam-hello`, `rpicam-still`, `rpicam-vid`

**解決策**:
```bash
# libcamera 関連パッケージのインストール
sudo apt update
sudo apt install -y libcamera-apps libcamera-tools

# インストール確認（新しいコマンド名を使用）
rpicam-hello --list-cameras
```

**確認コマンド**:
```bash
# カメラの一覧表示（新コマンド）
rpicam-hello --list-cameras

# プレビュー表示（5秒）
rpicam-hello --timeout 5000

# 旧コマンド名でも動作する場合がある（シンボリックリンク）
# libcamera-hello --list-cameras
```

**コマンド名対応表**:
| 旧コマンド | 新コマンド | 用途 |
|-----------|-----------|------|
| `libcamera-hello` | `rpicam-hello` | カメラプレビュー |
| `libcamera-still` | `rpicam-still` | 静止画撮影 |
| `libcamera-vid` | `rpicam-vid` | 動画撮影 |
| `libcamera-raw` | `rpicam-raw` | RAW撮影 |
| `libcamera-jpeg` | `rpicam-jpeg` | JPEG撮影 |

**実際の動作確認結果** (2024-12-17):
```
Available cameras
-----------------
0 : imx708_wide_noir [4608x2592 10-bit RGGB]
    Modes: 'SRGGB10_CSI2P' : 1536x864 [120.13 fps]
                             2304x1296 [56.03 fps]
                             4608x2592 [14.35 fps]
```

**ステータス**: [x] 解決済み

---

### 問題 2: カメラが認識されない

**発生日**: -

**エラーメッセージ**:
```
ERROR: Could not open camera
No cameras available
```

**原因**:
- カメラケーブルの接続不良
- カメラインターフェースが無効
- I2C が無効

**解決策**:
```bash
# 1. カメラインターフェースの有効化
sudo raspi-config
# → Interface Options → Legacy Camera → Disable（libcamera を使用）

# 2. I2C の有効化
sudo raspi-config
# → Interface Options → I2C → Enable

# 3. 再起動
sudo reboot

# 4. カメラの物理接続確認
# - フラットケーブルの向き（青いテープ側が基板側）
# - コネクタのロック確認
```

**確認コマンド**:
```bash
# I2C デバイスの確認
sudo i2cdetect -y 10
# IMX708 (Camera Module V3) は 0x1a に表示される

# カメラデバイスの確認
ls -l /dev/video*
```

**ステータス**: [ ] 未解決 / [ ] 解決済み

---

## Hailo-8L 関連の問題

### 問題 3: hailortcli コマンドが見つからない

**発生日**: 2024-12-17

**エラーメッセージ**:
```
/bin/bash: hailortcli: コマンドが見つかりません
```

**原因**:
- HailoRT SDK がインストールされていない
- Debian 13 (trixie) / Raspberry Pi OS では `hailo-all` メタパッケージでインストール

**解決策**:
```bash
# Raspberry Pi OS 向け（推奨）
sudo apt update
sudo apt install -y hailo-all

# これにより以下がインストールされる:
# - hailort (HailoRT SDK)
# - hailort-pcie-driver (PCIe ドライバ)
# - hailo-tappas-core (TAPPAS フレームワーク)
# - python3-hailort (Python バインディング)
# - rpicam-apps-hailo-postprocess (カメラ統合)
# - その他依存関係（OpenCV, GStreamer 等）

# インストール確認
hailortcli fw-control identify
```

**実際の動作確認結果** (2024-12-17):
```
Executing on device: 0001:01:00.0
Identifying board
Control Protocol Version: 2
Firmware Version: 4.23.0 (release,app,extended context switch buffer)
Logger Version: 0
Board Name: Hailo-8
Device Architecture: HAILO8L
Serial Number: HLDDLBB243302146
Part Number: HM21LB1C2LAE
Product Name: HAILO-8L AI ACC M.2 B+M KEY MODULE EXT TMP
```

**確認コマンド**:
```bash
# PCI デバイスの確認
lspci | grep -i hailo

# カーネルモジュールの確認
lsmod | grep hailo

# デバイスファイルの確認
ls -l /dev/hailo*
```

**ステータス**: [x] 解決済み

---

### 問題 4: Hailo デバイスが認識されない

**発生日**: -

**エラーメッセージ**:
```
Error: Failed to identify board
No Hailo devices found
```

**原因**:
- M.2 HAT+ または Hailo-8L の物理的な接続不良
- カーネルモジュールがロードされていない
- 権限の問題

**解決策**:
```bash
# 1. カーネルモジュールの手動ロード
sudo modprobe hailo_pci

# 2. デバイスの権限確認と修正
sudo chmod 666 /dev/hailo*

# 3. ユーザーをグループに追加
sudo usermod -aG video $USER
sudo usermod -aG dialout $USER

# 4. 再ログインまたは再起動
sudo reboot
```

**確認コマンド**:
```bash
# モジュールの確認
lsmod | grep hailo

# デバイスの確認
ls -l /dev/hailo*

# ファームウェア情報の確認
hailortcli fw-control identify
```

**ステータス**: [ ] 未解決 / [ ] 解決済み

---

## Python 環境の問題

### 問題 5: Python バージョンの互換性

**発生日**: 2024-12-17

**状況**:
- システム Python: 3.13.5
- 要件: Python 3.11+

**備考**:
Python 3.13 は要件を満たしているが、一部のライブラリで互換性問題が発生する可能性がある。

**解決策**:
```bash
# 仮想環境の作成（システム Python を使用）
python3 -m venv .venv
source .venv/bin/activate

# 互換性問題が発生した場合は Python 3.11 をインストール
sudo apt install -y python3.11 python3.11-venv
python3.11 -m venv .venv
source .venv/bin/activate
```

**ステータス**: [ ] 未解決 / [ ] 解決済み

---

### 問題 6: pip パッケージのインストールエラー

**発生日**: -

**エラーメッセージ**:
```
ERROR: Could not build wheels for XXX
```

**原因**:
- ビルドツールが不足
- システム依存関係が不足

**解決策**:
```bash
# ビルドツールのインストール
sudo apt install -y \
    build-essential \
    python3-dev \
    cmake \
    pkg-config

# numpy/OpenCV のビルド依存関係
sudo apt install -y \
    libatlas-base-dev \
    libhdf5-dev \
    libopenblas-dev

# pip のアップグレード
pip install --upgrade pip setuptools wheel
```

**ステータス**: [ ] 未解決 / [ ] 解決済み

---

## アプリケーション実行時の問題

### 問題 7: モデルファイルが見つからない

**発生日**: 2024-12-17

**エラーメッセージ**:
```
FileNotFoundError: Model file not found: models/yolov8n_hailo.hef
```

**原因**:
- モデルファイルが配置されていない
- デフォルトモデル名と実際のモデルファイル名が異なる

**解決策**:

**方法1: hailo-all パッケージのモデルを使用（推奨）**
```bash
# hailo-all パッケージに含まれるモデルを確認
ls -la /usr/share/hailo-models/

# シンボリックリンクを作成
mkdir -p models
ln -sf /usr/share/hailo-models/yolov8s_h8l.hef models/yolov8s_h8l.hef
ln -sf /usr/share/hailo-models/yolov6n_h8l.hef models/yolov6n_h8l.hef

# --model オプションでモデルを指定して実行
python raspi_hailo8l_yolo.py --model models/yolov8s_h8l.hef
```

**方法2: 公式サンプルからダウンロード**
```bash
git clone https://github.com/hailo-ai/hailo-rpi5-examples.git /tmp/hailo-examples
cp /tmp/hailo-examples/resources/*.hef models/
```

**利用可能なモデル（hailo-all パッケージ）**:
| モデル | ファイル名 | サイズ | 用途 |
|--------|-----------|-------|------|
| YOLOv6n | yolov6n_h8l.hef | 14.5MB | 軽量・高速 |
| YOLOv8s | yolov8s_h8l.hef | 36.6MB | バランス型 |
| YOLOX-S | yolox_s_leaky_h8l_rpi.hef | 22.4MB | 高精度 |

**ステータス**: [x] 解決済み

---

### 問題 8: Qt/OpenCV の GUI エラー（SSH経由の場合）

**発生日**: 2024-12-17

**エラーメッセージ**:
```
qt.qpa.xcb: could not connect to display
qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "" even though it was found.
This application failed to start because no Qt platform plugin could be initialized.
```

または
```
cv2.error: OpenCV(4.x.x) error: (-2:Unspecified error) Can't initialize GTK backend
```

**原因**:
- SSH経由でディスプレイ（DISPLAY環境変数）が設定されていない
- Qt/OpenCV の GUI バックエンドがディスプレイに接続できない

**解決策**:

**方法1: オフスクリーンバックエンドを使用（SSH経由で動画保存のみ）**
```bash
# QT_QPA_PLATFORM=offscreen を設定して実行
QT_QPA_PLATFORM=offscreen python raspi_hailo8l_yolo.py \
    --model models/yolov8s_h8l.hef \
    --res 640x480 \
    --save
```

**方法2: VNC/リモートデスクトップを使用**
```bash
# VNC サーバーをインストール
sudo apt install -y realvnc-vnc-server

# VNC を有効化
sudo raspi-config
# → Interface Options → VNC → Enable

# VNC クライアントから接続してから実行
python raspi_hailo8l_yolo.py --model models/yolov8s_h8l.hef
```

**方法3: X11 フォワーディングを使用**
```bash
# クライアント側で X サーバーを起動後、SSH 接続
ssh -X pi@raspberrypi

# DISPLAY 環境変数を確認
echo $DISPLAY
# :10.0 などが表示されればOK

python raspi_hailo8l_yolo.py --model models/yolov8s_h8l.hef
```

**実際の動作確認結果** (2024-12-17):
```bash
# オフスクリーンモードで正常に動作
QT_QPA_PLATFORM=offscreen python raspi_hailo8l_yolo.py \
    --model models/yolov8s_h8l.hef --res 640x480 --save

# 出力結果
ls -la output/
# detection_20251217_183250.mp4 (786KB) が生成された
```

**ステータス**: [x] 解決済み

---

## パフォーマンスの問題

### 問題 9: FPS が低い（5 FPS 以下）

**発生日**: -

**原因**:
- CPU ガバナーが省電力モード
- 高解像度を使用している
- 熱スロットリング

**解決策**:
```bash
# 1. CPU ガバナーを performance に変更
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# 2. 解像度を下げる
python raspi_hailo8l_yolo.py --res 640x480

# 3. 軽量モデルを使用
python raspi_hailo8l_yolo.py --model models/yolov8n_hailo.hef

# 4. CPU 温度の確認
vcgencmd measure_temp
```

**ステータス**: [ ] 未解決 / [ ] 解決済み

---

### 問題 10: HailoRT API 互換性エラー（InferVStreams.create_input_vstreams）

**発生日**: 2024-12-17

**エラーメッセージ**:
```
エラー: Hailo デバイスの初期化に失敗しました: type object 'InferVStreams' has no attribute 'create_input_vstreams'
```

**症状**:
- バウンディングボックスが最初に検出された位置から更新されない
- 実際にはダミー検出（固定位置）が使用されている

**原因**:
- HailoRT SDK v4.x で API が変更された
- 旧 API (`InferVStreams.create_input_vstreams`) は廃止
- 新 API では `InputVStreamParams`, `OutputVStreamParams` を使用

**解決策**:

コードの修正が必要です。主な変更点：

1. **インポートの追加**:
```python
from hailo_platform import (HEF, VDevice, HailoStreamInterface,
                           InferVStreams, ConfigureParams,
                           InputVStreamParams, OutputVStreamParams,
                           FormatType)
```

2. **初期化処理の変更**:
```python
# HEF ファイルの読み込み
self.hef = HEF(model_path)

# 入出力名の取得
input_vstream_infos = self.hef.get_input_vstream_infos()
output_vstream_infos = self.hef.get_output_vstream_infos()
self.input_name = input_vstream_infos[0].name
self.output_name = output_vstream_infos[0].name

# ネットワークグループの設定
configure_params = ConfigureParams.create_from_hef(
    hef=self.hef, interface=HailoStreamInterface.PCIe)
network_groups = self.device.configure(self.hef, configure_params)
self.network_group = network_groups[0]

# VStreams パラメータの作成
self.input_vstreams_params = InputVStreamParams.make(
    self.network_group, format_type=FormatType.UINT8)
self.output_vstreams_params = OutputVStreamParams.make(
    self.network_group, format_type=FormatType.FLOAT32)
```

3. **推論処理の変更**:
```python
with self.network_group.activate():
    with InferVStreams(self.network_group,
                      self.input_vstreams_params,
                      self.output_vstreams_params) as infer_pipeline:
        input_dict = {self.input_name: preprocessed}
        output_dict = infer_pipeline.infer(input_dict)
        outputs = output_dict[self.output_name]
```

4. **後処理の変更**（NMS 後処理済み出力形式に対応）:
```python
# 出力形式: [batch][class_id] = np.ndarray(N, 5)
# 各検出は [y1, x1, y2, x2, confidence] の形式
for class_id, class_detections in enumerate(batch_output):
    for detection in class_detections:
        y1, x1, y2, x2, confidence = detection
```

**確認コマンド**:
```bash
# HailoRT API の確認
python3 -c "from hailo_platform import InputVStreamParams; print('API OK')"

# 推論テスト
python raspi_hailo8l_yolo.py --model models/yolov8s_h8l.hef --res 640x480
```

**ステータス**: [x] 解決済み

---

### 問題 11: バウンディングボックスが [0, 0, 0, 0] になる（座標変換エラー）

**発生日**: 2024-12-17

**症状**:
- 物体検出は成功しているが、バウンディングボックスが画面に表示されない
- 検出結果の座標が `[0, 0, 0, 0]` になる
- confidence（信頼度）は正常な値（例: 0.559）が返される

**エラーメッセージ**:
```
Detections: 1
  person: 0.559 @ [0, 0, 0, 0]
```

**原因**:
- HailoRT NMS出力の座標が **0-1の正規化値** で返される
- コードが座標をピクセル値として処理していたため、スケーリングが正しく行われなかった
- 出力形式: `[y1, x1, y2, x2, confidence]`（すべて0.0-1.0の範囲）

**解決策**:

`postprocess_detections` メソッドで、正規化座標を元画像サイズに乗算する：

```python
# 修正前（間違い）：入力サイズでスケーリング
scale_x = original_shape[1] / input_shape[1]  # 640/640 = 1.0
x1_scaled = int(x1 * scale_x)  # 0.8 * 1.0 = 0（intで切り捨て）

# 修正後（正解）：正規化座標を直接元画像サイズに乗算
orig_h, orig_w = original_shape  # (480, 640)
x1 = int(x1_norm * orig_w)  # 0.8 * 640 = 512
y1 = int(y1_norm * orig_h)  # 0.8 * 480 = 384
```

**正しい座標変換コード**:
```python
# HailoRT NMS出力形式: [y1, x1, y2, x2, confidence]
# 座標は0-1の正規化値
y1_norm, x1_norm, y2_norm, x2_norm, confidence = detection

# 正規化座標を元画像のピクセル座標に変換
x1 = int(x1_norm * orig_w)
y1 = int(y1_norm * orig_h)
x2 = int(x2_norm * orig_w)
y2 = int(y2_norm * orig_h)
```

**確認コマンド**:
```bash
# 検出結果のテスト
source .venv/bin/activate
python3 -c "
from raspi_hailo8l_yolo import YOLODetector, CameraManager
detector = YOLODetector('models/yolov8s_h8l.hef')
camera = CameraManager((640, 480))
frame = camera.read_frame()
detections = detector.detect(frame)
for det in detections:
    print(f'{det[\"class_name\"]}: {det[\"confidence\"]:.3f} @ {det[\"bbox\"]}')
camera.release()
"
# 期待値: person: 0.627 @ [2, 21, 542, 478]（座標が0以外の正常な値）
```

**ステータス**: [x] 解決済み

---

## 問題報告テンプレート

新しい問題が発生した場合は、以下のテンプレートを使用して記録してください：

```markdown
### 問題 X: [問題のタイトル]

**発生日**: YYYY-MM-DD

**エラーメッセージ**:
```
[エラーメッセージをここに貼り付け]
```

**原因**:
- [原因を箇条書きで記載]

**解決策**:
```bash
[解決コマンドを記載]
```

**確認コマンド**:
```bash
[確認用のコマンドを記載]
```

**ステータス**: [ ] 未解決 / [ ] 解決済み
```

---

## 参考リンク

- [Hailo Developer Zone](https://hailo.ai/developer-zone/)
- [Raspberry Pi Camera Documentation](https://www.raspberrypi.com/documentation/accessories/camera.html)
- [hailo-rpi5-examples (GitHub)](https://github.com/hailo-ai/hailo-rpi5-examples)

---

**最終更新**: 2024-12-17
