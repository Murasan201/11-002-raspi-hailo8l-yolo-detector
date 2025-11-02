# Raspberry Pi Hailo-8L YOLO Detector

Raspberry Pi 5 + 公式 AI Kit（Hailo-8L）+ Raspberry Pi Camera Module V3 を用いた、リアルタイム YOLO 物体検出アプリケーションです。

## 特徴

- **単一ファイル構成**: `raspi_hailo8l_yolo.py` で完結
- **高速推論**: Hailo-8L AIアクセラレータによる高速物体検出
- **リアルタイム処理**: Camera Module V3 からのライブ映像処理
- **柔軟な設定**: 解像度、信頼度閾値、IoU閾値などをコマンドライン引数で調整可能
- **保存機能**: 検出結果の動画保存、CSVログ出力に対応

## 必要なハードウェア

- Raspberry Pi 5
- Raspberry Pi AI Kit（Hailo-8L）
- Raspberry Pi Camera Module V3（IMX708）
- microSD カード（32GB以上推奨）
- ディスプレイ（HDMI）

## 必要なソフトウェア

- Raspberry Pi OS（Bookworm 以降）
- Python 3.11+
- HailoRT SDK
- OpenCV
- picamera2

## セットアップ手順

### 1. OS とファームウェアの更新

```bash
sudo apt update && sudo apt full-upgrade -y
sudo reboot
```

### 2. カメラの有効化

```bash
sudo raspi-config
```

- `Interface Options` → `Camera` → `Enable` を選択
- 再起動後、カメラの動作確認：

```bash
libcamera-hello --preview
```

### 3. HailoRT SDK のインストール

公式手順に従ってHailoRT SDKとカーネルモジュールを導入してください：

```bash
# Hailo デバイスの認識確認
hailortcli fw-control identify
```

### 4. Python環境のセットアップ

```bash
# 仮想環境の作成
python3 -m venv .venv
source .venv/bin/activate

# 依存関係のインストール
pip install -r requirements.txt
```

### 5. モデルファイルの配置

Hailo-8L用に最適化されたYOLOモデル（`.hef`形式）を `models/` ディレクトリに配置してください：

```bash
# 例: YOLOv8n Hailoモデル
models/yolov8n_hailo.hef
```

**注意**: モデルファイルは配布条件に留意し、各自で取得・配置してください。

## 使用方法

### Camera Module V3版（基本実装）

#### 基本的な実行

```bash
# 仮想環境の有効化
source .venv/bin/activate

# 基本実行（1280x720、信頼度0.25）
python raspi_hailo8l_yolo.py

# 解像度とパラメータを指定
python raspi_hailo8l_yolo.py --res 1280x720 --conf 0.25 --iou 0.45
```

### オプション一覧

| オプション | デフォルト | 説明 |
|----------|----------|------|
| `--model` | `models/yolov8n_hailo.hef` | HEFモデルファイルのパス |
| `--res` | `1280x720` | カメラ解像度（640x480, 1280x720, 1920x1080） |
| `--conf` | `0.25` | 信頼度閾値（0.0-1.0） |
| `--iou` | `0.45` | IoU閾値（NMS用、0.0-1.0） |
| `--device` | `0` | カメラデバイスID |
| `--save` | - | 動画保存を有効化 |
| `--log` | - | 検出結果のCSVログ保存を有効化 |

### 使用例

```bash
# 高解像度で実行
python raspi_hailo8l_yolo.py --res 1920x1080

# 高精度設定（信頼度を上げる）
python raspi_hailo8l_yolo.py --conf 0.5 --iou 0.3

# 動画保存とログ出力を有効化
python raspi_hailo8l_yolo.py --save --log

# USB Webカメラを使用（Camera Module V3が利用できない場合）
python raspi_hailo8l_yolo.py --device 0
```

### USB Webcam版（ロジクール等対応・高性能版）

**参考**: [Raspberry Pi 5 + Hailo-8L + Webcam で YOLO 物体検出](https://murasan-net.com/2024/11/13/raspberry-pi-5-hailo-8l-webcam/)

#### セットアップ

```bash
# 自動セットアップスクリプトの実行
chmod +x setup_usb_webcam.sh
./setup_usb_webcam.sh

# または手動セットアップ
sudo apt install python3-gi gstreamer1.0-plugins-base gstreamer1.0-plugins-good
source .venv/bin/activate
pip install -r requirements_usb.txt
```

#### 基本的な実行

```bash
# 仮想環境の有効化
source .venv/bin/activate

# 基本実行（1280x720、YOLOv8s）
python raspi_hailo8l_yolo_usb.py

# 詳細設定での実行
python raspi_hailo8l_yolo_usb.py --network yolov8s --device /dev/video0 --width 1280 --height 720
```

#### USB版専用オプション

| オプション | デフォルト | 説明 |
|----------|----------|------|
| `--network` | `yolov8s` | YOLOネットワーク（yolov6n, yolov8s, yolox_s_leaky） |
| `--device` | `/dev/video0` | USBカメラデバイスパス |
| `--width` | `1280` | カメラ映像幅 |
| `--height` | `720` | カメラ映像高さ |
| `--fps` | `30` | 目標フレームレート |
| `--fullscreen` | - | フルスクリーン表示 |

#### 使用例（USB版）

```bash
# 高解像度で実行
python raspi_hailo8l_yolo_usb.py --width 1920 --height 1080

# YOLOv6n（軽量モデル）で実行
python raspi_hailo8l_yolo_usb.py --network yolov6n

# フルスクリーン + 動画保存
python raspi_hailo8l_yolo_usb.py --fullscreen --save

# 別のUSBカメラを使用
python raspi_hailo8l_yolo_usb.py --device /dev/video2
```

#### USB Webcam のトラブルシューティング

```bash
# カメラの確認
v4l2-ctl --list-devices

# 対応フォーマットの確認
v4l2-ctl --device=/dev/video0 --list-formats-ext

# GStreamerテスト
gst-launch-1.0 v4l2src device=/dev/video0 ! videoconvert ! autovideosink
```

## 操作方法

### 共通操作

- **終了**: `q` キーを押す
- **画面表示**: リアルタイムで検出結果が表示されます
  - 緑色の矩形: バウンディングボックス
  - ラベル: クラス名と信頼度
  - 左上情報: FPS、解像度、推論時間

### USB版追加操作

- **フルスクリーン切り替え**: `f` キーを押す

## 出力ファイル

### 動画保存（`--save` オプション）

```
output/detection_20241229_143052.mp4
```

### ログファイル（`--log` オプション）

```
logs/detections_20241229_143052.csv
```

CSVフォーマット：
```
timestamp,frame_id,class_name,confidence,x1,y1,x2,y2
2024-12-29T14:30:52.123456,1,person,0.85,100,150,300,400
```

## トラブルシューティング

### よくある問題

1. **カメラが認識されない**
   ```bash
   # カメラの確認
   libcamera-hello --list-cameras

   # I2Cの有効化
   sudo raspi-config  # Interface → I2C → Enable
   ```

2. **Hailoデバイスが見つからない**
   ```bash
   # デバイスの確認
   lsusb | grep Hailo
   hailortcli fw-control identify

   # ドライバの再読み込み
   sudo modprobe hailo_pci
   ```

3. **依存関係のエラー**
   ```bash
   # picamera2の手動インストール
   sudo apt install python3-picamera2

   # OpenCVの再インストール
   pip uninstall opencv-python
   pip install opencv-python
   ```

4. **メモリ不足**
   ```bash
   # スワップファイルの増加
   sudo dphys-swapfile swapoff
   sudo nano /etc/dphys-swapfile  # CONF_SWAPSIZE=2048
   sudo dphys-swapfile setup
   sudo dphys-swapfile swapon
   ```

### パフォーマンス最適化

1. **GPU メモリの調整**
   ```bash
   sudo nano /boot/firmware/config.txt
   # gpu_mem=128 を追加
   ```

2. **CPUガバナーの設定**
   ```bash
   echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
   ```

## 開発・カスタマイズ

### コード構造

- `YOLODetector`: Hailo-8L推論エンジン
- `CameraManager`: カメラ入力管理
- `DetectionLogger`: ログ出力管理
- `draw_detections()`: 描画関数
- `main()`: メイン処理

### 新しいモデルの追加

1. Hailo Model Zoo からモデルをダウンロード
2. `models/` ディレクトリに配置
3. `--model` オプションでパスを指定

### クラス名のカスタマイズ

`YOLODetector` クラスの `class_names` リストを編集することで、検出対象クラスを変更できます。

## ライセンス

- ソースコード: MIT License
- モデルファイル: 各ベンダーのライセンスに従う

## 参考資料

- [Hailo-AI Official Examples](https://github.com/hailo-ai/hailo-rpi5-examples)
- [Raspberry Pi Camera Documentation](https://www.raspberrypi.org/documentation/accessories/camera.html)
- [HailoRT Developer Guide](https://hailo.ai/developer-zone/documentation/)

## バージョン情報

- Version: 1.0.0
- 最終更新: 2024-12-29
- 対応OS: Raspberry Pi OS Bookworm以降
- 対応ハードウェア: Raspberry Pi 5 + Hailo-8L AI Kit

---

## サポート

問題が発生した場合は、以下の情報を含めてIssueを作成してください：

- Raspberry Pi OSのバージョン
- HailoRT SDKのバージョン
- エラーメッセージの全文
- 実行したコマンド
- ハードウェア構成