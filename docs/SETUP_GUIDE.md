# Setup Guide - Raspberry Pi Hailo-8L YOLO Detector

本ドキュメントは、Raspberry Pi 5 + Hailo-8L AI Kit + Camera Module V3 を用いた YOLO 物体検出システムの環境構築手順を詳細に説明します。

> **前提**: このガイドに従うことで、ゼロから完全な動作環境を構築できます。

---

## 📋 目次

1. [ハードウェア要件](#1-ハードウェア要件)
2. [ソフトウェア要件](#2-ソフトウェア要件)
3. [事前準備](#3-事前準備)
4. [Raspberry Pi OS のセットアップ](#4-raspberry-pi-os-のセットアップ)
5. [システムの初期設定と更新](#5-システムの初期設定と更新)
6. [カメラモジュールのセットアップ](#6-カメラモジュールのセットアップ)
7. [Hailo-8L AI Kit のセットアップ](#7-hailo-8l-ai-kit-のセットアップ)
8. [Python 環境の構築](#8-python-環境の構築)
9. [プロジェクトのセットアップ](#9-プロジェクトのセットアップ)
10. [モデルファイルの準備](#10-モデルファイルの準備)
11. [動作確認](#11-動作確認)
12. [トラブルシューティング](#12-トラブルシューティング)

---

## 1. ハードウェア要件

### 必須コンポーネント

| コンポーネント | 仕様 | 備考 |
|--------------|------|------|
| **Raspberry Pi 5** | 4GB RAM 以上推奨 | 8GB モデル推奨（高解像度処理時） |
| **Raspberry Pi AI Kit** | Hailo-8L 搭載 | M.2 HAT+ と Hailo-8L モジュール |
| **Camera Module V3** | IMX708 センサー | 解像度: 最大 4608×2592 |
| **microSD カード** | 32GB 以上 | Class 10 / UHS-I 以上推奨 |
| **電源アダプタ** | 5V/5A（USB-C） | 公式 27W アダプタ推奨 |
| **ディスプレイ** | HDMI 対応 | セットアップ時に必要 |
| **キーボード・マウス** | USB 接続 | セットアップ時に必要 |

### オプション

- **冷却ファン / ヒートシンク**: 長時間動作時の熱対策に推奨
- **ケース**: Raspberry Pi 5 + AI Kit 対応のケース
- **USB Webcam**: Camera Module V3 の代替（ロジクール C270 等）

### ハードウェア接続

1. **M.2 HAT+ の取り付け**
   - Raspberry Pi 5 のボード上の M.2 スロットに HAT+ を接続
   - スペーサーとネジでしっかり固定

2. **Hailo-8L モジュールの装着**
   - M.2 HAT+ に Hailo-8L モジュールを挿入
   - ネジで固定

3. **Camera Module V3 の接続**
   - Raspberry Pi 5 の MIPI CSI コネクタ（CAM0 または CAM1）に接続
   - フラットケーブルの向きに注意（青いテープ側が基板側）

---

## 2. ソフトウェア要件

### オペレーティングシステム

- **Raspberry Pi OS** (64-bit, Bookworm 以降)
  - 推奨: Raspberry Pi OS (64-bit) with desktop
  - 最小バージョン: Debian 12 (Bookworm)

### システムソフトウェア

| ソフトウェア | バージョン | 用途 |
|------------|----------|------|
| **Python** | 3.11+ | アプリケーション実行環境 |
| **HailoRT SDK** | 最新版 | Hailo-8L ドライバと推論ランタイム |
| **libcamera** | 最新版 | Camera Module V3 制御 |
| **OpenCV** | 4.5+ | 画像処理 |
| **picamera2** | 最新版 | Python カメラインターフェース |

### Python ライブラリ

詳細は `requirements.txt` を参照。主要なライブラリ：
- `opencv-python` / `opencv-contrib-python`
- `numpy`
- `picamera2`
- `hailo-platform` (HailoRT SDK)

---

## 3. 事前準備

### 3.1 必要なツールの準備

- **Raspberry Pi Imager**: OS イメージの書き込みツール
  - ダウンロード: https://www.raspberrypi.com/software/

### 3.2 ネットワーク環境の確認

- インターネット接続が可能な有線 LAN または Wi-Fi 環境
- パッケージのダウンロードに数百 MB～1GB 程度必要

---

## 4. Raspberry Pi OS のセットアップ

### 4.1 OS イメージの書き込み

1. **Raspberry Pi Imager を起動**

2. **OS の選択**
   - `Raspberry Pi OS (64-bit)` を選択
   - 推奨: "Raspberry Pi OS (64-bit) with desktop"

3. **ストレージの選択**
   - microSD カードを選択

4. **詳細設定（歯車アイコン）**
   - ホスト名: `raspberrypi` または任意の名前
   - SSH を有効化: チェック
   - ユーザー名とパスワード: 任意に設定（例: `pi` / `raspberry`）
   - Wi-Fi 設定: 必要に応じて設定
   - ロケール設定: `Asia/Tokyo`, `jp` キーボード

5. **書き込み実行**
   - "書き込む" をクリック
   - 完了まで 5～10 分程度待機

### 4.2 初回起動

1. microSD カードを Raspberry Pi 5 に挿入
2. ディスプレイ、キーボード、マウスを接続
3. 電源を接続して起動
4. 初回起動ウィザードに従って基本設定を完了

---

## 5. システムの初期設定と更新

### 5.1 システムの更新

```bash
# パッケージリストの更新
sudo apt update

# システム全体のアップグレード（10～30分程度）
sudo apt full-upgrade -y

# 不要なパッケージの削除
sudo apt autoremove -y

# 再起動
sudo reboot
```

### 5.2 システム設定の確認

再起動後、以下を確認：

```bash
# OS バージョンの確認
lsb_release -a
# 期待値: Debian 12 (bookworm) 以降

# Python バージョンの確認
python3 --version
# 期待値: Python 3.11.x 以降

# カーネルバージョンの確認
uname -r
# 期待値: 6.1.x 以降
```

### 5.3 基本ツールのインストール

```bash
# 開発ツールとユーティリティのインストール
sudo apt install -y \
    git \
    build-essential \
    python3-dev \
    python3-pip \
    python3-venv \
    cmake \
    v4l-utils \
    i2c-tools

# Git の設定（任意）
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

---

## 6. カメラモジュールのセットアップ

### 6.1 カメラインターフェースの有効化

```bash
# raspi-config を起動
sudo raspi-config
```

以下の手順で設定：
1. `3 Interface Options` を選択
2. `I1 Legacy Camera` を選択 → **Disable** を選択（libcamera を使用するため）
3. `Finish` で終了
4. 再起動を促されたら `Yes`

```bash
sudo reboot
```

### 6.2 カメラの動作確認

再起動後、以下のコマンドでカメラをテスト：

```bash
# libcamera-apps のインストール（未インストールの場合）
sudo apt install -y libcamera-apps libcamera-tools

# カメラの認識確認
# ※ Debian 13 (trixie) 以降では、コマンド名が変更されています
#    旧: libcamera-hello → 新: rpicam-hello
rpicam-hello --list-cameras
# 期待値: Camera Module V3 (IMX708) が表示される

# プレビュー表示テスト（5秒間）
rpicam-hello --timeout 5000

# 静止画撮影テスト
rpicam-still -o test.jpg
```

> **注意**: Debian 12 (Bookworm) では `libcamera-hello` コマンド、
> Debian 13 (Trixie) 以降では `rpicam-hello` コマンドを使用します。

**コマンド名対応表**:
| 旧コマンド | 新コマンド | 用途 |
|-----------|-----------|------|
| `libcamera-hello` | `rpicam-hello` | カメラプレビュー |
| `libcamera-still` | `rpicam-still` | 静止画撮影 |
| `libcamera-vid` | `rpicam-vid` | 動画撮影 |

**期待される出力例** (Camera Module V3 Wide NoIR の場合):
```
Available cameras
-----------------
0 : imx708_wide_noir [4608x2592 10-bit RGGB]
    Modes: 'SRGGB10_CSI2P' : 1536x864 [120.13 fps]
                             2304x1296 [56.03 fps]
                             4608x2592 [14.35 fps]
```

### 6.3 トラブルシューティング（カメラ）

**カメラが認識されない場合**:

1. **接続の確認**
   ```bash
   # I2C デバイスの確認
   sudo i2cdetect -y 10
   # IMX708 は 0x1a に表示されるはず
   ```

2. **ファームウェアの更新**
   ```bash
   sudo rpi-update
   sudo reboot
   ```

3. **カメラの物理的な接続確認**
   - フラットケーブルが正しい向きで挿入されているか
   - コネクタのロックがかかっているか

---

## 7. Hailo-8L AI Kit のセットアップ

### 7.1 HailoRT SDK のインストール

Raspberry Pi OS では `hailo-all` メタパッケージで、必要なすべてのコンポーネントを一度にインストールできます。

```bash
# hailo-all メタパッケージのインストール（推奨）
sudo apt update
sudo apt install -y hailo-all
```

`hailo-all` パッケージに含まれるコンポーネント:
- `hailort` - HailoRT SDK（ランタイムライブラリ）
- `hailort-pcie-driver` - PCIe ドライバとファームウェア
- `hailo-tappas-core` - TAPPAS フレームワーク（GStreamer プラグイン等）
- `python3-hailort` - Python バインディング
- `rpicam-apps-hailo-postprocess` - カメラ統合プラグイン
- その他依存関係（OpenCV, GStreamer, numpy 等）

> **注意**: インストールには約 1GB のディスク容量と、ダウンロードに約 370MB 必要です。

### 7.2 Hailo デバイスの認識確認

```bash
# Hailo デバイスの確認（PCI デバイス）
lspci | grep -i hailo
# 期待値: "Hailo Technologies Ltd. Hailo-8 AI Processor" が表示される

# カーネルモジュールの確認
lsmod | grep hailo
# 期待値: hailo_pci が表示される

# Hailo ファームウェアの確認
hailortcli fw-control identify
```

**期待される出力例** (2024-12-17 確認済み):
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

### 7.3 HailoRT Python バインディングの確認

`hailo-all` パッケージにより、Python バインディングは自動的にインストールされています。

```bash
# インストール確認
python3 -c "import hailo_platform; print('HailoRT: OK')"
python3 -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python3 -c "import numpy; print(f'NumPy: {numpy.__version__}')"
```

> **重要**: `hailo-all` がインストールされた場合、OpenCV, NumPy などの主要ライブラリは
> システムレベルでインストールされます。仮想環境を使用する場合は `--system-site-packages`
> オプションを使用してください。

### 7.4 トラブルシューティング（Hailo）

**Hailo デバイスが認識されない場合**:

1. **PCIe スロットの確認**
   ```bash
   lspci
   # Hailo デバイスが表示されるか確認
   ```

2. **カーネルモジュールの確認**
   ```bash
   lsmod | grep hailo
   # hailo_pci が表示されるか確認
   ```

3. **手動でモジュールをロード**
   ```bash
   sudo modprobe hailo_pci
   ```

4. **デバイスの権限確認**
   ```bash
   ls -l /dev/hailo*
   # 読み書き権限があるか確認
   ```

---

## 8. Python 環境の構築

### 8.1 仮想環境の作成

プロジェクト専用の Python 仮想環境を作成します。

```bash
# ホームディレクトリに移動
cd ~

# プロジェクト用ディレクトリの作成（任意の場所でOK）
mkdir -p ~/projects
cd ~/projects

# 仮想環境の作成
python3 -m venv hailo-yolo-env

# 仮想環境の有効化
source hailo-yolo-env/bin/activate

# pip のアップグレード
pip install --upgrade pip setuptools wheel
```

### 8.2 仮想環境の確認

```bash
# Python のパス確認
which python3
# 期待値: /home/pi/projects/hailo-yolo-env/bin/python3

# Python バージョン確認
python3 --version
# 期待値: Python 3.11.x 以降
```

### 8.3 仮想環境の自動有効化（オプション）

`.bashrc` に以下を追加すると、ターミナル起動時に自動的に仮想環境が有効化されます。

```bash
echo "source ~/projects/hailo-yolo-env/bin/activate" >> ~/.bashrc
source ~/.bashrc
```

---

## 9. プロジェクトのセットアップ

### 9.1 リポジトリのクローン

```bash
# プロジェクトディレクトリに移動
cd ~/projects

# Git リポジトリのクローン
git clone https://github.com/Murasan201/11-002-raspi-hailo8l-yolo-detector.git

# プロジェクトディレクトリに移動
cd 11-002-raspi-hailo8l-yolo-detector
```

### 9.2 依存関係のインストール

#### Camera Module V3 版（基本）

```bash
# 仮想環境が有効化されていることを確認
source ~/projects/hailo-yolo-env/bin/activate

# 依存関係のインストール
pip install -r requirements.txt
```

#### USB Webcam 版（オプション）

USB Webcam を使用する場合は、追加のシステムパッケージが必要です。

```bash
# GStreamer 関連パッケージのインストール
sudo apt install -y \
    python3-gi \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-libav

# 自動セットアップスクリプトの実行
chmod +x setup_usb_webcam.sh
./setup_usb_webcam.sh
```

または手動でインストール：

```bash
pip install -r requirements_usb.txt
```

### 9.3 ディレクトリ構造の確認

```bash
# プロジェクト構造の表示
tree -L 2
```

**期待される構造**:
```
11-002-raspi-hailo8l-yolo-detector/
├── docs/                           # ドキュメント
│   ├── README.md
│   ├── SETUP_GUIDE.md
│   ├── CLAUDE.md
│   └── ...
├── raspi_hailo8l_yolo.py          # Camera Module V3 版
├── raspi_hailo8l_yolo_usb.py      # USB Webcam 版
├── requirements.txt                # Python 依存関係
├── requirements_usb.txt            # USB 版追加依存関係
├── setup_usb_webcam.sh             # USB Webcam セットアップスクリプト
├── README.md                       # プロジェクト README
└── LICENSE
```

---

## 10. モデルファイルの準備

### 10.1 モデルディレクトリの作成

```bash
# プロジェクトルートで実行
mkdir -p models
```

### 10.2 YOLO モデルの取得

Hailo-8L 用に最適化された YOLO モデル（`.hef` 形式）が必要です。

#### オプション1: hailo-all パッケージのモデルを使用（推奨）

`hailo-all` パッケージをインストールすると、YOLOモデルが `/usr/share/hailo-models/` にインストールされます。

```bash
# インストール済みモデルの確認
ls -lh /usr/share/hailo-models/

# models ディレクトリにシンボリックリンクを作成
mkdir -p models
ln -sf /usr/share/hailo-models/yolov8s_h8l.hef models/yolov8s_h8l.hef
ln -sf /usr/share/hailo-models/yolov6n_h8l.hef models/yolov6n_h8l.hef
ln -sf /usr/share/hailo-models/yolox_s_leaky_h8l_rpi.hef models/yolox_s_leaky_h8l_rpi.hef
```

**利用可能なモデル（hailo-all パッケージ）**:
| モデル | ファイル名 | サイズ | 用途 |
|--------|-----------|-------|------|
| YOLOv6n | `yolov6n_h8l.hef` | 14.5MB | 軽量・高速 |
| YOLOv8s | `yolov8s_h8l.hef` | 36.6MB | バランス型（推奨） |
| YOLOX-S | `yolox_s_leaky_h8l_rpi.hef` | 22.4MB | 高精度 |
| YOLOv8s-Pose | `yolov8s_pose_h8l_pi.hef` | 24.5MB | ポーズ推定 |

#### オプション2: 公式サンプルから取得

Raspberry Pi 5 向けの公式サンプルリポジトリから取得：

```bash
# 公式サンプルのクローン
cd ~/projects
git clone https://github.com/hailo-ai/hailo-rpi5-examples.git

# サンプルモデルの確認
ls hailo-rpi5-examples/resources/*.hef

# モデルファイルをコピー
cp hailo-rpi5-examples/resources/*.hef \
   ~/projects/11-002-raspi-hailo8l-yolo-detector/models/
```

#### オプション3: Hailo Model Zoo から取得

公式の Hailo Model Zoo リポジトリから取得（カスタムモデルが必要な場合）：

```bash
cd ~/projects
git clone https://github.com/hailo-ai/hailo_model_zoo.git
# 公式手順に従ってモデルを取得・変換
```

### 10.3 モデルファイルの確認

```bash
# モデルディレクトリの確認
ls -lh models/
# 期待値: .hef ファイルが表示される
```

### 10.4 推奨モデル

| モデル名 | サイズ | 速度 | 精度 | 用途 |
|---------|-------|------|------|------|
| YOLOv6n | 小(14.5MB) | 最高速 | 中 | リアルタイム検出（軽量） |
| YOLOv8s | 中(36.6MB) | 高速 | 高 | バランス型（**推奨**） |
| YOLOX-S | 中(22.4MB) | 中速 | 高 | 高精度が必要な場合 |

> **注意**: モデル名の末尾 `_h8l` は Hailo-8L 用に最適化されていることを示します。

---

## 11. 動作確認

### 11.1 カメラとHailoの総合確認

```bash
# プロジェクトディレクトリに移動
cd ~/projects/11-002-raspi-hailo8l-yolo-detector

# 仮想環境の有効化
source ~/projects/hailo-yolo-env/bin/activate

# カメラの確認
libcamera-hello --timeout 3000

# Hailo デバイスの確認
hailortcli fw-control identify
```

### 11.2 アプリケーションの実行

#### Camera Module V3 版

```bash
# 基本実行（デフォルト設定: 1280x720, 信頼度0.25）
python raspi_hailo8l_yolo.py

# カスタム設定での実行
python raspi_hailo8l_yolo.py --res 1280x720 --conf 0.25 --iou 0.45

# 動画保存を有効化
python raspi_hailo8l_yolo.py --save

# 検出ログを有効化
python raspi_hailo8l_yolo.py --log

# すべてのオプションを使用
python raspi_hailo8l_yolo.py --res 1920x1080 --conf 0.3 --iou 0.4 --save --log
```

#### USB Webcam 版

```bash
# 基本実行
python raspi_hailo8l_yolo_usb.py

# カスタム設定での実行
python raspi_hailo8l_yolo_usb.py --network yolov8s --width 1280 --height 720
```

### 11.3 動作確認のポイント

実行後、以下を確認：

1. **画面表示**
   - カメラ映像が表示される
   - 検出オブジェクトに緑色の矩形が表示される
   - 左上に FPS、解像度、推論時間が表示される

2. **パフォーマンス**
   - FPS が 10 以上（1280x720 の場合）
   - 推論時間が 50ms 以下（モデルと解像度に依存）

3. **検出精度**
   - 人、ボトル、椅子などの一般的なオブジェクトが検出される
   - 信頼度スコアが表示される

4. **終了**
   - `q` キーで正常終了できる

### 11.4 出力ファイルの確認

#### 動画保存（`--save` オプション使用時）

```bash
# 出力ディレクトリの確認
ls -lh output/
# 期待値: detection_YYYYMMDD_HHMMSS.mp4 ファイルが作成される

# 動画の再生
vlc output/detection_*.mp4
```

#### ログファイル（`--log` オプション使用時）

```bash
# ログディレクトリの確認
ls -lh logs/
# 期待値: detections_YYYYMMDD_HHMMSS.csv ファイルが作成される

# ログの内容確認
head -n 20 logs/detections_*.csv
```

**CSV フォーマット**:
```
timestamp,frame_id,class_name,confidence,x1,y1,x2,y2
2024-12-17T18:30:45.123456,1,person,0.85,100,150,300,400
2024-12-17T18:30:45.156789,1,bottle,0.72,450,200,550,350
```

---

## 12. トラブルシューティング

### 12.1 カメラ関連の問題

#### 問題: カメラが認識されない

**確認手順**:
```bash
# カメラデバイスの確認
libcamera-hello --list-cameras

# I2C デバイスの確認
sudo i2cdetect -y 10
```

**解決策**:
1. フラットケーブルの接続を確認（向きとロック）
2. カメラインターフェースが有効化されているか確認
3. ファームウェアの更新: `sudo rpi-update && sudo reboot`

#### 問題: カメラ映像が暗い / 色が不自然

**解決策**:
1. 自動露出補正の調整（アプリケーション内で設定）
2. カメラレンズの保護フィルムを剥がす
3. 照明環境の改善

### 12.2 Hailo 関連の問題

#### 問題: Hailo デバイスが認識されない

**確認手順**:
```bash
# PCI デバイスの確認
lspci | grep -i hailo

# カーネルモジュールの確認
lsmod | grep hailo

# デバイスファイルの確認
ls -l /dev/hailo*
```

**解決策**:
1. M.2 HAT+ と Hailo-8L の物理的な接続を確認
2. カーネルモジュールを手動でロード: `sudo modprobe hailo_pci`
3. HailoRT SDK を再インストール
4. 再起動: `sudo reboot`

#### 問題: 推論エラーが発生する

**エラー例**: `Failed to initialize Hailo device`

**解決策**:
1. モデルファイル（`.hef`）が正しいパスに配置されているか確認
2. モデルファイルが Hailo-8L 用にコンパイルされているか確認
3. HailoRT のバージョンとモデルの互換性を確認

### 12.3 パフォーマンス関連の問題

#### 問題: FPS が低い（5 FPS 以下）

**確認手順**:
```bash
# CPU 使用率の確認
htop

# CPU ガバナーの確認
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

**解決策**:
1. **CPU ガバナーを performance に変更**
   ```bash
   echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
   ```

2. **解像度を下げる**
   ```bash
   python raspi_hailo8l_yolo.py --res 640x480
   ```

3. **軽量モデルを使用**
   ```bash
   # YOLOv8n（最軽量）を使用
   python raspi_hailo8l_yolo.py --model models/yolov8n_hailo.hef
   ```

4. **GPU メモリを増やす**
   ```bash
   sudo nano /boot/firmware/config.txt
   # 以下を追加
   gpu_mem=256
   ```
   保存後、再起動: `sudo reboot`

#### 問題: 熱スロットリングが発生する

**確認手順**:
```bash
# CPU 温度の確認
vcgencmd measure_temp

# スロットリング状態の確認
vcgencmd get_throttled
# 0x0 なら正常、それ以外はスロットリング発生
```

**解決策**:
1. 冷却ファンやヒートシンクを追加
2. ケース内の通気性を改善
3. 高負荷時は解像度を下げる

### 12.4 Python 環境関連の問題

#### 問題: モジュールが見つからない

**エラー例**: `ModuleNotFoundError: No module named 'cv2'`

**解決策**:
```bash
# 仮想環境が有効化されているか確認
which python3
# /home/pi/projects/hailo-yolo-env/bin/python3 が表示されるべき

# 仮想環境を有効化
source ~/projects/hailo-yolo-env/bin/activate

# 依存関係を再インストール
pip install -r requirements.txt
```

#### 問題: picamera2 が動作しない

**解決策**:
```bash
# システムの picamera2 を使用
sudo apt install -y python3-picamera2

# または、pip でインストール
pip install picamera2
```

### 12.5 メモリ不足の問題

#### 問題: メモリ不足でクラッシュする

**確認手順**:
```bash
# メモリ使用量の確認
free -h

# スワップの確認
swapon --show
```

**解決策**:
1. **スワップファイルを増やす**
   ```bash
   sudo dphys-swapfile swapoff
   sudo nano /etc/dphys-swapfile
   # CONF_SWAPSIZE=2048 に変更
   sudo dphys-swapfile setup
   sudo dphys-swapfile swapon
   ```

2. **不要なアプリケーションを終了**
   ```bash
   # デスクトップ環境を無効化（CLI のみで使用）
   sudo systemctl set-default multi-user.target
   sudo reboot
   ```

3. **Raspberry Pi 5 の 8GB モデルを使用**

### 12.6 その他の問題

#### 問題: 動画保存ができない

**確認手順**:
```bash
# 出力ディレクトリが存在するか確認
ls -ld output/

# ディスク容量の確認
df -h
```

**解決策**:
```bash
# 出力ディレクトリを作成
mkdir -p output logs

# ディスク容量を確保（不要ファイルの削除）
sudo apt clean
sudo apt autoremove -y
```

#### 問題: 権限エラーが発生する

**エラー例**: `Permission denied`

**解決策**:
```bash
# ユーザーを video グループに追加
sudo usermod -aG video $USER

# ユーザーを dialout グループに追加（シリアルデバイス用）
sudo usermod -aG dialout $USER

# 再ログインまたは再起動
sudo reboot
```

---

## 13. パフォーマンス最適化（上級者向け）

### 13.1 システムの最適化

```bash
# CPU ガバナーを performance に固定
sudo systemctl disable ondemand
echo 'GOVERNOR="performance"' | sudo tee /etc/default/cpufrequtils
sudo systemctl restart cpufrequtils

# GPU メモリの増加
sudo nano /boot/firmware/config.txt
# 以下を追加
gpu_mem=256
dtoverlay=vc4-kms-v3d
max_framebuffers=2
```

### 13.2 アプリケーションの最適化

- **解像度の調整**: 1280x720 が速度と精度のバランスが良い
- **信頼度閾値の調整**: `--conf 0.3` ～ `0.5` で検出数を制御
- **軽量モデルの使用**: YOLOv8n が最速

---

## 14. 次のステップ

環境構築が完了したら、以下のドキュメントを参照してください：

- **[README.md](../README.md)** - 使用方法と各種オプションの詳細
- **[docs/CLAUDE.md](./CLAUDE.md)** - Claude Code を使った開発ワークフロー
- **[docs/python_coding_guidelines.md](./python_coding_guidelines.md)** - コーディング規約
- **[docs/COMMENT_STYLE_GUIDE.md](./COMMENT_STYLE_GUIDE.md)** - コメント記載標準

---

## 15. サポート

問題が発生した場合は、以下の情報を含めて Issue を作成してください：

- Raspberry Pi OS のバージョン: `lsb_release -a`
- Python バージョン: `python3 --version`
- HailoRT SDK バージョン: `hailortcli fw-control identify`
- エラーメッセージの全文
- 実行したコマンド
- ハードウェア構成

**リポジトリ**: https://github.com/Murasan201/11-002-raspi-hailo8l-yolo-detector

---

## 16. 参考文献（References）

本ガイドの作成にあたり、以下の公式ドキュメントおよびリソースを参照しました。

### Hailo 公式ドキュメント

1. **Hailo Developer Zone**
   - URL: https://hailo.ai/developer-zone/
   - 内容: HailoRT SDK、モデル変換ツール、開発者向けリソース

2. **HailoRT User Guide**
   - URL: https://hailo.ai/developer-zone/documentation/hailort-v4-20-0/
   - 内容: HailoRT SDK のインストール、API リファレンス、サンプルコード

3. **Hailo Model Zoo**
   - URL: https://github.com/hailo-ai/hailo_model_zoo
   - 内容: 事前学習済みモデル、モデル変換スクリプト、ベンチマーク結果

4. **hailo-rpi5-examples**
   - URL: https://github.com/hailo-ai/hailo-rpi5-examples
   - 内容: Raspberry Pi 5 + Hailo-8L 向けのサンプルコード、モデルファイル

### Raspberry Pi 公式ドキュメント

5. **Raspberry Pi Documentation - Camera**
   - URL: https://www.raspberrypi.com/documentation/accessories/camera.html
   - 内容: Camera Module V3 の仕様、libcamera/rpicam の使用方法

6. **Raspberry Pi Documentation - AI Kit**
   - URL: https://www.raspberrypi.com/documentation/accessories/ai-kit.html
   - 内容: AI Kit（Hailo-8L）のセットアップ、ハードウェア仕様

7. **Raspberry Pi OS Documentation**
   - URL: https://www.raspberrypi.com/documentation/computers/os.html
   - 内容: OS のインストール、設定、ネットワーク構成

8. **Raspberry Pi Imager**
   - URL: https://www.raspberrypi.com/software/
   - 内容: OS イメージの書き込みツール

### Python ライブラリ

9. **Picamera2 Documentation**
   - URL: https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf
   - 内容: Picamera2 ライブラリの API リファレンス、サンプルコード

10. **OpenCV Documentation**
    - URL: https://docs.opencv.org/
    - 内容: 画像処理、動画処理の API リファレンス

### YOLO モデル

11. **Ultralytics YOLOv8 Documentation**
    - URL: https://docs.ultralytics.com/
    - 内容: YOLOv8 モデルのアーキテクチャ、学習方法、エクスポート

12. **COCO Dataset**
    - URL: https://cocodataset.org/
    - 内容: 80 クラスの物体検出データセット（YOLO モデルの学習データ）

### ハードウェア仕様

13. **Raspberry Pi 5 Product Brief**
    - URL: https://datasheets.raspberrypi.com/rpi5/raspberry-pi-5-product-brief.pdf
    - 内容: Raspberry Pi 5 のハードウェア仕様

14. **IMX708 Datasheet (Camera Module V3)**
    - URL: https://www.raspberrypi.com/documentation/accessories/camera.html#imx708
    - 内容: Camera Module V3 のセンサー仕様

15. **Hailo-8L Datasheet**
    - URL: https://hailo.ai/products/ai-accelerators/hailo-8l-ai-accelerator-for-ai-light-applications/
    - 内容: Hailo-8L AI アクセラレータのハードウェア仕様（13 TOPS）

### コミュニティリソース

16. **Raspberry Pi Forums**
    - URL: https://forums.raspberrypi.com/
    - 内容: ユーザーコミュニティ、トラブルシューティング情報

17. **Hailo Community**
    - URL: https://community.hailo.ai/
    - 内容: Hailo ユーザーコミュニティ、FAQ、技術サポート

---

**最終更新**: 2024-12-17
**対応バージョン**: Raspberry Pi OS Bookworm 以降, Hailo-8L AI Kit
