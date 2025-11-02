#!/bin/bash
# Raspberry Pi 5 + Hailo-8L + USB Webcam セットアップスクリプト

echo "=== Raspberry Pi Hailo-8L USB Webcam セットアップ ==="

# システムアップデート
echo "システムを更新中..."
sudo apt update && sudo apt upgrade -y

# GStreamer関連パッケージのインストール
echo "GStreamer関連パッケージをインストール中..."
sudo apt install -y \
    python3-gi \
    python3-gi-cairo \
    gir1.2-gtk-3.0 \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-tools \
    gstreamer1.0-alsa

# V4L2関連ツールのインストール
echo "Video4Linux2ツールをインストール中..."
sudo apt install -y v4l-utils

# USBカメラの検出確認
echo "USBカメラを検出中..."
v4l2-ctl --list-devices

# 利用可能な解像度とフォーマットの確認
echo "カメラの対応フォーマットを確認中..."
v4l2-ctl --device=/dev/video0 --list-formats-ext

# Python仮想環境のセットアップ
if [ ! -d ".venv" ]; then
    echo "Python仮想環境を作成中..."
    python3 -m venv .venv
fi

echo "Python依存関係をインストール中..."
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements_usb.txt

# GStreamerのテスト
echo "GStreamerテスト中..."
gst-launch-1.0 v4l2src device=/dev/video0 ! video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! videoconvert ! autovideosink &
GSTREAMER_PID=$!
sleep 3
kill $GSTREAMER_PID 2>/dev/null

echo "=== セットアップ完了 ==="
echo ""
echo "使用方法:"
echo "source .venv/bin/activate"
echo "python raspi_hailo8l_yolo_usb.py"
echo ""
echo "利用可能なオプション:"
echo "python raspi_hailo8l_yolo_usb.py --network yolov8s --device /dev/video0 --width 1280 --height 720"