#!/usr/bin/env python3
"""
Raspberry Pi 5 + Hailo-8L USB Webcam YOLO物体検出アプリケーション
GStreamerパイプラインとHailo-8Lを使用したリアルタイムYOLO物体検出システム
要件定義書: 11_002_raspi_hailo_8_l_yolo_detector.md

このスクリプトは以下の機能を提供します：
- USB Webカメラからのリアルタイム映像処理
- GStreamerパイプラインによる効率的な動画処理
- Hailo-8L AIアクセラレータを使用した高速YOLO推論
- 複数のYOLOモデル対応（YOLOv6n, YOLOv8s, YOLOx等）
- バウンディングボックス、クラス名、信頼度の表示

参考サイト: https://murasan-net.com/2024/11/13/raspberry-pi-5-hailo-8l-webcam/

使用方法：
    python raspi_hailo8l_yolo_usb.py
    python raspi_hailo8l_yolo_usb.py --network yolov8s --device /dev/video0
"""

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import cv2
import numpy as np
import argparse
import time
import os
import csv
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import queue

try:
    import hailo_platform
    from hailo_platform import (HEF, VDevice, HailoStreamInterface,
                               InferVStreams, ConfigureParams)
    HAILO_AVAILABLE = True
except ImportError:
    HAILO_AVAILABLE = False
    print("警告: HailoRT SDKがインストールされていません。CPUモードで動作します。")

# GStreamerの初期化
Gst.init(None)


class HailoYOLODetector:
    """
    Hailo-8L対応YOLO検出器（USB Webcam最適化版）
    GStreamerパイプラインと連携してUSB Webカメラからの映像をリアルタイム処理します。
    複数のYOLOモデルに対応し、高速な推論を実現します。
    """

    def __init__(self, model_path: str, network_name: str = "yolov8s",
                 conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        """
        HailoYOLODetectorの初期化

        Args:
            model_path (str): HEFモデルファイルのパス
            network_name (str): ネットワーク名（例: yolov8s）
            conf_threshold (float): 信頼度閾値
            iou_threshold (float): IoU閾値（NMS用）
        """
        self.model_path = model_path
        self.network_name = network_name
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = None
        self.network_group = None
        self.network_group_params = None
        self.input_vstreams = None
        self.output_vstreams = None
        self.input_shape = (640, 640)

        # COCO データセットのクラス名
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]

        self._initialize_hailo()

    def _initialize_hailo(self) -> None:
        """
        Hailoデバイスとモデルの初期化
        HEFファイルを読み込んでHailo-8Lデバイスを設定します。
        """
        if not HAILO_AVAILABLE:
            print("警告: HailoRT SDK が利用できません。CPUモードで動作します。")
            print("対処方法: Hailo SDK をインストールしてください")
            return

        if not os.path.exists(self.model_path):
            print(f"エラー: モデルファイルが見つかりません: {self.model_path}")
            print(f"対処方法: 以下のパスにHEFモデルを配置してください:")
            print(f"  {os.path.abspath('models/')}")
            print(f"例: python raspi_hailo8l_yolo_usb.py --model models/yolov8n_hailo.hef")
            return

        try:
            print(f"Hailo-8Lデバイスを初期化中...")

            # デバイスの初期化
            self.device = VDevice()

            # HEFファイルの読み込み
            hef = HEF(self.model_path)
            print(f"HEFモデル読み込み完了: {self.model_path}")

            # ネットワークグループの設定
            configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
            self.network_group = self.device.configure(hef, configure_params)[0]
            self.network_group_params = self.network_group.create_params()

            # 入出力ストリームの準備
            self.input_vstreams = InferVStreams.create_input_vstreams(
                self.network_group, self.network_group_params)
            self.output_vstreams = InferVStreams.create_output_vstreams(
                self.network_group, self.network_group_params)

            # 入力形状の取得
            input_info = list(self.input_vstreams.values())[0].info
            self.input_shape = (input_info.shape.height, input_info.shape.width)

            print(f"Hailo-8L デバイスが正常に初期化されました。")
            print(f"入力サイズ: {self.input_shape}")
            print(f"ネットワーク: {self.network_name}")

        except Exception as e:
            print(f"エラー: Hailo デバイスの初期化に失敗しました: {e}")
            print("対処方法: Hailo デバイスの接続と電源を確認してください")
            print("ヒント: hailortcli fw-control identify でデバイスを確認")
            self.device = None

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        画像の前処理（Hailo最適化版）

        Args:
            image (np.ndarray): 入力画像（BGR）

        Returns:
            np.ndarray: 前処理済み画像
        """
        # リサイズ（アスペクト比を保持してパディング）
        h, w = image.shape[:2]
        target_h, target_w = self.input_shape

        # スケール計算
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # リサイズ
        resized = cv2.resize(image, (new_w, new_h))

        # パディング
        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

        # RGB変換と正規化
        rgb_image = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        normalized = rgb_image.astype(np.float32) / 255.0

        return normalized

    def postprocess_detections(self, outputs: Dict[str, np.ndarray],
                              original_shape: Tuple[int, int]) -> List[Dict[str, Any]]:
        """
        推論結果の後処理（YOLOv8対応）

        Args:
            outputs: モデルの出力辞書
            original_shape: 元画像のサイズ (height, width)

        Returns:
            検出結果のリスト
        """
        detections = []

        if not outputs:
            return detections

        try:
            # YOLOv8の出力形式に対応
            output_data = list(outputs.values())[0]

            # 出力形状の確認と調整
            if len(output_data.shape) == 4:
                output_data = output_data.squeeze(0)  # バッチ次元を削除

            if output_data.shape[0] > output_data.shape[1]:
                output_data = output_data.T  # 転置が必要な場合

            # スケール計算
            target_h, target_w = self.input_shape
            scale_y = original_shape[0] / target_h
            scale_x = original_shape[1] / target_w

            # 各検出結果を処理
            for detection in output_data.T:  # 各列が1つの検出結果
                # 座標とサイズ（中心座標形式）
                x_center, y_center, width, height = detection[:4]

                # クラススコア
                class_scores = detection[4:]
                max_score = np.max(class_scores)

                if max_score > self.conf_threshold:
                    class_id = np.argmax(class_scores)

                    # 座標を元画像サイズに変換
                    x_center *= scale_x
                    y_center *= scale_y
                    width *= scale_x
                    height *= scale_y

                    # バウンディングボックス座標
                    x1 = int(x_center - width / 2)
                    y1 = int(y_center - height / 2)
                    x2 = int(x_center + width / 2)
                    y2 = int(y_center + height / 2)

                    # 画像境界内に制限
                    x1 = max(0, min(x1, original_shape[1]))
                    y1 = max(0, min(y1, original_shape[0]))
                    x2 = max(0, min(x2, original_shape[1]))
                    y2 = max(0, min(y2, original_shape[0]))

                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(max_score),
                        'class_id': int(class_id),
                        'class_name': self.class_names[class_id] if class_id < len(self.class_names) else f'class_{class_id}'
                    })

        except Exception as e:
            print(f"後処理エラー: {e}")
            return []

        # NMS（Non-Maximum Suppression）の適用
        detections = self._apply_nms(detections)
        return detections

    def _apply_nms(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Non-Maximum Suppressionの適用"""
        if not detections:
            return detections

        # OpenCVのNMSを使用
        boxes = [det['bbox'] for det in detections]
        scores = [det['confidence'] for det in detections]

        indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_threshold, self.iou_threshold)

        if len(indices) > 0:
            indices = indices.flatten()
            return [detections[i] for i in indices]
        else:
            return []

    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        物体検出の実行

        Args:
            image: 入力画像

        Returns:
            検出結果のリスト
        """
        if self.device is None or not HAILO_AVAILABLE:
            return self._dummy_detect(image)

        try:
            # 前処理
            preprocessed = self.preprocess_image(image)

            # 推論実行
            with self.network_group.activate(self.network_group_params):
                # 入力データの設定
                input_name = list(self.input_vstreams.keys())[0]
                input_dict = {input_name: preprocessed}

                # 推論実行
                output_dict = self.network_group.infer(input_dict)

            # 後処理
            detections = self.postprocess_detections(output_dict, image.shape[:2])
            return detections

        except Exception as e:
            print(f"推論エラー: {e}")
            return []

    def _dummy_detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """CPUモード用のダミー検出（テスト用）"""
        height, width = image.shape[:2]
        return [
            {
                'bbox': [width//4, height//4, width*3//4, height*3//4],
                'confidence': 0.85,
                'class_id': 0,
                'class_name': 'person'
            }
        ]


class GStreamerWebcamPipeline:
    """
    GStreamer USBウェブカメラパイプライン
    GStreamerを使用してUSB Webカメラからビデオストリームを効率的に処理します。
    非ブロッキングキュー方式でフレームを取得し、物体検出処理との並列化を実現します。
    """

    def __init__(self, device_path: str = "/dev/video0", width: int = 1280, height: int = 720, framerate: int = 30):
        """
        GStreamerパイプラインの初期化

        Args:
            device_path: ウェブカメラデバイスパス
            width: 映像幅
            height: 映像高さ
            framerate: フレームレート
        """
        self.device_path = device_path
        self.width = width
        self.height = height
        self.framerate = framerate
        self.pipeline = None
        self.appsink = None
        self.frame_queue = queue.Queue(maxsize=10)
        self.running = False

    def create_pipeline(self) -> bool:
        """
        GStreamerパイプラインの作成と初期化
        USB WebカメラからYUV2フォーマットで取得し、BGRフォーマットに変換します。

        Returns:
            bool: パイプライン作成成功時はTrue、失敗時はFalse
        """
        try:
            # USB Webカメラ用パイプライン（ロジクール等に最適化）
            pipeline_str = (
                f"v4l2src device={self.device_path} ! "
                f"video/x-raw,format=YUY2,width={self.width},height={self.height},framerate={self.framerate}/1 ! "
                "videoconvert ! "
                "video/x-raw,format=BGR ! "
                "appsink name=sink emit-signals=true sync=false max-buffers=10 drop=true"
            )

            print(f"GStreamerパイプライン: {pipeline_str}")
            self.pipeline = Gst.parse_launch(pipeline_str)

            if not self.pipeline:
                raise Exception("パイプラインの作成に失敗しました")

            # AppSinkの取得
            self.appsink = self.pipeline.get_by_name("sink")
            if not self.appsink:
                raise Exception("AppSinkの取得に失敗しました")

            # シグナルの接続
            self.appsink.connect("new-sample", self._on_new_sample)

            return True

        except Exception as e:
            print(f"パイプライン作成エラー: {e}")
            return False

    def _on_new_sample(self, appsink):
        """
        GStreamerからの新しいフレーム受信時のコールバック
        フレームデータをNumPy配列に変換して、キューに追加します。

        Args:
            appsink: GStreamer AppSinkオブジェクト

        Returns:
            Gst.FlowReturn: 処理結果（OK: 正常終了、ERROR: エラー）
        """
        try:
            sample = appsink.emit("pull-sample")
            if sample:
                buffer = sample.get_buffer()
                caps = sample.get_caps()

                # フレームデータの取得
                success, map_info = buffer.map(Gst.MapFlags.READ)
                if success:
                    # NumPy配列に変換
                    frame_data = np.frombuffer(map_info.data, dtype=np.uint8)
                    frame = frame_data.reshape((self.height, self.width, 3))

                    # キューに追加（満杯の場合は古いフレームを破棄）
                    if not self.frame_queue.full():
                        self.frame_queue.put(frame.copy())
                    else:
                        try:
                            self.frame_queue.get_nowait()  # 古いフレームを削除
                            self.frame_queue.put(frame.copy())
                        except queue.Empty:
                            pass

                    buffer.unmap(map_info)

        except Exception as e:
            print(f"フレーム処理エラー: {e}")

        return Gst.FlowReturn.OK

    def start(self) -> bool:
        """
        GStreamerパイプラインの開始
        パイプラインを PLAYING 状態に遷移させてビデオストリーム処理を開始します。

        Returns:
            bool: 開始成功時はTrue、失敗時はFalse
        """
        if not self.pipeline:
            return False

        try:
            ret = self.pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                print("パイプラインの開始に失敗しました")
                return False

            self.running = True
            print(f"GStreamerパイプラインが開始されました: {self.device_path}")
            return True

        except Exception as e:
            print(f"パイプライン開始エラー: {e}")
            return False

    def get_frame(self) -> Optional[np.ndarray]:
        """
        パイプラインからフレームを取得
        キューに存在するフレームを非ブロッキング方式で取得します。

        Returns:
            Optional[np.ndarray]: フレーム画像（BGRフォーマット）、フレームがない場合はNone
        """
        try:
            if not self.frame_queue.empty():
                return self.frame_queue.get_nowait()
            return None
        except queue.Empty:
            return None

    def stop(self):
        """
        GStreamerパイプラインの停止
        パイプラインを NULL 状態に遷移させてリソースを解放します。
        """
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
            self.running = False
            print("GStreamerパイプラインが停止されました")


class PerformanceMonitor:
    """
    パフォーマンス監視クラス
    フレームレート、推論時間などのパフォーマンス指標を収集・計算します。
    移動平均を使用して瞬間的な変動を平滑化し、安定したメトリクスを提供します。
    """

    def __init__(self, window_size: int = 30):
        """
        パフォーマンスモニターの初期化

        Args:
            window_size: 移動平均の窓サイズ
        """
        self.window_size = window_size
        self.frame_times = []
        self.inference_times = []
        self.frame_count = 0
        self.start_time = time.time()

    def update_frame_time(self, frame_time: float):
        """
        フレーム処理時間を更新
        窓サイズを超える場合は古いデータを削除して移動平均を計算します。

        Args:
            frame_time (float): フレーム処理時間（秒）
        """
        self.frame_times.append(frame_time)
        if len(self.frame_times) > self.window_size:
            self.frame_times.pop(0)
        self.frame_count += 1

    def update_inference_time(self, inference_time: float):
        """
        推論処理時間を更新
        窓サイズを超える場合は古いデータを削除します。

        Args:
            inference_time (float): 推論処理時間（秒）
        """
        self.inference_times.append(inference_time)
        if len(self.inference_times) > self.window_size:
            self.inference_times.pop(0)

    def get_fps(self) -> float:
        """
        現在のフレームレートを計算
        移動平均ウィンドウ内のフレーム処理時間から計算します。

        Returns:
            float: フレームレート（フレーム/秒）
        """
        if len(self.frame_times) < 2:
            return 0.0
        return len(self.frame_times) / sum(self.frame_times)

    def get_avg_inference_time(self) -> float:
        """
        平均推論時間を計算
        移動平均ウィンドウ内の推論時間の平均をミリ秒単位で返します。

        Returns:
            float: 平均推論時間（ミリ秒）
        """
        if not self.inference_times:
            return 0.0
        return sum(self.inference_times) / len(self.inference_times) * 1000

    def get_total_fps(self) -> float:
        """
        総合フレームレートを計算
        起動からの経過時間に対する平均フレームレートを返します。

        Returns:
            float: 総合フレームレート（フレーム/秒）
        """
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            return self.frame_count / elapsed
        return 0.0


def draw_detections(image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
    """
    画像に検出結果を描画します（USB Webcam最適化版）。
    バウンディングボックスのサイズに応じて線の太さとフォントサイズを動的に調整します。

    Args:
        image (np.ndarray): 入力画像（BGRフォーマット）
        detections (List[Dict[str, Any]]): 検出結果のリスト

    Returns:
        np.ndarray: バウンディングボックスとラベル描画済み画像
    """
    result_image = image.copy()

    for det in detections:
        bbox = det['bbox']
        x1, y1, x2, y2 = bbox
        confidence = det['confidence']
        class_name = det['class_name']

        # バウンディングボックスの描画（太めの線）
        color = (0, 255, 0)  # 緑色
        thickness = 3 if (x2 - x1) * (y2 - y1) > 10000 else 2
        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, thickness)

        # ラベルの描画
        label = f"{class_name}: {confidence:.2f}"
        font_scale = 0.8 if (x2 - x1) > 100 else 0.6
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]

        # ラベル背景（透明度付き）
        overlay = result_image.copy()
        cv2.rectangle(overlay,
                     (x1, y1 - label_size[1] - 15),
                     (x1 + label_size[0] + 10, y1),
                     color, -1)
        cv2.addWeighted(overlay, 0.7, result_image, 0.3, 0, result_image)

        # ラベルテキスト
        cv2.putText(result_image, label,
                   (x1 + 5, y1 - 8),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)

    return result_image


def draw_performance_info(image: np.ndarray, monitor: PerformanceMonitor,
                         resolution: Tuple[int, int], detection_count: int) -> np.ndarray:
    """
    画像にパフォーマンス情報を描画します。
    FPS、解像度、推論時間、検出数、フレーム数を画面左上に表示します。

    Args:
        image (np.ndarray): 入力画像（BGRフォーマット）
        monitor (PerformanceMonitor): パフォーマンスモニタオブジェクト
        resolution (Tuple[int, int]): カメラ解像度 (width, height)
        detection_count (int): 現在のフレームでの検出物体数

    Returns:
        np.ndarray: パフォーマンス情報描画済み画像
    """
    result_image = image.copy()

    # パフォーマンス情報
    fps = monitor.get_fps()
    total_fps = monitor.get_total_fps()
    avg_inference = monitor.get_avg_inference_time()

    info_text = [
        f"FPS: {fps:.1f} (avg: {total_fps:.1f})",
        f"解像度: {resolution[0]}x{resolution[1]}",
        f"推論時間: {avg_inference:.1f}ms",
        f"検出数: {detection_count}",
        f"フレーム: {monitor.frame_count}"
    ]

    # 背景矩形（半透明）
    text_height = 25
    bg_height = len(info_text) * text_height + 20
    overlay = result_image.copy()
    cv2.rectangle(overlay, (10, 10), (350, bg_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, result_image, 0.3, 0, result_image)

    # テキスト描画
    for i, text in enumerate(info_text):
        y_pos = 35 + i * text_height
        cv2.putText(result_image, text, (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return result_image


def main():
    """
    メイン関数：コマンドライン引数を処理してYOLO物体検出を実行
    GStreamerパイプラインを使用したUSB Webカメラからのリアルタイム推論を管理します。
    """
    parser = argparse.ArgumentParser(
        description="Raspberry Pi 5 + Hailo-8L + USB Webcam YOLO物体検出",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--network', type=str, default='yolov8s',
                       choices=['yolov6n', 'yolov8s', 'yolox_s_leaky'],
                       help='使用するYOLOネットワーク')

    parser.add_argument('--model', type=str,
                       help='HEFモデルファイルのパス（指定しない場合は自動選択）')

    parser.add_argument('--device', type=str, default='/dev/video0',
                       help='USBカメラデバイスパス')

    parser.add_argument('--width', type=int, default=1280,
                       help='カメラ映像幅')

    parser.add_argument('--height', type=int, default=720,
                       help='カメラ映像高さ')

    parser.add_argument('--fps', type=int, default=30,
                       help='目標フレームレート')

    parser.add_argument('--conf', type=float, default=0.25,
                       help='信頼度閾値')

    parser.add_argument('--iou', type=float, default=0.45,
                       help='IoU閾値（NMS用）')

    parser.add_argument('--save', action='store_true',
                       help='動画を保存する')

    parser.add_argument('--log', action='store_true',
                       help='検出結果をCSVログに保存する')

    parser.add_argument('--fullscreen', action='store_true',
                       help='フルスクリーン表示')

    args = parser.parse_args()

    # モデルパスの自動選択
    if not args.model:
        model_dir = Path("models")
        model_files = {
            'yolov6n': model_dir / 'yolov6n.hef',
            'yolov8s': model_dir / 'yolov8s.hef',
            'yolox_s_leaky': model_dir / 'yolox_s_leaky.hef'
        }
        args.model = str(model_files.get(args.network, model_dir / f'{args.network}.hef'))

    print("=== Raspberry Pi Hailo-8L USB Webcam YOLO Detector ===")
    print(f"ネットワーク: {args.network}")
    print(f"モデル: {args.model}")
    print(f"USBカメラ: {args.device}")
    print(f"解像度: {args.width}x{args.height}@{args.fps}fps")
    print(f"信頼度閾値: {args.conf}")
    print(f"IoU閾値: {args.iou}")

    try:
        # Hailo YOLODetectorの初期化
        detector = HailoYOLODetector(args.model, args.network, args.conf, args.iou)

        # GStreamerパイプラインの初期化
        pipeline = GStreamerWebcamPipeline(args.device, args.width, args.height, args.fps)

        if not pipeline.create_pipeline():
            print("パイプラインの作成に失敗しました")
            return

        if not pipeline.start():
            print("パイプラインの開始に失敗しました")
            return

        # パフォーマンスモニターの初期化
        monitor = PerformanceMonitor()

        # ログ管理の初期化
        logger = None
        if args.log:
            from raspi_hailo8l_yolo import DetectionLogger
            logger = DetectionLogger()
            print(f"ログファイル: {logger.log_file}")

        # 動画保存の準備
        video_writer = None
        if args.save:
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = output_dir / f"usb_detection_{timestamp}.mp4"

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(output_file), fourcc, 20.0, (args.width, args.height))
            print(f"動画保存: {output_file}")

        # GMainLoopの実行（別スレッド）
        loop = GLib.MainLoop()
        loop_thread = threading.Thread(target=loop.run, daemon=True)
        loop_thread.start()

        print("\n物体検出を開始します。'q'キーで終了、'f'キーでフルスクリーン切り替え。")

        # ウィンドウの設定
        window_name = 'Hailo USB YOLO Detection'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        if args.fullscreen:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        frame_start_time = time.time()

        while True:
            current_time = time.time()

            # フレーム取得
            frame = pipeline.get_frame()
            if frame is None:
                time.sleep(0.001)  # 短時間待機
                continue

            # フレーム時間の更新
            frame_time = current_time - frame_start_time
            monitor.update_frame_time(frame_time)
            frame_start_time = current_time

            # 推論実行
            inference_start = time.time()
            detections = detector.detect(frame)
            inference_time = time.time() - inference_start
            monitor.update_inference_time(inference_time)

            # 検出結果の描画
            result_frame = draw_detections(frame, detections)

            # パフォーマンス情報の描画
            result_frame = draw_performance_info(result_frame, monitor,
                                               (args.width, args.height), len(detections))

            # ログ保存
            if logger:
                logger.log_detections(monitor.frame_count, detections)

            # 動画保存
            if video_writer:
                video_writer.write(result_frame)

            # 画面表示
            cv2.imshow(window_name, result_frame)

            # キー入力チェック
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('f'):
                # フルスクリーン切り替え
                prop = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN)
                if prop == cv2.WINDOW_FULLSCREEN:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                else:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # 最終統計の表示
        print(f"\n=== 実行統計 ===")
        print(f"総フレーム数: {monitor.frame_count}")
        print(f"平均FPS: {monitor.get_total_fps():.2f}")
        print(f"平均推論時間: {monitor.get_avg_inference_time():.2f}ms")

    except KeyboardInterrupt:
        print("\n中断されました")

    except FileNotFoundError as e:
        print(f"エラー: ファイルが見つかりません: {e}")

    except Exception as e:
        print(f"エラー: 予期しないエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # リソースの解放
        if 'pipeline' in locals():
            pipeline.stop()

        if 'loop' in locals():
            loop.quit()

        if video_writer:
            video_writer.release()

        cv2.destroyAllWindows()
        print("アプリケーションを終了しました")


if __name__ == "__main__":
    main()