#!/usr/bin/env python3
"""
Raspberry Pi 5 + Hailo-8L YOLO物体検出アプリケーション
Hailo-8L AIアクセラレータを使用したリアルタイムYOLO物体検出システム
要件定義書: 11_002_raspi_hailo_8_l_yolo_detector.md

このスクリプトは以下の機能を提供します：
- リアルタイムカメラ映像からのYOLO物体検出
- Hailo-8L AIアクセラレータを使用した高速推論
- バウンディングボックス、クラス名、信頼度の表示
- 動画保存、ログ出力機能

使用方法：
    python raspi_hailo8l_yolo.py --res 1280x720 --conf 0.25 --iou 0.45
    python raspi_hailo8l_yolo.py --res 1280x720 --save
"""

import cv2
import numpy as np
import argparse
import time
import os
import csv
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False
    print("警告: picamera2がインストールされていません。USB Webカメラモードで動作します。")

try:
    import hailo_platform
    from hailo_platform import (HEF, VDevice, HailoStreamInterface,
                               InferVStreams, ConfigureParams)
    HAILO_AVAILABLE = True
except ImportError:
    HAILO_AVAILABLE = False
    print("警告: HailoRT SDKがインストールされていません。CPUモードで動作します。")


class YOLODetector:
    """
    YOLO物体検出器クラス（Hailo-8L対応）
    Hailo-8L AIアクセラレータを使用したリアルタイム物体検出を実行します。
    初心者向けに設計され、前処理・推論・後処理が明確に分離されています。
    """

    def __init__(self, model_path: str, conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45):
        """
        YOLODetectorの初期化

        Args:
            model_path: HEFモデルファイルのパス
            conf_threshold: 信頼度閾値
            iou_threshold: IoU閾値（NMS用）
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = None
        self.network_group = None
        self.network_group_params = None
        self.input_vstreams = None
        self.output_vstreams = None

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
            print(f"例: python raspi_hailo8l_yolo.py --model models/yolov8n_hailo.hef")
            return

        try:
            # Hailo デバイスの初期化
            self.device = VDevice()

            # HEF（Hailo Execution Format）ファイルの読み込み
            hef = HEF(self.model_path)

            # ネットワークグループの設定（モデルをHailoデバイスにロード）
            self.network_group = self.device.configure(hef)[0]
            self.network_group_params = self.network_group.create_params()

            # 入出力ストリームの準備（推論用のデータ入出力パイプライン）
            self.input_vstreams = InferVStreams.create_input_vstreams(
                self.network_group, self.network_group_params)
            self.output_vstreams = InferVStreams.create_output_vstreams(
                self.network_group, self.network_group_params)

            print("Hailo-8L デバイスが正常に初期化されました。")

        except Exception as e:
            print(f"エラー: Hailo デバイスの初期化に失敗しました: {e}")
            print("対処方法: Hailo デバイスの接続と電源を確認してください")
            print("ヒント: hailortcli fw-control identify でデバイスを確認")
            self.device = None

    def preprocess_image(self, image: np.ndarray,
                        target_size: Tuple[int, int] = (640, 640)
                        ) -> np.ndarray:
        """
        画像の前処理（リサイズ、RGB変換、正規化）

        Args:
            image (np.ndarray): 入力画像（BGRフォーマット）
            target_size (Tuple[int, int]): 目標サイズ (width, height)

        Returns:
            np.ndarray: 前処理済み画像（正規化済み、バッチ次元付き）
        """
        # リサイズ：入力画像をモデルの入力サイズに統一
        resized = cv2.resize(image, target_size)

        # RGB変換：OpenCVはBGR形式ですが、YOLOはRGB形式を期待
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # 正規化 (0-255 -> 0-1)：ニューラルネットワークは0-1範囲の入力を期待
        normalized = rgb_image.astype(np.float32) / 255.0

        # バッチ次元の追加：Hailo推論APIはバッチ処理に対応しているため、
        # （バッチサイズ, 高さ, 幅, チャンネル）形式に変更
        preprocessed = np.expand_dims(normalized, axis=0)

        return preprocessed

    def postprocess_detections(self, outputs: List[np.ndarray],
                              original_shape: Tuple[int, int],
                              input_shape: Tuple[int, int] = (640, 640)
                              ) -> List[Dict[str, Any]]:
        """
        推論結果の後処理（座標変換、NMS適用）

        Args:
            outputs (List[np.ndarray]): モデルの出力（YOLOv8形式）
            original_shape (Tuple[int, int]): 元画像のサイズ (height, width)
            input_shape (Tuple[int, int]): 入力画像のサイズ (height, width)

        Returns:
            List[Dict[str, Any]]: 検出結果のリスト。各辞書は以下を含む:
                - 'bbox' (list): バウンディングボックス座標 [x1, y1, x2, y2]
                - 'confidence' (float): 信頼度スコア（0.0-1.0）
                - 'class_id' (int): クラスID
                - 'class_name' (str): クラス名
        """
        detections = []

        if not outputs:
            return detections

        # YOLO出力の解析（YOLOv8形式）：[x_center, y_center, w, h, obj_conf, class_scores...]
        output = outputs[0]

        # スケール計算：入力サイズ(640x640)から元画像サイズへの変換係数
        # これは画像をリサイズしたため、出力座標を元画像スケールに変換する必要がある
        scale_x = original_shape[1] / input_shape[1]
        scale_y = original_shape[0] / input_shape[0]

        # 各検出結果を処理
        for detection in output:
            # detection[4] はオブジェクト確信度（物体が存在する確率）
            confidence = detection[4]

            if confidence > self.conf_threshold:
                # バウンディングボックスの座標（YOLO形式：中心座標と幅高さ）
                x_center, y_center, width, height = detection[:4]

                # 元画像サイズに変換
                x_center *= scale_x
                y_center *= scale_y
                width *= scale_x
                height *= scale_y

                # 中心座標と幅高さから左上・右下座標に変換
                x1 = int(x_center - width / 2)
                y1 = int(y_center - height / 2)
                x2 = int(x_center + width / 2)
                y2 = int(y_center + height / 2)

                # クラス確率の処理（detection[5:]に各クラスの確率が含まれる）
                class_scores = detection[5:]
                class_id = np.argmax(class_scores)
                class_confidence = class_scores[class_id]

                if class_confidence > self.conf_threshold:
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(class_confidence),
                        'class_id': int(class_id),
                        'class_name': (self.class_names[class_id]
                                     if class_id < len(self.class_names)
                                     else f'class_{class_id}')
                    })

        # NMS（Non-Maximum Suppression）の適用：重なっているボックスを除外
        detections = self._apply_nms(detections)

        return detections

    def _apply_nms(self, detections: List[Dict[str, Any]]
                   ) -> List[Dict[str, Any]]:
        """
        Non-Maximum Suppression（NMS）の適用

        Args:
            detections (List[Dict[str, Any]]): 検出結果のリスト

        Returns:
            List[Dict[str, Any]]: NMS適用後の検出結果
        """
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
        画像から物体検出を実行

        Args:
            image (np.ndarray): 入力画像（BGRフォーマット）

        Returns:
            List[Dict[str, Any]]: 検出結果のリスト
        """
        if self.device is None or not HAILO_AVAILABLE:
            # Hailoが利用できない場合、CPUモードでのダミー検出（テスト用）
            return self._dummy_detect(image)

        try:
            # 前処理：画像をモデル入力形式に変換
            preprocessed = self.preprocess_image(image)

            # 推論実行：Hailo-8Lで高速推論を実行
            with self.network_group.activate(self.network_group_params):
                # 入力データの設定
                input_dict = {name: preprocessed
                            for name in self.input_vstreams.keys()}

                # 推論実行
                output_dict = self.network_group.infer(input_dict)

                # 出力の取得
                outputs = list(output_dict.values())

            # 後処理：推論結果から物体情報を抽出
            detections = self.postprocess_detections(outputs, image.shape[:2])

            return detections

        except Exception as e:
            print(f"エラー: 推論に失敗しました: {e}")
            print("対処方法: モデルファイルとHailoデバイスの接続を確認してください")
            return []

    def _dummy_detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        CPUモード用のダミー検出（Hailo非利用時のテスト用）

        Args:
            image (np.ndarray): 入力画像

        Returns:
            List[Dict[str, Any]]: テスト用の仮想検出結果
        """
        # Hailo非利用時のテスト用仮想検出結果
        # 実運用ではCPUベースのYOLO推論を実装することを推奨
        height, width = image.shape[:2]
        return [
            {
                'bbox': [width//4, height//4, width*3//4, height*3//4],
                'confidence': 0.85,
                'class_id': 0,
                'class_name': 'person'
            }
        ]


class CameraManager:
    """
    カメラ管理クラス
    Camera Module V3またはUSB Webカメラの初期化と制御を行います。
    自動フォールバック機能により、Picamera2が利用できない場合はUSBカメラに切り替わります。
    """

    def __init__(self, resolution: Tuple[int, int] = (1280, 720),
                 device_id: int = 0):
        """
        カメラマネージャーの初期化

        Args:
            resolution (Tuple[int, int]): カメラ解像度 (width, height)
            device_id (int): カメラデバイスID
        """
        self.resolution = resolution
        self.device_id = device_id
        self.camera = None
        self.cap = None
        self.use_picamera = PICAMERA2_AVAILABLE

        self._initialize_camera()

    def _initialize_camera(self) -> None:
        """
        Picamera2（Camera Module V3）の初期化
        Picamera2が利用できない場合は、自動的にUSBカメラに切り替わります。
        """
        if self.use_picamera:
            try:
                self.camera = Picamera2()

                # Picamera2の設定（RGB888フォーマット）
                camera_config = self.camera.create_still_configuration(
                    main={"size": self.resolution, "format": "RGB888"}
                )
                self.camera.configure(camera_config)
                self.camera.start()

                print(f"Camera Module V3 を使用します。解像度: {self.resolution}")

            except Exception as e:
                print(f"警告: Camera Module V3 の初期化に失敗しました: {e}")
                print("対処方法: libcamera-hello でカメラ接続を確認してください")
                print("ヒント: raspi-config で camera インターフェースを有効化")
                self.use_picamera = False
                self._initialize_webcam()
        else:
            self._initialize_webcam()

    def _initialize_webcam(self) -> None:
        """
        USB Webカメラの初期化
        OpenCVのVideoCapture APIを使用してカメラを初期化します。
        """
        try:
            self.cap = cv2.VideoCapture(self.device_id)

            if not self.cap.isOpened():
                raise Exception(f"カメラデバイス {self.device_id} を開けません")

            # 解像度設定（リクエストベース。実際の解像度はカメラの対応状況に依存）
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

            print(f"USB Webカメラを使用します。解像度: {self.resolution}")

        except Exception as e:
            print(f"エラー: USB Webカメラの初期化に失敗しました: {e}")
            print("対処方法: カメラの接続を確認してください")
            print("ヒント: ls /dev/video* でカメラデバイスを確認")
            raise

    def read_frame(self) -> Optional[np.ndarray]:
        """
        カメラからフレームを読み取る

        Returns:
            Optional[np.ndarray]: フレーム画像（BGRフォーマット）
                読み取り失敗時はNoneを返す
        """
        try:
            if self.use_picamera and self.camera:
                # Picamera2からフレーム取得（RGB形式）
                frame = self.camera.capture_array()
                # RGB -> BGR 変換（OpenCVはBGR形式を使用）
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                return frame

            elif self.cap:
                # USB Webカメラからフレーム取得
                ret, frame = self.cap.read()
                if ret:
                    return frame
                else:
                    return None

        except Exception as e:
            print(f"エラー: フレーム読み取りに失敗しました: {e}")
            return None

    def release(self) -> None:
        """
        カメラリソースの解放
        必ずアプリケーション終了時に呼び出してください。
        """
        if self.camera:
            self.camera.stop()
            self.camera.close()

        if self.cap:
            self.cap.release()


class DetectionLogger:
    """
    検出結果ログ管理クラス
    YOLOの検出結果をCSVフォーマットで記録します。
    タイムスタンプ、フレームID、クラス名、信頼度、バウンディングボックス座標を保存します。
    """

    def __init__(self, log_dir: str = "logs"):
        """
        ログ管理の初期化

        Args:
            log_dir (str): ログディレクトリ（デフォルト: 'logs'）
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # ログファイル名（タイムスタンプ付き）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"detections_{timestamp}.csv"

        # CSVヘッダーの書き込み
        with open(self.log_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'frame_id', 'class_name', 'confidence', 'x1', 'y1', 'x2', 'y2'])

    def log_detections(self, frame_id: int,
                      detections: List[Dict[str, Any]]) -> None:
        """
        検出結果のCSVログ保存

        Args:
            frame_id (int): フレームID
            detections (List[Dict[str, Any]]): 検出結果のリスト
        """
        timestamp = datetime.now().isoformat()

        with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            for det in detections:
                bbox = det['bbox']
                writer.writerow([
                    timestamp, frame_id, det['class_name'],
                    det['confidence'], bbox[0], bbox[1], bbox[2], bbox[3]
                ])


def draw_detections(image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
    """
    画像に検出結果を描画

    Args:
        image: 入力画像
        detections: 検出結果のリスト

    Returns:
        描画済み画像
    """
    result_image = image.copy()

    for det in detections:
        bbox = det['bbox']
        x1, y1, x2, y2 = bbox
        confidence = det['confidence']
        class_name = det['class_name']

        # バウンディングボックスの描画
        color = (0, 255, 0)  # 緑色
        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)

        # ラベルの描画
        label = f"{class_name}: {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

        # ラベル背景
        cv2.rectangle(result_image,
                     (x1, y1 - label_size[1] - 10),
                     (x1 + label_size[0], y1),
                     color, -1)

        # ラベルテキスト
        cv2.putText(result_image, label,
                   (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return result_image


def draw_info(image: np.ndarray, fps: float, resolution: Tuple[int, int],
              inference_time: float) -> np.ndarray:
    """
    画像に情報表示を描画

    Args:
        image: 入力画像
        fps: フレームレート
        resolution: 解像度
        inference_time: 推論時間(ms)

    Returns:
        情報描画済み画像
    """
    result_image = image.copy()

    # 情報テキスト
    info_text = [
        f"FPS: {fps:.1f}",
        f"Resolution: {resolution[0]}x{resolution[1]}",
        f"Inference: {inference_time:.1f}ms"
    ]

    # 背景矩形
    text_height = 25
    bg_height = len(info_text) * text_height + 10
    cv2.rectangle(result_image, (10, 10), (300, bg_height), (0, 0, 0), -1)

    # テキスト描画
    for i, text in enumerate(info_text):
        y_pos = 30 + i * text_height
        cv2.putText(result_image, text, (15, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return result_image


def parse_resolution(resolution_str: str) -> Tuple[int, int]:
    """
    解像度文字列をパース

    Args:
        resolution_str: "1280x720" 形式の文字列

    Returns:
        (width, height) タプル
    """
    try:
        width, height = map(int, resolution_str.split('x'))
        return (width, height)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid resolution format: {resolution_str}")


def main():
    """
    メイン関数：コマンドライン引数を処理してYOLO物体検出を実行
    Hailo-8Lを使用したリアルタイム推論とビデオ出力を管理します。
    """
    parser = argparse.ArgumentParser(
        description="Raspberry Pi 5 + Hailo-8L AI Kit YOLO物体検出",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--model', type=str,
                       default='models/yolov8n_hailo.hef',
                       help='HEFモデルファイルのパス')

    parser.add_argument('--res', type=parse_resolution,
                       default=(1280, 720),
                       help='カメラ解像度 (例: 1280x720)')

    parser.add_argument('--conf', type=float, default=0.25,
                       help='信頼度閾値')

    parser.add_argument('--iou', type=float, default=0.45,
                       help='IoU閾値（NMS用）')

    parser.add_argument('--device', type=int, default=0,
                       help='カメラデバイスID')

    parser.add_argument('--save', action='store_true',
                       help='動画を保存する')

    parser.add_argument('--log', action='store_true',
                       help='検出結果をCSVログに保存する')

    args = parser.parse_args()

    print("=== Raspberry Pi Hailo-8L YOLO Detector ===")
    print(f"モデル: {args.model}")
    print(f"解像度: {args.res[0]}x{args.res[1]}")
    print(f"信頼度閾値: {args.conf}")
    print(f"IoU閾値: {args.iou}")

    try:
        # YOLODetectorの初期化
        detector = YOLODetector(args.model, args.conf, args.iou)

        # カメラマネージャーの初期化
        camera = CameraManager(args.res, args.device)

        # ログ管理の初期化
        logger = None
        if args.log:
            logger = DetectionLogger()
            print(f"ログファイル: {logger.log_file}")

        # 動画保存の準備
        video_writer = None
        if args.save:
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = output_dir / f"detection_{timestamp}.mp4"

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(output_file), fourcc, 20.0, args.res)
            print(f"動画保存: {output_file}")

        # FPS計算用変数
        fps_counter = 0
        fps_time = time.time()
        current_fps = 0.0
        frame_id = 0

        print("\n物体検出を開始します。'q'キーで終了。")

        while True:
            # フレーム読み取り
            frame = camera.read_frame()
            if frame is None:
                print("フレームを取得できませんでした")
                break

            # 推論時間の測定
            inference_start = time.time()
            detections = detector.detect(frame)
            inference_time = (time.time() - inference_start) * 1000  # ms

            # 検出結果の描画
            result_frame = draw_detections(frame, detections)

            # 情報表示の描画
            result_frame = draw_info(result_frame, current_fps, args.res, inference_time)

            # ログ保存
            if logger:
                logger.log_detections(frame_id, detections)

            # 動画保存
            if video_writer:
                video_writer.write(result_frame)

            # 画面表示
            cv2.imshow('Hailo YOLO Detection', result_frame)

            # FPS計算
            fps_counter += 1
            if fps_counter % 10 == 0:
                current_time = time.time()
                current_fps = 10 / (current_time - fps_time)
                fps_time = current_time

            frame_id += 1

            # キー入力チェック
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

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
        if 'camera' in locals():
            camera.release()

        if video_writer:
            video_writer.release()

        cv2.destroyAllWindows()
        print("アプリケーションを終了しました")


if __name__ == "__main__":
    main()