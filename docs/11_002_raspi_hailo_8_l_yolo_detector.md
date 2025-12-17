# 要件定義書
**Project:** 11-002-raspi-hailo8l-yolo-detector
**目的:** Raspberry Pi 5 + 公式 AI Kit（Hailo-8L）+ Raspberry Pi Camera Module V3 を用いて、YOLO によるリアルタイム物体検出を行う学習/実用向けアプリを開発する。

**最終更新:** 2025-12-17
**バージョン:** 1.1.0

## 1. 背景・参照
- 参照記事（作者ブログ）: Raspberry Pi 5 + Hailo-8L + Webcam で YOLO 物体検出の実践内容
- ベースとなる公式プロジェクト: [hailo-rpi5-examples (GitHub)](https://github.com/hailo-ai/hailo-rpi5-examples)
- 本リポジトリは上記記事・公式リポジトリの手順/知見を再現・整理し、**初心者でもセットアップ〜動作確認まで迷わない**ことを狙う。

## 2. スコープ

### 対象ハードウェア
- Raspberry Pi 5
- Raspberry Pi AI Kit（Hailo-8L）
- Raspberry Pi Camera Module V3（IMX708）
- USB Webカメラ（ロジクール等、オプション）

### 対象ソフトウェア
- Raspberry Pi OS (Bookworm 以降、Debian 13 trixie 対応)
- HailoRT SDK v4.x（`hailo-all` パッケージ）
- Python 3.11+
- Picamera2（Camera Module V3 用）
- OpenCV
- GStreamer（USB Webcam 版用）

### アプリケーション構成
- **Camera Module V3 版**: `raspi_hailo8l_yolo.py` - Picamera2 + HailoRT 直接API
- **USB Webcam 版**: `raspi_hailo8l_yolo_usb.py` - GStreamer パイプライン

## 3. 成果物
- 動作可能なアプリケーション
  - `raspi_hailo8l_yolo.py`（Camera Module V3 版）
  - `raspi_hailo8l_yolo_usb.py`（USB Webcam 版）
- `README.md`（セットアップ～実行手順）
- `requirements.txt`（Python 依存）
- `docs/` ディレクトリ（技術ドキュメント一式）
  - `SETUP_GUIDE.md` - 環境構築ガイド
  - `TROUBLESHOOTING.md` - トラブルシューティング
  - `CLAUDE.md` - Claude Code 開発ルール
  - `python_coding_guidelines.md` - Python コーディング規約
  - `COMMENT_STYLE_GUIDE.md` - コメント記載標準
- （任意）`assets/` デモ画像・スクリーンショット

## 4. 機能要件（FR）

### FR-1: カメラ入力
- **Camera Module V3**: Picamera2 経由で映像取得（RGB888 フォーマット）
- **USB Webcam**: OpenCV VideoCapture または GStreamer パイプライン
- 解像度プリセット: 640×480 / 1280×720 / 1920×1080（引数で指定）
- フレームレート: 目標 20–30 FPS（実機性能に依存）
- **カメラ反転**: `--flip` オプションで上下反転対応（逆さま設置時）

### FR-2: 推論（YOLO × Hailo-8L）
- Hailo-8L 用に最適化済みの YOLO モデルを **HEF（Hailo Execution Format）** 形式でロード
- **HailoRT v4.x API** を使用:
  - `VDevice` - デバイス管理
  - `HEF` - モデルファイル読み込み
  - `InferVStreams` - 推論実行
  - `InputVStreamParams` / `OutputVStreamParams` - 入出力パラメータ
- 前処理: リサイズ（640x640）、RGB変換、UINT8形式
- 後処理: NMS後処理済み出力の座標変換（正規化座標→ピクセル座標）

### FR-3: 描画/出力
- 画面表示（バウンディングボックス、クラス名、スコア）
- オプション: 動画保存（`--save`）→ `output/detection_YYYYMMDD_HHMMSS.mp4`
- オプション: ログ出力（`--log`）→ `logs/detections_YYYYMMDD_HHMMSS.csv`

### FR-4: 起動オプション

#### Camera Module V3 版共通オプション
| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--model` | `models/yolov8s_h8l.hef` | HEFモデルファイルのパス |
| `--res` | `1280x720` | カメラ解像度 |
| `--conf` | `0.25` | 信頼度閾値（0.0-1.0） |
| `--iou` | `0.45` | IoU閾値（NMS用） |
| `--device` | `0` | カメラデバイスID |
| `--flip` | - | カメラ映像を上下反転 |
| `--save` | - | 動画保存を有効化 |
| `--log` | - | CSVログ保存を有効化 |
| `--classes` | - | 検出対象クラス（スペース区切り） |
| `--list-classes` | - | 使用可能なクラス一覧を表示 |

#### USB Webcam 版専用オプション
| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--network` | `yolov8s` | YOLOネットワーク種別 |
| `--width` | `1280` | カメラ映像幅 |
| `--height` | `720` | カメラ映像高さ |
| `--fps` | `30` | 目標フレームレート |
| `--fullscreen` | - | フルスクリーン表示 |

### FR-5: 操作
- **終了**: `q` キー、`Q` キー、または `ESC` キーで安全終了
- **フルスクリーン切替**（USB版）: `f` キー

### FR-6: クラスフィルタリング
特定の種類のオブジェクトのみを検出する機能。

#### 機能概要
- **デフォルト**: 全80クラス（COCO データセット）を検出
- **フィルタリング**: `--classes` オプションで検出対象を限定
- **クラス一覧表示**: `--list-classes` オプションで使用可能なクラス名を確認

#### 使用例
```bash
# 人物のみ検出
python raspi_hailo8l_yolo.py --classes person

# 複数クラスを指定（人物、車、犬）
python raspi_hailo8l_yolo.py --classes person car dog

# 使用可能なクラス一覧を表示
python raspi_hailo8l_yolo.py --list-classes
```

#### COCO 80クラス一覧
| ID | クラス名 | ID | クラス名 | ID | クラス名 | ID | クラス名 |
|----|----------|----|-----------|----|----------|----|-----------|
| 0 | person | 20 | elephant | 40 | wine glass | 60 | dining table |
| 1 | bicycle | 21 | bear | 41 | cup | 61 | toilet |
| 2 | car | 22 | zebra | 42 | fork | 62 | tv |
| 3 | motorcycle | 23 | giraffe | 43 | knife | 63 | laptop |
| 4 | airplane | 24 | backpack | 44 | spoon | 64 | mouse |
| 5 | bus | 25 | umbrella | 45 | bowl | 65 | remote |
| 6 | train | 26 | handbag | 46 | banana | 66 | keyboard |
| 7 | truck | 27 | tie | 47 | apple | 67 | cell phone |
| 8 | boat | 28 | suitcase | 48 | sandwich | 68 | microwave |
| 9 | traffic light | 29 | frisbee | 49 | orange | 69 | oven |
| 10 | fire hydrant | 30 | skis | 50 | broccoli | 70 | toaster |
| 11 | stop sign | 31 | snowboard | 51 | carrot | 71 | sink |
| 12 | parking meter | 32 | sports ball | 52 | hot dog | 72 | refrigerator |
| 13 | bench | 33 | kite | 53 | pizza | 73 | book |
| 14 | bird | 34 | baseball bat | 54 | donut | 74 | clock |
| 15 | cat | 35 | baseball glove | 55 | cake | 75 | vase |
| 16 | dog | 36 | skateboard | 56 | chair | 76 | scissors |
| 17 | horse | 37 | surfboard | 57 | couch | 77 | teddy bear |
| 18 | sheep | 38 | tennis racket | 58 | potted plant | 78 | hair drier |
| 19 | cow | 39 | bottle | 59 | bed | 79 | toothbrush |

#### カテゴリ別分類
- **人物**: person
- **乗り物**: bicycle, car, motorcycle, airplane, bus, train, truck, boat
- **動物**: bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe
- **アクセサリ**: backpack, umbrella, handbag, tie, suitcase
- **スポーツ**: frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket
- **食器**: bottle, wine glass, cup, fork, knife, spoon, bowl
- **食べ物**: banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake
- **家具**: chair, couch, potted plant, bed, dining table, toilet
- **電子機器**: tv, laptop, mouse, remote, keyboard, cell phone
- **家電**: microwave, oven, toaster, sink, refrigerator
- **その他**: traffic light, fire hydrant, stop sign, parking meter, bench, book, clock, vase, scissors, teddy bear, hair drier, toothbrush

## 5. 非機能要件（NFR）

### NFR-1: パフォーマンス
- Hailo-8L 使用時に 10 FPS 以上を目標（モデル/解像度に依存）
- 実測値: YOLOv8s、1280x720 で約 15-20 FPS

### NFR-2: 可搬性
- 依存関係は `requirements.txt` に明示
- システムパッケージ（HailoRT）は `--system-site-packages` で仮想環境からアクセス

### NFR-3: 可読性
- 100〜150 行単位で関数を分割
- 全関数/クラスに docstring を付与（Google style）
- 日本語コメントで教育目的の説明を追加

### NFR-4: 信頼性
- 例外処理（カメラ未接続、モデル未発見、デバイス初期化失敗）
- ユーザー向けエラーメッセージに対処方法を含める

## 6. 前提・制約
- Hailo モデル（.hef）は配布条件に留意
  - `hailo-all` パッケージでシステムにインストール済み（`/usr/share/hailo-models/`）
  - シンボリックリンクで `models/` ディレクトリから参照
- Camera V3 用ドライバ/ファームは Raspberry Pi OS 最新に更新済みであること
- ディスプレイ未接続の場合でも **ファイル保存モード**で検証可能

## 7. 環境構築（概要）

### 7.1 OS/ファーム更新
```bash
sudo apt update && sudo apt full-upgrade -y
sudo reboot
```

### 7.2 カメラ有効化
```bash
sudo raspi-config  # Interface → Camera 有効化（再起動）

# カメラ確認（OS バージョンにより異なる）
rpicam-hello --list-cameras    # Debian 13 (trixie)
libcamera-hello --list-cameras # Debian 12 (bookworm)
```

### 7.3 HailoRT SDK インストール
```bash
# hailo-all メタパッケージで一括インストール
sudo apt update
sudo apt install -y hailo-all
sudo reboot

# デバイス認識確認
hailortcli fw-control identify
```

### 7.4 Python 環境
```bash
# --system-site-packages でシステムの HailoRT にアクセス可能にする
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
pip install -r requirements.txt

# HailoRT 確認
python3 -c "from hailo_platform import HEF; print('HailoRT OK')"
```

### 7.5 モデル配置
```bash
mkdir -p models

# hailo-all パッケージのモデルへのシンボリックリンク作成
ln -sf /usr/share/hailo-models/yolov8s_h8l.hef models/yolov8s_h8l.hef
ln -sf /usr/share/hailo-models/yolov6n_h8l.hef models/yolov6n_h8l.hef
```

## 8. 実行手順（例）
```bash
source .venv/bin/activate

# Camera Module V3 版（基本）
python raspi_hailo8l_yolo.py --res 1280x720 --conf 0.25 --iou 0.45

# カメラを逆さまに設置している場合
python raspi_hailo8l_yolo.py --flip

# 動画保存する場合
python raspi_hailo8l_yolo.py --res 1280x720 --save

# 特定クラスのみ検出（人物のみ）
python raspi_hailo8l_yolo.py --classes person

# 複数クラスを検出（人物、車、犬）
python raspi_hailo8l_yolo.py --classes person car dog

# 使用可能なクラス一覧を表示
python raspi_hailo8l_yolo.py --list-classes

# USB Webcam 版
python raspi_hailo8l_yolo_usb.py --network yolov8s --width 1280 --height 720
```

## 9. 画面仕様
- 左上: FPS, 解像度, 推論時間(ms)
- 各検出物: カテゴリ名 + 確度（小数）、緑色矩形枠
- ラベル背景: 緑色塗りつぶし、白文字

## 10. ロギング/保存仕様
- `--save` 指定時: `output/detection_YYYYmmdd_HHMMSS.mp4`
- `--log` 指定時: `logs/detections_YYYYmmdd_HHMMSS.csv`

CSVフォーマット:
```
timestamp,frame_id,class_name,confidence,x1,y1,x2,y2
2024-12-29T14:30:52.123456,1,person,0.85,100,150,300,400
```

## 11. エラー/例外ハンドリング
- カメラ取得失敗 → メッセージ提示・終了（対処方法: `libcamera-hello` で確認）
- Hailo デバイス/モデル未初期化 → 提示・終了（対処方法: `hailortcli fw-control identify` で確認）
- HailoRT API エラー → スタックトレース出力、ユーザー向けヒント表示
- 低メモリ/高温時 → FPS 自動ダウンスケール（将来拡張）

## 12. テスト観点（受入条件）
- [ ] Camera V3 プレビューが出る（`rpicam-hello`）
- [ ] 既知物体（人/ボトル等）が 1280×720 で検出できる
- [ ] バウンディングボックスが正しい位置に表示される
- [ ] `--flip` オプションで映像が上下反転する
- [ ] `--save` で動画が保存される
- [ ] `--log` で CSV ログが保存される
- [ ] `--conf` 変更で検出件数が変化する
- [ ] `q` / `Q` / `ESC` キーで正常終了する
- [ ] `--classes` で指定したクラスのみ検出される
- [ ] `--list-classes` でクラス一覧が表示される
- [ ] クラスフィルタリング時もラベル（クラス名・信頼度）が表示される
- [ ] 10 分連続動作でクラッシュ/著しい熱スロットリングがない

## 13. 技術詳細

### HailoRT v4.x API 使用方法
```python
from hailo_platform import (HEF, VDevice, HailoStreamInterface,
                           InferVStreams, ConfigureParams,
                           InputVStreamParams, OutputVStreamParams,
                           FormatType)

# デバイス初期化
device = VDevice()
hef = HEF(model_path)

# ネットワーク設定
configure_params = ConfigureParams.create_from_hef(
    hef=hef, interface=HailoStreamInterface.PCIe)
network_groups = device.configure(hef, configure_params)
network_group = network_groups[0]

# VStreams パラメータ
input_vstreams_params = InputVStreamParams.make(
    network_group, format_type=FormatType.UINT8)
output_vstreams_params = OutputVStreamParams.make(
    network_group, format_type=FormatType.FLOAT32)

# 推論実行
with network_group.activate():
    with InferVStreams(network_group,
                      input_vstreams_params,
                      output_vstreams_params) as infer_pipeline:
        output_dict = infer_pipeline.infer({input_name: preprocessed})
```

### NMS 出力形式
HailoRT v4.x の NMS 後処理済み出力:
- 形式: `outputs[batch][class_id] = np.ndarray(N, 5)`
- 各検出: `[y1, x1, y2, x2, confidence]`
- 座標: 0-1 の正規化値（ピクセル座標への変換が必要）

## 14. ライセンス/著作権
- ソース: MIT License
- モデル/SDK: 各ベンダーのライセンスに従う（再配布不可なら配布しない）

---

## 推奨リポジトリ構成
```
11-002-raspi-hailo8l-yolo-detector/
├── raspi_hailo8l_yolo.py       # Camera Module V3 版
├── raspi_hailo8l_yolo_usb.py   # USB Webcam 版
├── requirements.txt
├── requirements_usb.txt        # USB 版追加依存
├── setup_usb_webcam.sh         # USB 版セットアップスクリプト
├── README.md
├── .gitignore
├── docs/                       # 技術ドキュメント
│   ├── README.md               # ドキュメント索引
│   ├── SETUP_GUIDE.md          # 環境構築ガイド
│   ├── TROUBLESHOOTING.md      # トラブルシューティング
│   ├── CLAUDE.md               # Claude Code 開発ルール
│   ├── python_coding_guidelines.md
│   ├── COMMENT_STYLE_GUIDE.md
│   └── 11_002_raspi_hailo_8_l_yolo_detector.md  # 本仕様書
├── models/                     # シンボリックリンク配置
│   ├── yolov8s_h8l.hef -> /usr/share/hailo-models/...
│   └── yolov6n_h8l.hef -> /usr/share/hailo-models/...
├── output/                     # 動画出力先（自動生成）
└── logs/                       # ログ出力先（自動生成）
```
