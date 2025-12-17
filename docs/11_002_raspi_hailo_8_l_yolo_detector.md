# 要件定義書
**Project:** 11-002-raspi-hailo8l-yolo-detector  
**目的:** Raspberry Pi 5 + 公式 AI Kit（Hailo-8L）+ Raspberry Pi Camera Module V3 を用いて、YOLO によるリアルタイム物体検出を行う単一ファイル構成の学習/実用向けアプリを開発する。

## 1. 背景・参照
- 参照記事（作者ブログ）: Raspberry Pi 5 + Hailo-8L + Webcam で YOLO 物体検出の実践内容  
- ベースとなる公式プロジェクト: [hailo-rpi5-examples (GitHub)](https://github.com/hailo-ai/hailo-rpi5-examples)  
- 本リポジトリは上記記事・公式リポジトリの手順/知見を再現・整理し、**初心者でもセットアップ〜動作確認まで迷わない**ことを狙う。

## 2. スコープ
- **対象ハード**: Raspberry Pi 5、Raspberry Pi AI Kit（Hailo-8L）、Raspberry Pi Camera Module V3（IMX708）  
- **対象ソフト**: Raspberry Pi OS (Bookworm 以降)、HailoRT SDK、Python 3.11+、OpenCV、（必要に応じて）YOLOv8 互換エンジン/モデル  
- **本アプリ**: 単一 Python ファイル（`raspi_hailo8l_yolo.py`）で完結。カメラ入力 → 前処理 → Hailo-8L で推論 → 後処理（NMS等）→ 画面表示/保存 を関数で分割。

## 3. 成果物
- 動作可能な単一ファイルアプリ `raspi_hailo8l_yolo.py`  
- `README.md`（セットアップ～実行手順）  
- `requirements.txt`（Python 依存）  
- （任意）`assets/` デモ画像・スクリーンショット

## 4. 機能要件（FR）
1. **カメラ入力**
   - Camera V3 から MIPI 経由で映像取得（`libcamera` パイプ or OpenCV）  
   - 解像度プリセット: 640×480 / 1280×720 / 1920×1080（引数で指定）  
   - フレームレート: 目標 20–30 FPS（実機性能に依存）

2. **推論（YOLO × Hailo-8L）**
   - Hailo-8L 用に最適化済みの YOLO（例: YOLOv8n 等）を **HEF/BOB/BLOb** 形式でロード  
   - 前処理: リサイズ、正規化、レイアウト変換  
   - 後処理: 信頼度閾値、NMS、クラス名マッピング（COCO想定）

3. **描画/出力**
   - 画面表示（バウンディングボックス、クラス名、スコア）  
   - オプション: 画像/動画保存（`--save`）  
   - オプション: ログ出力（CSV/JSON で推論結果書き出し）

4. **起動オプション**
   - `--res 1280x720` / `--conf 0.25` / `--iou 0.45` / `--save` / `--device 0` など  
   - `q` キーで安全終了

## 5. 非機能要件（NFR）
- **パフォーマンス**: Hailo-8L 使用時に 10 FPS 以上を目標（モデル/解像度に依存）  
- **可搬性**: 単一ファイル構成、依存関係は `requirements.txt` に明示  
- **可読性**: 100〜150 行単位で関数を分割、主要関数に docstring を付与  
- **信頼性**: 例外処理（カメラ未接続、モデル未発見、デバイス初期化失敗）

## 6. 前提・制約
- Hailo モデル（.hef / .blob 等）は配布条件に留意。  
  - 公開配布不可の場合は **取得手順と配置場所のみ記載**（例: `models/yolov8n_hailo.hef` をユーザー各自で取得配置）。  
- Camera V3 用ドライバ/ファームは Raspberry Pi OS 最新に更新済みであること。  
- ディスプレイ未接続の場合でも **ファイル保存モード**で検証可能。

## 7. 環境構築（概要）
1. **OS/ファーム更新**
   - `sudo apt update && sudo apt full-upgrade -y`  
2. **カメラ有効化**
   - `sudo raspi-config` → Interface → Camera 有効化（再起動）  
   - `libcamera-hello` でプレビュー確認  
3. **HailoRT / ドライバ**
   - 公式手順で HailoRT SDK とカーネルモジュール導入  
   - `hailortcli fw-control identify` で認識確認  
4. **Python ライブラリ**
   - `python3 -m venv .venv && source .venv/bin/activate`  
   - `pip install -r requirements.txt`  
5. **モデル配置**
   - `models/yolov8n_hailo.hef`（または相当物）を配置

## 8. 実行手順（例）
```bash
source .venv/bin/activate
python raspi_hailo8l_yolo.py --res 1280x720 --conf 0.25 --iou 0.45
# 保存する場合
python raspi_hailo8l_yolo.py --res 1280x720 --save
```

## 9. 画面仕様
- 左上: FPS, 解像度, 推論時間(ms)  
- 各検出物: カテゴリ名 + 確度（%）、矩形枠

## 10. ロギング/保存仕様
- `--save` 指定時: `output/` に `YYYYmmdd_HHMMSS.mp4`（または連番 PNG）  
- `--log` 指定時: `logs/` に `detections_*.csv` を保存（フレームID, class, conf, x1,y1,x2,y2）

## 11. エラー/例外ハンドリング
- カメラ取得失敗 → メッセージ提示・終了  
- Hailo デバイス/モデル未初期化 → 提示・終了  
- 低メモリ/高温時 → FPS 自動ダウンスケール（将来拡張）

## 12. テスト観点（受入条件）
- Camera V3 プレビューが出る  
- 既知物体（人/ボトル等）が 1280×720 で検出できる  
- `--save` で動画/画像が保存される  
- `--conf` 変更で検出件数が変化する  
- 10 分連続動作でクラッシュ/著しい熱スロットリングがない

## 13. ライセンス/著作権
- ソース: MIT（予定）  
- モデル/SDK: 各ベンダーのライセンスに従う（再配布不可なら配布しない）

---

## 推奨リポジトリ構成
```
11-002-raspi-hailo8l-yolo-detector/
├── raspi_hailo8l_yolo.py   # 単一ファイル本体
├── requirements.txt
├── README.md
├── models/                 # ※配布しない。取得・配置手順のみ記載
│   └── yolov8n_hailo.hef (not included)
├── output/                 # 出力先（自動生成）
└── logs/                   # ログ出力先（自動生成）
```

