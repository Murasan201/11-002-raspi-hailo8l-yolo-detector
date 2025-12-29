# プロジェクトジャーナル

## 2025-12-29: コードMVP化の方針決定

### 背景
`raspi_hailo8l_yolo.py` はプログラミング技術書に掲載するサンプルコード用に作成した。
しかし、開発を進める中でコードの規模が肥大化してしまった。

### 現状の問題
- 機能要求は達成している
- しかしコード量が多すぎて書籍への掲載ができない状態
- 初心者にとって理解しづらい複雑な構成になっている

### 対応方針
規模縮小かつ初心者でもわかりやすいよう、**最小構成でMVP化（最小化）** を実施する。

### 制約事項
別プロジェクト `12-002-pet-monitoring-yolov8` が本コードをライブラリとして使用している。
以下のAPIは後方互換性を維持する必要がある：

| API | 種別 |
|-----|------|
| `YOLODetector` | クラス |
| `CameraManager` | クラス |
| `draw_detections` | 関数 |
| `COCO_CLASSES` | 定数 |

### 事前準備
1. `raspi_hailo8l_yolo_full.py` として動作確認済みコードをバックアップ
2. 外部参照情報を調査し、維持すべきAPIを特定
3. `docs/raspi_hailo8l_yolo_spec.md` に外部参照情報を記録

### 次のアクション
- ~~現在のコード構成を分析~~
- ~~削減可能な機能を特定~~
- ~~MVP版の実装~~

---

## 2025-12-29: MVP化実施完了

### 実施内容
`raspi_hailo8l_yolo.py` のMVP化を完了した。

### 削減結果

| 項目 | Full版 | MVP版 | 削減率 |
|-----|--------|-------|--------|
| 総行数 | 1230行 | 402行 | **67%削減** |
| クラス数 | 3 | 2 | 33%削減 |
| 関数数 | 10 | 2 | 80%削減 |
| CLI引数 | 10 | 6 | 40%削減 |

### 削減した機能

**クラス:**
- `DetectionLogger` - CSVログ機能

**YOLODetectorメソッド:**
- `is_initialized` プロパティ
- `target_classes` プロパティ
- `set_target_classes()` - 動的クラス変更
- `get_available_classes()` - クラス一覧取得
- `is_class_targeted()` - クラス判定
- `_dummy_detect()` - テスト用ダミー検出
- `preprocess_image()` - 公開メソッド化を解除（detectに統合）
- `postprocess_detections()` - 公開メソッド化を解除（_postprocessに変更）
- `_apply_nms()` - NMS機能（Hailo内部で実行済み）

**CameraManagerメソッド:**
- `_initialize_webcam()` - USB Webカメラ対応
- `__enter__` / `__exit__` - コンテキストマネージャー
- `use_picamera` 引数 - カメラ選択

**関数:**
- `setup_logging()` / `get_logger()` - ロギング機能
- `is_hailo_available()` / `is_picamera_available()` - 可用性チェック
- `draw_info()` - パフォーマンス情報描画
- `parse_resolution()` - 解像度パース（main内でインライン化）

**CLI引数:**
- `--save` - 動画保存
- `--log` - CSVログ保存
- `--iou` - IoU閾値
- `--list-classes` - クラス一覧表示

**その他:**
- 遅延インポート機構 → 即時インポートに簡素化
- ロギング機能 → print文で代替
- USB Webカメラ対応 → Camera Module V3専用に

### 維持した機能（外部参照互換性）

| API | シグネチャ |
|-----|-----------|
| `YOLODetector.__init__` | `(model_path, conf_threshold=0.25, target_classes=None)` |
| `YOLODetector.detect` | `(image) -> List[Dict]` |
| `CameraManager.__init__` | `(resolution=(1280,720), device_id=0, flip_vertical=False)` |
| `CameraManager.read_frame` | `() -> Optional[np.ndarray]` |
| `CameraManager.release` | `() -> None` |
| `draw_detections` | `(image, detections, color, thickness) -> np.ndarray` |
| `COCO_CLASSES` | `List[str]` (80クラス) |

### 動作確認結果
- 構文チェック: OK
- インポートテスト: OK
- APIシグネチャ互換性: OK

### バックアップ
Full版は `raspi_hailo8l_yolo_full.py` として保存済み（変更禁止）

---

## 2025-12-29: ドキュメント更新

### 更新したドキュメント

#### README.md
- 「ファイル構成」セクションを追加
  - MVP版とFull版の位置づけを明記
  - ライブラリとしての使用例を追加
- 「オプション一覧」をMVP版とFull版に分離
- 「使用例」をMVP版とFull版に分離
- 「コード構造」をMVP版とFull版に分離

#### docs/raspi_hailo8l_yolo_spec.md
- 「機能仕様（MVP版）」セクションを追加
  - 実施結果（削減率）
  - クラス構成（YOLODetector, CameraManager）
  - 関数構成（draw_detections, main）
  - コマンドライン引数
  - 外部参照互換性の確認結果

#### docs/JOURNAL.md
- MVP化実施完了の記録
- ドキュメント更新の記録（本セクション）

### プロジェクト状態
- MVP版: 完成・動作確認済み
- Full版: バックアップとして保存（変更禁止）
- ドキュメント: 更新完了
