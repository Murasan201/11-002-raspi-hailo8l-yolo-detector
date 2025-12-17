# ドキュメント索引

このディレクトリには、プロジェクトに関連するすべてのドキュメントが含まれています。

## 📋 プロジェクト仕様

### [11_002_raspi_hailo_8_l_yolo_detector.md](./11_002_raspi_hailo_8_l_yolo_detector.md)
**要件定義書**
- プロジェクトの全体仕様と機能要件（日本語）
- システム構成、開発環境、ハードウェア仕様
- 実装ガイドラインとテスト要件
- **重要**: 開発作業を開始する前に必ず参照すること

## 🛠️ セットアップ手順

### [SETUP_GUIDE.md](./SETUP_GUIDE.md)
**環境構築ガイド**
- ハードウェア・ソフトウェア要件の詳細
- Raspberry Pi OS のセットアップ手順
- カメラモジュールとHailo-8L AI Kitの設定
- Python環境の構築とプロジェクトのインストール
- モデルファイルの準備と配置
- 動作確認とトラブルシューティング
- **重要**: プロジェクトを初めて実行する前に必ず参照すること

## 🔧 トラブルシューティング

### [TROUBLESHOOTING.md](./TROUBLESHOOTING.md)
**問題解決ナレッジベース**
- 環境構築時に発生した問題と解決策の記録
- カメラ関連の問題（libcamera-hello → rpicam-hello の変更等）
- Hailo-8L 関連の問題（hailortcli が見つからない等）
- Python 環境の問題（依存関係、仮想環境の設定）
- アプリケーション実行時の問題（GUI エラー、モデルファイル）
- パフォーマンスの問題（FPS 向上、熱対策）
- **重要**: 問題が発生したらまずこのドキュメントを確認すること

## 🤖 AI開発ガイドライン

### [CLAUDE.md](./CLAUDE.md)
**Claude Code開発ルール**
- Claude Codeを使用したこのプロジェクト専用の開発ワークフロー
- 日本語でのコミュニケーションルール
- プロジェクト固有の開発ガイドラインと制約
- テストコマンドとエラーハンドリング戦略

## 📝 コーディング規約

### [python_coding_guidelines.md](./python_coding_guidelines.md)
**Pythonコーディング標準**
- PEP 8準拠のコーディングスタイル
- 命名規則（snake_case, PascalCase）
- 型注釈（type hints）の使用方法
- エラーハンドリングのベストプラクティス
- 関数構造とモジュール設計の指針

### [COMMENT_STYLE_GUIDE.md](./COMMENT_STYLE_GUIDE.md)
**コメント記載標準**
- 教育的コード向けのコメントスタイル
- Docstring形式（Google style）
- クラス・関数・メソッドのドキュメント要件
- インラインコメントの書き方（「何」ではなく「なぜ」を説明）
- エラーメッセージの記載方法（問題説明 + 対処方法）

## 📂 ディレクトリ構造

```
docs/
├── README.md                                      # このファイル（索引）
├── SETUP_GUIDE.md                                 # 環境構築ガイド
├── TROUBLESHOOTING.md                             # トラブルシューティング
├── 11_002_raspi_hailo_8_l_yolo_detector.md       # 要件定義書
├── CLAUDE.md                                      # Claude Code開発ルール
├── python_coding_guidelines.md                   # Pythonコーディング規約
└── COMMENT_STYLE_GUIDE.md                        # コメントスタイルガイド
```

## 🔗 関連リソース

- **プロジェクトリポジトリ**: https://github.com/Murasan201/11-002-raspi-hailo8l-yolo-detector
- **メインREADME**: [../README.md](../README.md) - セットアップ手順と使用方法

## 📌 読む順番（推奨）

### 初めて使う場合（環境構築から実行まで）
1. **[SETUP_GUIDE.md](./SETUP_GUIDE.md)** - 環境構築手順（ハードウェアからソフトウェアまで）
2. **[../README.md](../README.md)** - 使用方法とオプション説明

### 開発を行う場合
1. **[SETUP_GUIDE.md](./SETUP_GUIDE.md)** - 環境構築（未実施の場合）
2. **[CLAUDE.md](./CLAUDE.md)** - プロジェクト全体のルールを理解
3. **[11_002_raspi_hailo_8_l_yolo_detector.md](./11_002_raspi_hailo_8_l_yolo_detector.md)** - 要件定義を確認
4. **[python_coding_guidelines.md](./python_coding_guidelines.md)** - コーディング規約を把握
5. **[COMMENT_STYLE_GUIDE.md](./COMMENT_STYLE_GUIDE.md)** - ドキュメント作成方法を習得

---

> **注意**: すべてのコード開発は、これらのドキュメントに定義された標準とガイドラインに従う必要があります。
