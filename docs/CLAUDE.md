# Claude Code Rules for Raspberry Pi Hailo-8L YOLO Detector

## Project Information
- **Repository**: https://github.com/Murasan201/11-002-raspi-hailo8l-yolo-detector
- **Project ID**: 11-002-raspi-hailo8l-yolo-detector
- **Requirements Document**: `11_002_raspi_hailo_8_l_yolo_detector.md` (Japanese)

## Development Workflow
**IMPORTANT**: Always start by reading the requirements document `11_002_raspi_hailo_8_l_yolo_detector.md` before beginning any development work. This document contains the complete project specifications, functional requirements, and implementation guidelines.

## Project Management & Development Delegation

### PM（プロジェクトマネージャー）の役割
PMは以下の役割を担い、**直接コーディング作業を行わない**：
- プロジェクト全体の進捗管理
- 要件の明確化と優先順位付け
- タスクの分割と割り当て
- 成果物のレビューと品質管理

### サブエージェントへの作業依頼ルール
**CRITICAL**: PMがコーディング作業を行う際は、必ずサブエージェント（Task tool）を起動して作業を委譲すること。

#### 依頼時の必須事項
1. **参照仕様書の明示的な指定**
   - 必ず参照すべきドキュメントのパスを指定する
   - サブエージェントは指定された仕様書を読み込んでから作業を開始する

2. **指定すべき仕様書の種類**
   | 作業内容 | 必須参照ドキュメント |
   |---------|-------------------|
   | 新機能実装 | `docs/11_002_raspi_hailo_8_l_yolo_detector.md`（要件定義書） |
   | コード修正/リファクタリング | `docs/python_coding_guidelines.md`（コーディング規約） |
   | ドキュメント/コメント追加 | `docs/COMMENT_STYLE_GUIDE.md`（コメント標準） |
   | 全般的な開発作業 | `docs/CLAUDE.md`（本ドキュメント） |

3. **依頼プロンプトのテンプレート**
   ```
   以下のタスクを実行してください。

   【タスク】
   <具体的な作業内容>

   【参照仕様書】
   - docs/<仕様書名>.md

   【制約事項】
   - 仕様書のルールに従うこと
   - 日本語でコメントを記載すること
   ```

#### サブエージェント起動例
```
Task tool を使用:
- subagent_type: "general-purpose" または "Plan"
- model: "sonnet"  ← コーディング作業では必ず sonnet を指定
- prompt: 上記テンプレートに従った依頼内容
```

#### モデル指定ルール
**IMPORTANT**: コーディング作業のためにサブエージェントを起動する際は、必ず `model: "sonnet"` を指定すること。
- **sonnet**: コーディング作業（実装、修正、リファクタリング）
- **haiku**: 軽微な調査、簡単な質問応答（コーディング以外）
- **opus**: 複雑な設計判断、アーキテクチャ検討（PMが直接使用）

### 作業フロー
```
1. PM: 要件を分析しタスクを定義
      ↓
2. PM: 参照仕様書を特定
      ↓
3. PM: Task tool でサブエージェントを起動
      ↓
4. サブエージェント: 仕様書を読み込み
      ↓
5. サブエージェント: コーディング作業を実施
      ↓
6. PM: 成果物をレビュー
```

## Communication Rules
- **Language**: Always respond in Japanese (日本語で返答する)
- **Documentation**: Code comments and documentation should be in Japanese when appropriate

## Project Overview
This project implements a real-time YOLO object detection system using:
- Raspberry Pi 5 + Hailo-8L AI Kit + Camera Module V3
- Single-file Python application (`raspi_hailo8l_yolo.py`)
- Real-time camera processing with AI acceleration

## Development Guidelines

### Code Style and Comments
All code development MUST strictly follow the guidelines defined in:
- **`COMMENT_STYLE_GUIDE.md`**: Defines comment style and docstring format for educational materials
- **`python_coding_guidelines.md`**: Defines Python coding standards and best practices

#### Key Coding Standards (from python_coding_guidelines.md)
- **PEP 8 compliance**: 4-space indents, snake_case for functions/variables, PascalCase for classes
- **Docstrings**: All functions and classes MUST have docstrings with Args, Returns sections
- **Type hints**: Use type annotations in function signatures
- **Main guard**: Always use `if __name__ == "__main__":` for executable scripts
- **Error handling**: Catch specific exceptions, provide user-friendly error messages with remediation hints
- **Function structure**: Functions should be 100-150 lines, well-organized with clear sections

#### Key Comment Standards (from COMMENT_STYLE_GUIDE.md)
- **File header**: Include project name, brief description, and reference to requirements document
- **Class docstring**: Multi-line format explaining role and design intent
- **Method docstring**: Clear description with Args, Returns sections including type information
- **Inline comments**: Explain *why*, not *what*; include educational context for beginners
- **Error messages**: Provide both error description and remediation hints (e.g., "check configuration")
- **Avoid**: Self-evident comments like "# increment x"

### Code Structure
- **Multi-file design**: Separate applications for Camera Module V3 and USB Webcam
  - `raspi_hailo8l_yolo.py` - Camera Module V3 optimized
  - `raspi_hailo8l_yolo_usb.py` - USB Webcam with GStreamer
- **Class-based**: Well-organized classes (YOLODetector, CameraManager, DetectionLogger)
- **Function-based**: Break into 100-150 line functions with comprehensive docstrings
- **Clean separation**: camera input → preprocessing → inference → postprocessing → display

### Dependencies
- Use `requirements.txt` for Python dependencies
- Target Python 3.11+ on Raspberry Pi OS Bookworm
- Key libraries: OpenCV, HailoRT SDK, numpy

### Testing Commands
```bash
# Lint and format (if available)
python -m flake8 raspi_hailo8l_yolo.py
python -m black raspi_hailo8l_yolo.py

# Test camera functionality
libcamera-hello

# Test Hailo device
hailortcli fw-control identify

# Run application
python raspi_hailo8l_yolo.py --res 1280x720 --conf 0.25
```

### Error Handling
- Handle camera initialization failures gracefully
- Check Hailo device availability before inference
- Provide clear error messages for missing models/dependencies

### Performance Targets
- Target 10+ FPS with Hailo-8L acceleration
- Support multiple resolutions: 640x480, 1280x720, 1920x1080
- Efficient memory usage for embedded environment

### File Organization
```
├── raspi_hailo8l_yolo.py   # Main application
├── requirements.txt        # Python dependencies
├── README.md              # Setup and usage instructions
├── models/                # Model files (user-provided)
├── output/                # Saved videos/images
└── logs/                  # Detection logs
```

### Security Notes
- No sensitive data or API keys in code
- Models may have licensing restrictions (don't redistribute)
- Follow defensive security practices for embedded systems

## Troubleshooting Documentation Rules

### エラー記録の義務
**CRITICAL**: 環境構築や開発作業中に発生したすべてのエラーとその解決策は、必ず `docs/TROUBLESHOOTING.md` に記録すること。

### 記録すべき情報
1. **発生日**: エラーが発生した日付（YYYY-MM-DD形式）
2. **エラーメッセージ**: 実際に表示されたエラーメッセージ（コードブロックで記載）
3. **原因**: エラーの原因を箇条書きで記載
4. **解決策**: 解決手順をコマンド付きで記載
5. **確認コマンド**: 解決後の確認方法
6. **ステータス**: 解決済み / 未解決

### 記録のタイミング
- エラー発生直後に記録を開始
- 解決策が判明した時点で更新
- 同種のエラーが再発した場合は追記

### TROUBLESHOOTING.md の構成
```markdown
### 問題 X: [問題のタイトル]

**発生日**: YYYY-MM-DD

**エラーメッセージ**:
```
[エラーメッセージ]
```

**原因**:
- [原因1]
- [原因2]

**解決策**:
```bash
[解決コマンド]
```

**確認コマンド**:
```bash
[確認コマンド]
```

**ステータス**: [x] 解決済み / [ ] 未解決
```

### 記録の目的
- 将来同じエラーに遭遇した際の参照資料
- 書籍・技術記事執筆時の情報源
- 他の開発者への知識共有

---

## Code Quality Standards

### Coding Standards and Guidelines
このプロジェクトのすべてのコード開発は以下のドキュメントに従うこと：
- **COMMENT_STYLE_GUIDE.md**: コメント記載とdocstring形式の標準
- **python_coding_guidelines.md**: Python コーディング規約と命名規則

#### 遵守すべき主要ルール
- **PEP 8準拠**: 4スペースインデント、snake_case（変数・関数）、PascalCase（クラス）
- **型注釈**: すべての関数署名に type hints を記載（Args、Returns に対応）
- **Docstring形式**: Google style フォーマット（説明 + Args + Returns）
- **エラーハンドリング**: 特定の例外をキャッチ、ユーザー向けのヒント付きメッセージ
- **if __name__ == "__main__"**: 実行可能なスクリプトに必須
- **インラインコメント**: *何*ではなく*なぜ*を説明、初心者向けの補足情報を含む

### Documentation Requirements
1. **Every function/method** must have a docstring following COMMENT_STYLE_GUIDE.md
2. **Every class** must have a multi-line docstring explaining its purpose and design intent
3. **Error messages** must include both the error description and remediation instructions
4. **Inline comments** should explain design decisions and provide educational context

### Error Handling Strategy
- Validate model files exist before initialization
- Provide clear user instructions when Hailo device is not available
- Include specific check commands (e.g., `hailortcli fw-control identify`)
- Use FileNotFoundError and specific exception types instead of generic Exception
- Always include actionable remediation in error messages

### Review Checklist
Before committing code, verify:
- [ ] File header includes project name and requirements document reference
- [ ] All classes have multi-line docstrings with design intent
- [ ] All functions/methods have docstrings with Args, Returns, and type information
- [ ] Error messages include remediation hints and check commands
- [ ] Inline comments explain *why*, not *what* (avoid "# increment x" type comments)
- [ ] PEP 8 compliance (4-space indents, naming conventions, line length)
- [ ] Type hints used throughout
- [ ] if __name__ == "__main__": used for executable scripts
- [ ] Specific exceptions caught, not bare Exception
- [ ] Test commands work: `libcamera-hello`, `hailortcli fw-control identify`