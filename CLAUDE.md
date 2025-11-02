# Claude Code Rules for Raspberry Pi Hailo-8L YOLO Detector

## Project Information
- **Repository**: https://github.com/Murasan201/11-002-raspi-hailo8l-yolo-detector
- **Project ID**: 11-002-raspi-hailo8l-yolo-detector
- **Requirements Document**: `11_002_raspi_hailo_8_l_yolo_detector.md` (Japanese)

## Development Workflow
**IMPORTANT**: Always start by reading the requirements document `11_002_raspi_hailo_8_l_yolo_detector.md` before beginning any development work. This document contains the complete project specifications, functional requirements, and implementation guidelines.

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

## Code Quality Standards

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