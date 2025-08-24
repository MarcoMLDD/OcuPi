# OcuPi: MediaPipe Drowsiness Detection

An iteration of the previous OpenCLOS, OcuPi aims to solve the issues faced with OpenCLOS. OcuPi is a real-time drowsiness detection system using MediaPipe Face Mesh to monitor eye and mouth movements for fatigue detection. This is still buggy (that's for sure), certain things don't work out, but that's why it's open-source! So everyone could contribute, so feel free! The thing with this is that I coded a lot of the basic structure (most were buggy) and needed a bit of help from AI, so it's my effort + vibe coding lol. Anyway, thank you!

## Features

- **Real-time Monitoring**: Uses webcam feed for continuous monitoring
- **MediaPipe Integration**: Accurate facial landmark detection
- **PERCLOS Calculation & Scoring**: Industry-standard eye closure measurement. Instead of regular EAR and MAR parameters, a scoring system was implemented
- **Yawn Detection**: Mouth aspect ratio analysis
- **Adaptive Thresholds**: Automatic calibration for individual users
- **Cross-Platform**: Works on both desktop and Orange Pi (ARM)
- **Dual Interface**: GUI mode with live feed or CLI mode for headless operation
- **Alert System**: Visual and audible alerts for drowsiness detection
- **BlackBox Feature**: Log of information from a session

## Requirements

Based on my analysis of the OcuPi application code, here are **all the dependencies needed to run it**:

## **Complete Dependencies List**

### **Built-in Python Modules** (No installation required)
These come with Python by default:
- `os` - Operating system interface
- `sys` - System-specific parameters  
- `time` - Time-related functions
- `threading` - Thread-based parallelism
- `queue` - Queue data structures
- `signal` - Signal handling
- `platform` - Platform identification
- `argparse` - Command-line argument parsing
- `json` - JSON encoder/decoder
- `math` - Mathematical functions
- `logging` - Logging facility
- `collections` (deque) - Specialized container datatypes
- `datetime` - Date and time handling
- `socket` - Network interface
- `subprocess` - Subprocess management

### **Third-Party Packages** (Require pip installation)

#### **Core Dependencies** (Required):
```bash
pip install psutil>=5.9.0           # System and process utilities
pip install opencv-python>=4.8.0    # Computer vision library  
pip install numpy>=1.26.0           # Numerical computing
pip install mediapipe>=0.10.0       # Face detection/landmarks
```

#### **GUI Dependencies** (Optional - for GUI mode):
```bash
pip install Pillow>=10.0.0          # Image processing (PIL)
# tkinter - Usually comes with Python, but on some systems:
# Ubuntu/Debian: sudo apt-get install python3-tk
# CentOS/RHEL: sudo yum install tkinter
```

#### **Audio Dependencies** (Optional - for sound alerts):
```bash
pip install pygame>=2.5.0           # Sound/audio support
```

### **Installation Commands**

#### **Minimal Installation** (CLI mode only):
```bash
pip install psutil>=5.9.0 opencv-python>=4.8.0 numpy>=1.26.0 mediapipe>=0.10.0
```

#### **Full Installation** (GUI + Audio support):
```bash
pip install psutil>=5.9.0 opencv-python>=4.8.0 numpy>=1.26.0 mediapipe>=0.10.0 Pillow>=10.0.0 pygame>=2.5.0
```

#### **From requirements.txt** (if using the full project):
```bash
pip install -r /app/backend/requirements.txt
```

### **Dependency Notes**

#### **Optional Dependencies**:
- **`tkinter`**: Only needed for GUI mode (`--gui`). CLI mode works without it.
- **`Pillow (PIL)`**: Only needed for GUI mode to display video frames.
- **`pygame`**: Only needed for custom audio alerts. System beep fallback available.

#### **System Dependencies**:
- **Camera access**: For actual drowsiness detection (optional for testing)
- **Audio system**: For sound alerts (optional - falls back to system beep)

#### **Platform-Specific Notes**:
- **Linux**: May need `sudo apt-get install python3-tk` for tkinter
- **Windows**: tkinter usually included with Python
- **macOS**: tkinter usually included with Python

### **Verification**
To verify all dependencies are installed correctly:
```bash
python -c "
import psutil, cv2, numpy, mediapipe
print('✅ Core dependencies installed successfully')
try:
    import tkinter, PIL
    print('✅ GUI dependencies available')
except ImportError:
    print('⚠️  GUI dependencies missing (CLI mode only)')
try:
    import pygame
    print('✅ Audio dependencies available')
except ImportError:
    print('⚠️  Audio dependencies missing (system beep fallback)')
"
```

The application is designed to **gracefully handle missing optional dependencies** and will continue to work in CLI mode even if GUI or audio components are unavailable.

## Acknowledgments

I don't really know who to acknowledge, but I'm definitely grateful to the people who developed the needed dependencies to run this!
And I would like to acknowledge my project group mates for their contributions to the documents and the hardware:
1. J. Rodriguez
2. R. Punzalan
