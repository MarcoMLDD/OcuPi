# OcuPi: MediaPipe Drowsiness Detection

An iteration of the previous OpenCLOS, OcuPi aims to solve the issues faced with OpenCLOS. OcuPi is a real-time drowsiness detection system using MediaPipe Face Mesh to monitor eye and mouth movements for fatigue detection.

## Features

- **Real-time Monitoring**: Uses webcam feed for continuous monitoring
- **MediaPipe Integration**: Accurate facial landmark detection
- **PERCLOS Calculation**: Industry-standard eye closure measurement
- **Yawn Detection**: Mouth aspect ratio analysis
- **Adaptive Thresholds**: Automatic calibration for individual users
- **Cross-Platform**: Works on both desktop and Raspberry Pi (ARM)
- **Dual Interface**: GUI mode with live feed or CLI mode for headless operation
- **Alert System**: Visual and audible alerts for drowsiness detection

## Requirements

```bash
pip install opencv-python mediapipe psutil pillow
