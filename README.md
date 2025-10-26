# OcuPi: Driver Drowsiness Detection System

**OcuPi** is a computer vision-based driver monitoring system that detects drowsiness and loss of facial visibility using real-time image processing and mathematical algorithms.  
It uses **MediaPipe FaceMesh** to compute **Eye Aspect Ratio (EAR)** and **Mouth Aspect Ratio (MAR)** and trigger alerts when the driver appears drowsy or inattentive.  
This deployment-ready version supports both **GUI** and **Headless** operation modes. 
This project is part of a research project, with the help of J. Rodriguez and R. Punzalan, parts of the OcuPi team. This repository will **NOT** be updated nor modified in any significant way after the first release.

---

## Features

- **Real-Time Drowsiness Detection**  
  Uses EAR and MAR metrics derived from facial landmark tracking to detect eye closure and yawning events.

- **Headless and GUI Modes**  
  - **Headless Mode**: Runs autonomously without a display (ideal for embedded setups).  
  - **GUI Mode**: Includes a diagnostic interface built with Tkinter and PIL for debugging and live monitoring.

- **Automatic Calibration**  
  Automatically establishes EAR/MAR baselines at startup for user-specific detection accuracy.

- **Audio and System Alerts**  
  Plays alert sounds (`alarm.wav`, `NoFace.wav`, `CalibStart.wav`, `CalibEnd.wav`) using Pygame.  
  Falls back to system beep if sound files are missing.

- **Enhanced Logging System**  
  Generates structured logs with EAR, MAR, score, and event metadata in `OcuPi_BlackBox/`.

- **Multi-Camera Support**  
  Automatically detects and connects to the best available camera with detailed backend scanning.

- **Cross-Platform Compatibility**  
  Works on Windows, Linux, and macOS (tested with OpenCV + MediaPipe).

---

## System Requirements

| Component | Requirement |
|------------|-------------|
| Python | 3.8 or higher |
| Camera | USB or built-in webcam |
| RAM | ≥ 2 GB recommended |
| OS | Linux, Windows, macOS |
| Libraries | See below |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/MarcoMLDD/OcuPi-Deployment.git
cd OcuPi-Deployment

# Install dependencies
pip install -r requirements.txt
```

If you don’t have a `requirements.txt`, install manually:
```bash
pip install opencv-python mediapipe psutil numpy pygame pillow
```

---

## How It Works

1. **Facial Landmark Detection**  
   OcuPi uses MediaPipe FaceMesh to detect 468 facial landmarks in real time.

2. **EAR and MAR Calculation**  
   - **EAR (Eye Aspect Ratio):** Measures eye openness based on six key eye landmarks.  
   - **MAR (Mouth Aspect Ratio):** Measures mouth opening to detect yawns.

3. **Calibration**  
   The system calibrates EAR and MAR thresholds during the first few seconds of startup.

4. **Drowsiness Scoring**  
   A continuous score (0–100%) is computed using exponential moving averages (EMA) of EAR/MAR and PERCLOS (percentage of eye closure).

5. **Alerts and Logging**  
   - Triggers `DROWSINESS` or `NO_FACE` alerts.  
   - Logs detailed events and frame statistics for review.

---

## Usage

### **Headless Mode (default)**
Runs without GUI — suitable for embedded devices (e.g., Orange Pi, Raspberry Pi).

```bash
python3 OCUPI.py
```

### **GUI Mode**
Runs with a diagnostic dashboard for visualization and manual control.

```bash
python3 OCUPI.py --gui
```

### **Controls (Headless Mode)**
| Key | Function |
|-----|-----------|
| `q` | Quit |
| `s` | Silence alerts for 5 seconds |
| `r` | Recalibrate |
| `v` | Toggle video display |
| `1–9` | Switch camera |

---

## Logs and Output

All detection sessions are logged in:
```
OcuPi_BlackBox/ocupi_session_YYYYMMDD_HHMMSS.log
```

Log entries include:
- EAR, MAR, and Drowsiness Score
- Face visibility status
- Event triggers (calibration, alerts, errors)
- System info (CPU, memory usage)

---

## Audio Files

Place the following `.wav` files in the same directory as `OCUPI.py`:

| Alert Type | File Name |
|-------------|------------|
| Drowsiness | `alarm.wav` |
| No Face Detected | `NoFace.wav` |
| Calibration Start | `CalibStart.wav` |
| Calibration End | `CalibEnd.wav` |

If missing, the system will use a **system beep** fallback.

---

## Project Structure

```
OcuPi/
│
├── OCUPI.py                # Main detection program
├── alarm.wav               # Drowsiness alert sound
├── NoFace.wav              # No-face alert sound
├── CalibStart.wav          # Calibration start sound
├── CalibEnd.wav            # Calibration end sound
└── OcuPi_BlackBox/         # Auto-generated log folder
```

---

## Developer Notes

- Code is modular and threaded for responsive GUI updates.
- System monitoring (CPU, RAM) is integrated via `psutil`.
- Uses `deque` buffers for rolling EAR/MAR analysis.
- Logging granularity can be adjusted in `log_detection_data()`.
- Tested on **OpenCV 4.10+** and **MediaPipe 0.10+**.

---

## Command-Line Arguments

| Argument | Description | Default |
|-----------|--------------|----------|
| `--gui` | Launches diagnostic GUI mode | Disabled |
| *(none)* | Runs in headless mode | Enabled |

Example:
```bash
python3 OCUPI.py --gui
```

---

## License

This project is released under the **MIT License**.  
See the `LICENSE` file for more details.

---

## Credits

Developed by **Marc Lawrence D. Dizon**  
With support from J. Rodriguez & R. Punzalan, research project teammates.
Built with Python, OpenCV, MediaPipe, and Tkinter.  
Dedicated to improving road safety through computer-based driver monitoring.
