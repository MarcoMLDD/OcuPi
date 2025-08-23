import os
import sys
import psutil
import cv2
import time
import numpy as np
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import queue
import signal
import platform
import argparse
from PIL import Image, ImageTk
import json
import math
import mediapipe as mp
from collections import deque

class OcuPiDetector:
    def __init__(self, cli_mode=False, force_arm=False):
        self.cli_mode = cli_mode
        self.force_arm = force_arm
        self.video_window_name = "OcuPi: MediaPipe"
        self.running = True
        self._stop_event = threading.Event()
        self._gui_running = True
        
        self.HAS_PYGAME = False
        try:
            import pygame
            pygame.mixer.init()
            self.HAS_PYGAME = True
        except ImportError:
            pass

        self.process = psutil.Process(os.getpid())
        self.last_sys_update = 0
        self.sys_update_interval = 1.0
        self.cpu_usage = 0
        self.memory_usage = 0
        self.system_cpu = 0
        self.system_mem = 0
        
        self.gui_ready = False
        self.camera_ready = False
        self.detection_running = False
        self.models_ready = False
        self.manual_silence = False
        self.sound_playing = False
        self.custom_sound_path = None
        
        self.CALIB_SECS = 4.0
        self.EMA_ALPHA = 0.35
        self.perclos_window = 10.0
        self.yawn_window = 10.0
        self.enter_thresh = 0.60
        self.exit_thresh = 0.45
        
        self.perclos_buf = deque()
        self.yawn_buf = deque()
        self.time_buf = deque()
        
        self.ear_baseline = None
        self.mar_baseline = None
        self.ear_s = None
        self.mar_s = None
        self.score_s = None
        self.drowsy_flag = False
        self.calibration_start = 0
        
        self.eye_closed_frames = 0
        self.no_face_frames = 0
        self.alarm_interval = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        self.calibration_data = {}
        self.calibration_file = "ocupi_calibration.json"
        
        self.status_queue = queue.Queue()
        self.command_queue = queue.Queue()
        self.video_queue = queue.Queue(maxsize=1)
        
        self.available_cameras = []
        self.selected_camera = 0
        self.cap = None
        self.video_thread = None
        self._video_thread_lock = threading.Lock()
        self.last_frame = None
        
        self.alert_sound = None
        self.volume = 0.5
        
        self.init_models()
        
        if not self.cli_mode:
            self.setup_gui()
        else:
            self.setup_cli()
        
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def setup_cli(self):
        print("\nInitializing OcuPi CLI Mode:")
        print("-" * 40)
        
        if not self.init_camera(self.selected_camera):
            print("Error: Failed to initialize camera")
            sys.exit(1)
        
        if not self.models_ready:
            print("Error: Required models not loaded")
            sys.exit(1)
        
        print("\nCLI Mode Ready:")
        print("- Press 'q' to quit")
        print("- Press 's' to silence alerts for 5 seconds")
        print("- Video feed will display in separate window")
        print("-" * 40 + "\n")

    def _window_exists(self, window_name):
        try:
            return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1
        except:
            return False

    def signal_handler(self, signum, frame):
        print(f"Received signal {signum}, shutting down...")
        self.running = False
        self._stop_event.set()
        self.cleanup()
        sys.exit(0)

    def init_models(self):
        try:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.mp_draw = mp.solutions.drawing_utils
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1, 
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.models_ready = True
            print("MediaPipe FaceMesh loaded successfully")
            
            self.load_calibration()
            
        except Exception as e:
            print(f"Error loading models: {e}")
            self.models_ready = False

    def euclid(self, a, b):
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return math.hypot(dx, dy)

    def ear_from_points(self, eye):
        A = self.euclid(eye[1], eye[5])
        B = self.euclid(eye[2], eye[4])
        C = self.euclid(eye[0], eye[3])
        return (A + B) / (2.0 * C)

    def mar_from_coords(self, coords):
        v1 = self.euclid(coords[13], coords[14])
        v2 = self.euclid(coords[17], coords[0])
        h = self.euclid(coords[78], coords[308])
        return (v1 + v2) / (2.0 * h)

    def ema(self, prev, x, alpha):
        return x if prev is None else (alpha * x + (1 - alpha) * prev)

    def load_calibration(self):
        if os.path.exists(self.calibration_file):
            try:
                with open(self.calibration_file, 'r') as f:
                    self.calibration_data = json.load(f)
                self.ear_baseline = self.calibration_data.get('ear_baseline')
                self.mar_baseline = self.calibration_data.get('mar_baseline')
                print(f"Loaded calibration: EAR={self.ear_baseline}, MAR={self.mar_baseline}")
            except:
                print("Failed to load calibration data")
                self.calibration_data = {}

    def save_calibration(self):
        try:
            with open(self.calibration_file, 'w') as f:
                json.dump(self.calibration_data, f)
            print("Calibration data saved")
        except:
            print("Failed to save calibration data")

    def setup_gui(self):
        try:
            self.root = tk.Tk()
            self.root.title("OcuPi: MediaPipe")
            
            if self.force_arm or platform.machine() in ('arm', 'armv7l', 'aarch64'):
                self.root.geometry("800x480")
                self.video_width, self.video_height = 320, 240
            else:
                self.root.geometry("1024x600")
                self.video_width, self.video_height = 480, 360
            
            self.root.configure(bg='#2c3e50')
            
            main_frame = ttk.Frame(self.root, padding="10")
            main_frame.pack(fill=tk.BOTH, expand=True)
            
            left_frame = ttk.Frame(main_frame, width=400)
            left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            
            right_frame = ttk.Frame(main_frame)
            right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=5)
            
            title_label = ttk.Label(left_frame, text="OcuPi: MediaPipe", 
                                  font=('Arial', 14, 'bold'))
            title_label.pack(pady=10)
            
            info_frame = ttk.LabelFrame(left_frame, text="System Information", padding="10")
            info_frame.pack(fill=tk.X, pady=5)
            
            self.sys_info_label = ttk.Label(info_frame, text="Loading system info...", font=('Arial', 9))
            self.sys_info_label.pack()
            
            self.cpu_label = ttk.Label(info_frame, text="CPU: --% (System: --%)", font=('Arial', 9))
            self.cpu_label.pack()
            
            self.mem_label = ttk.Label(info_frame, text="Memory: -- MB (System: --%)", font=('Arial', 9))
            self.mem_label.pack()
            
            ttk.Separator(info_frame).pack(fill=tk.X, pady=2)
            
            sys_info = self.get_system_info()
            info_text = f"Platform: {sys_info['platform']} | OpenCV: {sys_info['opencv_version']}"
            ttk.Label(info_frame, text=info_text, font=('Arial', 9)).pack()
            
            camera_frame = ttk.LabelFrame(left_frame, text="Camera Selection", padding="10")
            camera_frame.pack(fill=tk.X, pady=5)
            
            camera_control_frame = ttk.Frame(camera_frame)
            camera_control_frame.pack(fill=tk.X)
            
            ttk.Label(camera_control_frame, text="Camera:").pack(side=tk.LEFT)
            
            self.camera_var = tk.StringVar()
            self.camera_combo = ttk.Combobox(camera_control_frame, textvariable=self.camera_var, 
                                           width=15, state='readonly')
            self.camera_combo.pack(side=tk.LEFT, padx=5)
            
            self.scan_button = ttk.Button(camera_control_frame, text="🔍 Scan", 
                                         command=self.scan_cameras_gui)
            self.scan_button.pack(side=tk.LEFT, padx=5)
            
            self.connect_button = ttk.Button(camera_control_frame, text="📹 Connect", 
                                           command=self.connect_camera_gui)
            self.connect_button.pack(side=tk.LEFT, padx=5)
            
            status_frame = ttk.LabelFrame(left_frame, text="Status", padding="10")
            status_frame.pack(fill=tk.X, pady=10)
            
            self.status_label = ttk.Label(status_frame, text="Ready to start...", 
                                         font=('Arial', 10))
            self.status_label.pack()
            
            self.fps_label = ttk.Label(status_frame, text="FPS: --")
            self.fps_label.pack()
            
            self.ear_label = ttk.Label(status_frame, text="EAR: --")
            self.ear_label.pack()
            
            self.mar_label = ttk.Label(status_frame, text="MAR: --")
            self.mar_label.pack()
            
            self.score_label = ttk.Label(status_frame, text="Drowsiness: --%")
            self.score_label.pack()
            
            model_status = "✅ Models loaded" if self.models_ready else "❌ Models not loaded"
            self.model_label = ttk.Label(status_frame, text=model_status)
            self.model_label.pack()
            
            controls_frame = ttk.LabelFrame(left_frame, text="Controls", padding="10")
            controls_frame.pack(fill=tk.X, pady=10)
            
            button_frame = ttk.Frame(controls_frame)
            button_frame.pack()
            
            self.start_button = ttk.Button(button_frame, text="▶️ Start", 
                                          command=self.start_detection, width=10)
            self.start_button.pack(side=tk.LEFT, padx=5)
            
            self.stop_button = ttk.Button(button_frame, text="⏸️ Stop", 
                                         command=self.stop_detection, width=10, state='disabled')
            self.stop_button.pack(side=tk.LEFT, padx=5)
            
            self.sound_button = ttk.Button(button_frame, text="🔊 Sound ON", 
                                          command=self.toggle_sound, width=10)
            self.sound_button.pack(side=tk.LEFT, padx=5)
            
            self.silence_button = ttk.Button(button_frame, text="🔇 5s", 
                                            command=self.manual_silence_5s, width=10)
            self.silence_button.pack(side=tk.LEFT, padx=5)
            
            self.exit_button = ttk.Button(controls_frame, text="❌ Exit", 
                                         command=self.close_application, width=10)
            self.exit_button.pack(pady=5)
            
            settings_frame = ttk.LabelFrame(left_frame, text="Settings", padding="10")
            settings_frame.pack(fill=tk.X, pady=10)
            
            sound_frame = ttk.Frame(settings_frame)
            sound_frame.pack(fill=tk.X, pady=5)
            
            ttk.Label(sound_frame, text="Sound:").pack(side=tk.LEFT)
            self.sound_select_button = ttk.Button(sound_frame, text="Select File", 
                                                 command=self.select_sound_file, width=10)
            self.sound_select_button.pack(side=tk.LEFT, padx=5)
            
            self.sound_file_label = ttk.Label(sound_frame, text="System Beep", width=20)
            self.sound_file_label.pack(side=tk.LEFT, padx=5)
            
            volume_frame = ttk.Frame(settings_frame)
            volume_frame.pack(fill=tk.X, pady=5)
            
            ttk.Label(volume_frame, text="Volume:").pack(side=tk.LEFT)
            self.volume_var = tk.DoubleVar(value=self.volume)
            self.volume_scale = ttk.Scale(volume_frame, from_=0.0, to=1.0, 
                                         variable=self.volume_var, 
                                         command=self.update_volume, length=150)
            self.volume_scale.pack(side=tk.LEFT, padx=10)
            self.volume_value_label = ttk.Label(volume_frame, text=f"{self.volume:.1f}")
            self.volume_value_label.pack(side=tk.LEFT)
            
            video_frame = ttk.LabelFrame(right_frame, text="Camera Feed", padding="5")
            video_frame.pack(fill=tk.BOTH, expand=True)
            
            self.video_label = ttk.Label(video_frame)
            self.video_label.pack(fill=tk.BOTH, expand=True)
            
            self.root.protocol("WM_DELETE_WINDOW", self.close_application)
            
            self.root.after(100, self.update_gui_from_queue)
            self.root.after(100, self.update_video_display)
            self.root.after(100, self.update_system_info_gui)
            
            self.root.after(1000, self.scan_cameras_gui)
            
            self.gui_ready = True
            print("GUI initialized successfully")
            
        except Exception as e:
            print(f"Error setting up GUI: {e}")
            sys.exit(1)

    def update_system_info(self):
        try:
            self.cpu_usage = self.process.cpu_percent()
            self.memory_usage = self.process.memory_info().rss / 1024 / 1024
            self.system_cpu = psutil.cpu_percent()
            self.system_mem = psutil.virtual_memory().percent
        except Exception as e:
            print(f"Error updating system info: {e}")

    def update_system_info_gui(self):
        if not self._gui_running or not hasattr(self, 'root'):
            return
            
        self.update_system_info()
        
        self.safe_gui_update(self.cpu_label.config, 
                           text=f"CPU: {self.cpu_usage:.1f}% (System: {self.system_cpu:.1f}%)")
        self.safe_gui_update(self.mem_label.config, 
                           text=f"Memory: {self.memory_usage:.1f} MB (System: {self.system_mem:.1f}%)")
        
        if self._gui_running and hasattr(self, 'root'):
            self.root.after(1000, self.update_system_info_gui)

    def safe_gui_update(self, func, *args, **kwargs):
        if self._gui_running and hasattr(self, 'root'):
            try:
                if args or kwargs:
                    self.root.after(0, lambda: func(*args, **kwargs))
                else:
                    self.root.after(0, func)
            except RuntimeError:
                pass

    def scan_cameras_gui(self):
        self.safe_gui_update(lambda: self.scan_button.config(state='disabled', text="🔍 Scanning..."))
        self.status_queue.put({'status': 'Scanning cameras...'})
        
        def scan_thread():
            cameras = self.scan_cameras()
            self.safe_gui_update(self.update_camera_list, cameras)
        
        threading.Thread(target=scan_thread, daemon=True).start()

    def scan_cameras(self):
        self.available_cameras = []
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                self.available_cameras.append(i)
                cap.release()
            time.sleep(0.1)
        return self.available_cameras

    def update_camera_list(self, cameras):
        if not self._gui_running:
            return
            
        self.scan_button.config(state='normal', text="🔍 Scan")
        
        if cameras:
            camera_options = [f"Camera {i}" for i in cameras]
            self.camera_combo['values'] = camera_options
            self.camera_combo.current(0)
            self.status_queue.put({'status': f'Found {len(cameras)} camera(s)'})
        else:
            self.camera_combo['values'] = []
            self.status_queue.put({'status': 'No cameras found'})
            self.safe_gui_update(lambda: messagebox.showwarning("Camera Warning", "No cameras detected"))

    def connect_camera_gui(self):
        if not self.camera_var.get():
            messagebox.showwarning("Selection Error", "Please select a camera first")
            return
        
        camera_index = int(self.camera_var.get().split()[-1])
        
        self.connect_button.config(state='disabled', text="📹 Connecting...")
        
        def connect_thread():
            success = self.init_camera(camera_index)
            self.safe_gui_update(self.update_connection_status, success)
        
        threading.Thread(target=connect_thread, daemon=True).start()

    def init_camera(self, camera_index=0):
        try:
            if hasattr(self, 'cap') and self.cap is not None:
                self.cap.release()
                time.sleep(0.5)

            backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
            for backend in backends:
                try:
                    self.cap = cv2.VideoCapture(camera_index, backend)
                    if self.cap.isOpened():
                        if self.force_arm or platform.machine() in ('arm', 'armv7l', 'aarch64'):
                            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
                            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
                            self.cap.set(cv2.CAP_PROP_FPS, 15)
                        else:
                            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                            self.cap.set(cv2.CAP_PROP_FPS, 30)
                        
                        ret, _ = self.cap.read()
                        if ret:
                            print(f"Successfully initialized camera {camera_index} with backend {backend}")
                            self.selected_camera = camera_index
                            self.camera_ready = True
                            return True
                        
                    self.cap.release()
                except Exception as e:
                    print(f"Camera backend {backend} failed: {str(e)}")
                    continue

            print(f"Could not initialize camera {camera_index} with any backend")
            return False

        except Exception as e:
            print(f"Camera initialization error: {str(e)}")
            return False

    def update_connection_status(self, success):
        if not self._gui_running:
            return
            
        self.connect_button.config(state='normal', text="📹 Connect")
        
        if success:
            self.start_button.config(state='normal')
            messagebox.showinfo("Success", f"Camera {self.selected_camera} connected")
        else:
            messagebox.showerror("Error", "Failed to connect to camera")

    def start_detection(self):
        if not self.camera_ready:
            messagebox.showwarning("Camera Error", "Please connect a camera first")
            return
        
        if not self.models_ready:
            messagebox.showerror("Model Error", "Face detection models not loaded")
            return
        
        self.stop_detection()
        
        self._stop_event.clear()
        
        self.detection_running = True
        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')
        
        self.eye_closed_frames = 0
        self.no_face_frames = 0
        self.alarm_interval = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        self.calibration_start = time.time()
        
        self.perclos_buf.clear()
        self.yawn_buf.clear()
        self.time_buf.clear()
        
        self.ear_s = None
        self.mar_s = None
        self.score_s = None
        self.drowsy_flag = False
        
        if not os.path.exists("OcuPi_BlackBox"):
            os.makedirs("OcuPi_BlackBox")
        
        self.log_file = f"OcuPi_BlackBox/session_{int(time.time())}.log"
        with open(self.log_file, 'w') as f:
            f.write(f"OcuPi Session Log - {time.ctime()}\n")
            f.write(f"Camera: {self.selected_camera}\n")
            f.write(f"Calibration: EAR={self.ear_baseline}, MAR={self.mar_baseline}\n")
            f.write("="*50 + "\n")
        
        with self._video_thread_lock:
            if self.video_thread is None or not self.video_thread.is_alive():
                self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
                self.video_thread.start()

    def stop_detection(self):
        if not self.detection_running:
            return

        self.detection_running = False
        self._stop_event.set()

        self.safe_gui_update(lambda: self.start_button.config(state='normal'))
        self.safe_gui_update(lambda: self.stop_button.config(state='disabled'))

        if hasattr(self, 'root') and self._gui_running:
            self.root.after(500, self.check_video_thread_cleanup)

    def check_video_thread_cleanup(self):
        with self._video_thread_lock:
            if self.video_thread and not self.video_thread.is_alive():
                print("Video thread stopped successfully.")
                self.video_thread = None
            else:
                print("Waiting for video thread to exit...")
                if hasattr(self, 'root') and self._gui_running:
                    self.root.after(500, self.check_video_thread_cleanup)

    def update_gui_from_queue(self):
        if not self._gui_running or not hasattr(self, 'root'):
            return
            
        try:
            while not self.status_queue.empty():
                status_data = self.status_queue.get_nowait()
                
                if 'status' in status_data:
                    self.status_label.config(text=status_data['status'])
                if 'fps' in status_data:
                    self.fps_label.config(text=f"FPS: {status_data['fps']:.1f}")
                if 'ear' in status_data:
                    self.ear_label.config(text=f"EAR: {status_data['ear']:.3f}")
                if 'mar' in status_data:
                    self.mar_label.config(text=f"MAR: {status_data['mar']:.3f}")
                if 'score' in status_data:
                    self.score_label.config(text=f"Drowsiness: {status_data['score']:.1f}%")
                    
        except queue.Empty:
            pass
        
        if self._gui_running and hasattr(self, 'root'):
            self.root.after(100, self.update_gui_from_queue)

    def update_video_display(self):
        if not self._gui_running or not hasattr(self, 'root'):
            return
            
        try:
            if not self.video_queue.empty():
                frame = self.video_queue.get_nowait()
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                frame = cv2.resize(frame, (self.video_width, self.video_height))
                
                img = Image.fromarray(frame)
                
                imgtk = ImageTk.PhotoImage(image=img)
                
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
                
        except queue.Empty:
            pass
        except Exception as e:
            print(f"Error updating video display: {e}")
            
        if self._gui_running and hasattr(self, 'root'):
            self.root.after(30, self.update_video_display)

    def toggle_sound(self):
        self.manual_silence = not self.manual_silence
        if self.manual_silence:
            self.sound_button.config(text="🔇 Sound OFF")
        else:
            self.sound_button.config(text="🔊 Sound ON")

    def manual_silence_5s(self):
        self.manual_silence = True
        self.silence_button.config(text="🔇 Silenced...", state='disabled')
        
        def reset_silence():
            time.sleep(5)
            if self.running:
                self.manual_silence = False
                self.safe_gui_update(lambda: self.silence_button.config(text="🔇 5s", state='normal'))
                self.safe_gui_update(lambda: self.sound_button.config(text="🔊 Sound ON"))
        
        threading.Thread(target=reset_silence, daemon=True).start()

    def select_sound_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Alert Sound",
            filetypes=[("Audio Files", "*.wav *.mp3 *.ogg"), ("All Files", "*.*")]
        )
        
        if file_path:
            self.custom_sound_path = file_path
            self.sound_file_label.config(text=os.path.basename(file_path))
            
            if self.HAS_PYGAME:
                try:
                    import pygame
                    self.alert_sound = pygame.mixer.Sound(file_path)
                    print(f"Loaded custom sound: {file_path}")
                except Exception as e:
                    print(f"Error loading custom sound: {e}")
                    self.custom_sound_path = None
                    self.sound_file_label.config(text="System Beep")
                    messagebox.showerror("Error", "Failed to load sound file")

    def update_volume(self, value):
        self.volume = float(value)
        self.volume_value_label.config(text=f"{float(value):.1f}")
        
        if self.HAS_PYGAME and self.alert_sound:
            import pygame
            self.alert_sound.set_volume(self.volume)

    def close_application(self):
        print("Closing application...")
        self.running = False
        self._gui_running = False
        self._stop_event.set()
        
        self.stop_detection()
        
        self.cleanup()
        
        if hasattr(self, 'root'):
            try:
                self.root.quit()
                self.root.destroy()
            except:
                pass

    def video_loop(self):
        print("Starting video processing...")
        
        try:
            while self.running and not self._stop_event.is_set():
                if not self.detection_running:
                    break

                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame")
                    time.sleep(0.1)
                    continue
                
                processed_frame = self.process_frame(frame)
                
                if not self.cli_mode and processed_frame is not None:
                    try:
                        self.video_queue.put_nowait(processed_frame)
                    except queue.Full:
                        pass
                
                self.frame_count += 1
                elapsed = time.time() - self.start_time
                fps = self.frame_count / elapsed if elapsed > 0 else 0
                self.status_queue.put({'fps': fps})
                
                if self._stop_event.is_set():
                    break
                    
        except Exception as e:
            print(f"Video processing error: {e}")
        finally:
            print("Exiting video loop")
            self._stop_event.clear()

    def process_frame(self, frame):
        try:
            current_time = time.time()
            if current_time - self.last_sys_update > self.sys_update_interval:
                self.update_system_info()
                self.last_sys_update = current_time
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)
            
            h, w, _ = frame.shape
            face_found = bool(results.multi_face_landmarks)
            
            if face_found:
                self.no_face_frames = 0
                
                fl = results.multi_face_landmarks[0]
                coords = [(lm.x * w, lm.y * h) for lm in fl.landmark]
                
                leftEye_idx = [33, 160, 158, 133, 153, 144]
                rightEye_idx = [263, 387, 385, 362, 380, 373]
                leftEye = [coords[i] for i in leftEye_idx]
                rightEye = [coords[i] for i in rightEye_idx]
                
                ear = (self.ear_from_points(leftEye) + self.ear_from_points(rightEye)) / 2.0
                mar = self.mar_from_coords(coords)
                
                self.ear_s = self.ema(self.ear_s, ear, self.EMA_ALPHA)
                self.mar_s = self.ema(self.mar_s, mar, self.EMA_ALPHA)
                
                for idx, (x, y) in enumerate(coords):
                    if idx in leftEye_idx + rightEye_idx or idx in [13, 14, 17, 0, 78, 308]:
                        cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)
                
                ts = time.time()
                
                if ts - self.calibration_start < self.CALIB_SECS:
                    self.ear_baseline = ear if self.ear_baseline is None else 0.9*self.ear_baseline + 0.1*ear
                    self.mar_baseline = mar if self.mar_baseline is None else 0.9*self.mar_baseline + 0.1*mar
                    
                    cv2.putText(frame, "Calibrating... keep eyes open & mouth closed", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
                    cv2.putText(frame, f"EAR0~{self.ear_baseline:.3f}  MAR0~{self.mar_baseline:.3f}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
                    
                    score_pct = 0.0
                    
                    if ts - self.calibration_start > self.CALIB_SECS:
                        self.calibration_data = {
                            'ear_baseline': self.ear_baseline,
                            'mar_baseline': self.mar_baseline,
                            'calibration_time': time.ctime()
                        }
                        self.save_calibration()
                else:
                    if self.ear_baseline is None or self.mar_baseline is None:
                        self.ear_baseline = self.ear_s if self.ear_s is not None else ear
                        self.mar_baseline = self.mar_s if self.mar_s is not None else mar
                    
                    ear_closed_thresh = self.ear_baseline * 0.70
                    mar_yawn_thresh = max(0.5, self.mar_baseline * 1.80)
                    
                    eye_closed_now = 1 if (self.ear_s is not None and self.ear_s < ear_closed_thresh) else 0
                    yawn_now = 1 if (self.mar_s is not None and self.mar_s > mar_yawn_thresh) else 0
                    
                    self.time_buf.append(ts)
                    self.perclos_buf.append(eye_closed_now)
                    self.yawn_buf.append(yawn_now)
                    
                    while self.time_buf and ts - self.time_buf[0] > self.perclos_window:
                        self.time_buf.popleft()
                        self.perclos_buf.popleft()
                        self.yawn_buf.popleft()
                    
                    win_len = max(1, len(self.perclos_buf))
                    perclos = sum(self.perclos_buf) / win_len
                    yawn_frac = sum(self.yawn_buf) / win_len
                    
                    ear_term = 0.0
                    if self.ear_s is not None and self.ear_s < ear_closed_thresh:
                        ear_term = (ear_closed_thresh - self.ear_s) / max(1e-6, ear_closed_thresh)
                        ear_term = max(0.0, min(1.0, ear_term))
                    
                    mar_term = 0.0
                    if self.mar_s is not None and self.mar_s > mar_yawn_thresh:
                        mar_term = (self.mar_s - mar_yawn_thresh) / max(1e-6, (1.20 - mar_yawn_thresh))
                        mar_term = max(0.0, min(1.0, mar_term))
                    
                    perclos_term = max(0.0, min(1.0, (perclos - 0.40) / 0.60))
                    yawn_term = max(0.0, min(1.0, yawn_frac))
                    
                    score_raw = 0.50*perclos_term + 0.25*ear_term + 0.20*yawn_term + 0.05*mar_term
                    self.score_s = self.ema(self.score_s, score_raw, 0.30)
                    score_pct = 100.0 * max(0.0, min(1.0, self.score_s if self.score_s is not None else score_raw))
                    
                    if self.drowsy_flag:
                        self.drowsy_flag = score_pct >= (self.exit_thresh*100)
                    else:
                        self.drowsy_flag = score_pct >= (self.enter_thresh*100)
                    
                    cv2.putText(frame, f"EAR {self.ear_s:.3f}  thr {ear_closed_thresh:.3f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
                    cv2.putText(frame, f"MAR {self.mar_s:.3f}  thr {mar_yawn_thresh:.3f}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
                    cv2.putText(frame, f"PERCLOS {perclos*100:.0f}%  Yawn {yawn_frac*100:.0f}%", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,255,200), 2)
                    
                    if eye_closed_now:
                        self.eye_closed_frames += 1
                    else:
                        self.eye_closed_frames = 0
                    
                    if self.eye_closed_frames >= 25 and not self.manual_silence:
                        self.trigger_alert()
                        self.eye_closed_frames = 0
                    
                    if self.drowsy_flag and self.alarm_interval >= 25 and not self.manual_silence:
                        self.trigger_alert()
                        self.alarm_interval = 0
                    
                    self.alarm_interval += 1
                    
                    self.status_queue.put({
                        'ear': self.ear_s,
                        'mar': self.mar_s,
                        'score': score_pct
                    })
                
                cv2.putText(frame, f"Drowsiness: {score_pct:.1f}%", (10, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255) if self.drowsy_flag else (0,255,0), 2)
                if self.drowsy_flag:
                    cv2.putText(frame, "ALERT: DROWSY", (10, 165),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                
                with open(self.log_file, 'a') as f:
                    f.write(f"{time.ctime()}: EAR={self.ear_s:.3f}, MAR={self.mar_s:.3f}, Score={score_pct:.1f}%\n")
            else:
                self.no_face_frames += 1
                self.eye_closed_frames = 0
                
                if self.no_face_frames >= 15 and not self.manual_silence:
                    self.trigger_alert()
                    self.no_face_frames = 0
                
                self.status_queue.put({
                    'status': 'No face detected',
                    'ear': 0,
                    'mar': 0,
                    'score': 0
                })
                
                cv2.putText(frame, "No face detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)
                self.drowsy_flag = False
            
            if self.cli_mode and self._window_exists(self.video_window_name):
                cv2.imshow(self.video_window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running = False
                elif key == ord('s'):
                    self.manual_silence_5s()
            
            return frame
            
        except Exception as e:
            print(f"Frame processing error: {e}")
            return frame

    def trigger_alert(self):
        if self.sound_playing:
            return
            
        self.sound_playing = True
        
        def play_sound():
            try:
                if self.custom_sound_path and self.HAS_PYGAME and self.alert_sound:
                    import pygame
                    self.alert_sound.play()
                else:
                    print('\a')
            except Exception as e:
                print(f"Error playing sound: {e}")
            finally:
                self.sound_playing = False
        
        threading.Thread(target=play_sound, daemon=True).start()

    def get_system_info(self):
        info = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'opencv_version': cv2.__version__,
            'processor': platform.processor(),
            'machine': platform.machine()
        }
        return info

    def cleanup(self):
        print("Cleaning up resources...")
        
        self.running = False
        self._stop_event.set()
        
        if hasattr(self, 'cap') and self.cap is not None:
            try:
                self.cap.release()
            except:
                pass
        
        if self.cli_mode:
            cv2.destroyAllWindows()
        
        if hasattr(self, 'face_mesh'):
            try:
                self.face_mesh.close()
            except:
                pass

    def run(self):
        try:
            if self.cli_mode:
                print("Running in CLI mode...")
                self.video_loop()
            else:
                print("Running in GUI mode...")
                self.root.mainloop()
        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            self.cleanup()

def main():
    parser = argparse.ArgumentParser(description="OcuPi: MediaPipe Drowsiness Detection")
    parser.add_argument('--cli', action='store_true', help='Run in CLI mode')
    parser.add_argument('--arm', action='store_true', help='Force ARM mode')
    args = parser.parse_args()
    
    detector = OcuPiDetector(cli_mode=args.cli, force_arm=args.arm)
    detector.run()

if __name__ == "__main__":
    main()
