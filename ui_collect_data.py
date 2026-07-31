import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import time
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

class DataCollectionApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("800x650")
        
        self.setup_ui()
        
        # Camera and MediaPipe state
        self.cap = cv2.VideoCapture(0)
        self.timestamp_ms = 0
        self.hands = self.setup_mediapipe()
        self.current_folder = ""
        self.current_count = 0
        self.last_hand_crop = None
        
        # Start update loop
        self.update()

    def setup_ui(self):
        # Top Frame for controls
        top_frame = tk.Frame(self.window)
        top_frame.pack(pady=10)
        
        tk.Label(top_frame, text="Folder Name (Letter):", font=("Helvetica", 12)).grid(row=0, column=0, padx=5)
        
        self.folder_var = tk.StringVar()
        self.folder_entry = tk.Entry(top_frame, textvariable=self.folder_var, font=("Helvetica", 12), width=10)
        self.folder_entry.grid(row=0, column=1, padx=5)
        
        self.change_folder_btn = tk.Button(top_frame, text="Set Folder", font=("Helvetica", 12), command=self.set_folder)
        self.change_folder_btn.grid(row=0, column=2, padx=5)
        
        self.status_label = tk.Label(top_frame, text="Please set a folder.", font=("Helvetica", 12), fg="blue")
        self.status_label.grid(row=0, column=3, padx=15)
        
        # Canvas for video feed
        self.canvas = tk.Canvas(self.window, width=640, height=480, bg="gray")
        self.canvas.pack(pady=10)
        
        # Bottom Frame for capture
        bottom_frame = tk.Frame(self.window)
        bottom_frame.pack(pady=10)
        
        self.capture_btn = tk.Button(bottom_frame, text="📸 Capture Picture", font=("Helvetica", 16, "bold"), 
                                    bg="#4CAF50", fg="white", width=20, height=2, command=self.capture_image, state=tk.DISABLED)
        self.capture_btn.pack()

    def setup_mediapipe(self):
        try:
            base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_hands=1
            )
            return vision.HandLandmarker.create_from_options(options)
        except Exception as e:
            messagebox.showerror("MediaPipe Error", f"Failed to load MediaPipe model:\n{e}")
            return None

    def set_folder(self):
        folder_name = self.folder_var.get().strip().upper()
        
        if not (len(folder_name) == 1 and 'A' <= folder_name <= 'Z'):
            messagebox.showwarning("Invalid Input", "Please enter a single letter from A to Z.")
            return
            
        self.current_folder = folder_name
        save_dir = os.path.join("custom_dataset", self.current_folder)
        os.makedirs(save_dir, exist_ok=True)
        
        # Count existing
        existing_files = [f for f in os.listdir(save_dir) if f.endswith('.jpg')]
        self.current_count = len(existing_files)
        
        self.status_label.config(text=f"Saving to '{self.current_folder}' | Images: {self.current_count}", fg="green")
        self.capture_btn.config(state=tk.NORMAL)

    def capture_image(self):
        if not self.current_folder:
            return
            
        if self.last_hand_crop is not None:
            save_dir = os.path.join("custom_dataset", self.current_folder)
            filename = os.path.join(save_dir, f"{self.current_folder}_{self.current_count:04d}_{int(time.time())}.jpg")
            
            resized_crop = cv2.resize(self.last_hand_crop, (224, 224))
            cv2.imwrite(filename, resized_crop)
            
            self.current_count += 1
            self.status_label.config(text=f"Saved image! | Images: {self.current_count}", fg="green")
            
            # Brief visual feedback
            self.canvas.create_rectangle(0, 0, 640, 480, fill="green", stipple="gray50")
        else:
            self.status_label.config(text="No hand detected to capture!", fg="red")

    def update(self):
        ret, frame = self.cap.read()
        if ret and self.hands:
            # Process frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            
            results = self.hands.detect_for_video(mp_image, self.timestamp_ms)
            self.timestamp_ms += 33
            
            self.last_hand_crop = None
            
            if results.hand_landmarks:
                landmarks = results.hand_landmarks[0]
                h, w = frame.shape[:2]
                x_coords = [lm.x * w for lm in landmarks]
                y_coords = [lm.y * h for lm in landmarks]
                
                x_min = max(0, int(min(x_coords)) - 40)
                y_min = max(0, int(min(y_coords)) - 40)
                x_max = min(w, int(max(x_coords)) + 40)
                y_max = min(h, int(max(y_coords)) + 40)
                
                if x_max > x_min and y_max > y_min:
                    self.last_hand_crop = frame[y_min:y_max, x_min:x_max]
                    cv2.rectangle(frame_rgb, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    
                    # Draw the hand landmarks (the skeleton) manually on the display frame
                    HAND_CONNECTIONS = [
                        (0, 1), (1, 2), (2, 3), (3, 4),
                        (0, 5), (5, 6), (6, 7), (7, 8),
                        (5, 9), (9, 10), (10, 11), (11, 12),
                        (9, 13), (13, 14), (14, 15), (15, 16),
                        (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
                    ]
                    for connection in HAND_CONNECTIONS:
                        idx1, idx2 = connection
                        lm1 = landmarks[idx1]
                        lm2 = landmarks[idx2]
                        cx1, cy1 = int(lm1.x * w), int(lm1.y * h)
                        cx2, cy2 = int(lm2.x * w), int(lm2.y * h)
                        cv2.line(frame_rgb, (cx1, cy1), (cx2, cy2), (0, 255, 0), 2)
                        
                    for lm in landmarks:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame_rgb, (cx, cy), 4, (255, 0, 0), -1)
            
            # Display frame in Tkinter canvas
            frame_rgb = cv2.resize(frame_rgb, (640, 480))
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        # Call update repeatedly
        self.window.after(30, self.update)

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

if __name__ == "__main__":
    # Ensure Pillow is available for Tkinter
    try:
        from PIL import Image, ImageTk
    except ImportError:
        print("Pillow library is required. Please install it using: pip install Pillow")
        exit(1)
        
    root = tk.Tk()
    app = DataCollectionApp(root, "Sign Language Data Collection UI")
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()
