import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import time

def setup_mediapipe():
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1
    )
    return vision.HandLandmarker.create_from_options(options)

def main():
    letter = input("Enter the letter you want to record (A-Z) or 'q' to quit: ").strip().upper()
    if letter == 'Q':
        return
        
    if not (len(letter) == 1 and 'A' <= letter <= 'Z'):
        print("Invalid input. Please enter a single letter from A to Z.")
        return
        
    save_dir = os.path.join("custom_dataset", letter)
    os.makedirs(save_dir, exist_ok=True)
    
    # Count existing images to continue numbering
    existing_files = [f for f in os.listdir(save_dir) if f.endswith('.jpg')]
    count = len(existing_files)
    
    print(f"\nRecording data for '{letter}'")
    print(f"Images will be saved to: {save_dir}")
    print(f"Already have {count} images.")
    print("--------------------------------------------------")
    print("INSTRUCTIONS:")
    print("1. Position your hand in the frame.")
    print("2. Press 's' to save the current frame.")
    print("3. Try slightly different angles and distances for each save.")
    print("4. Press 'q' to stop recording this letter.\n")

    try:
        hands = setup_mediapipe()
    except Exception as e:
        print(f"Error loading MediaPipe task: {e}")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    timestamp_ms = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        try:
            results = hands.detect_for_video(mp_image, timestamp_ms)
        except Exception as e:
            print(f"Error in MediaPipe detection: {e}")
            break
            
        timestamp_ms += 33
        
        display_frame = frame.copy()
        hand_crop = None
        
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
                # Extract original frame region without drawings
                hand_crop = frame[y_min:y_max, x_min:x_max]
                
                # Draw the bounding box on the display frame
                cv2.rectangle(display_frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                
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
                    cv2.line(display_frame, (cx1, cy1), (cx2, cy2), (0, 255, 0), 2)
                    
                for lm in landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(display_frame, (cx, cy), 4, (0, 0, 255), -1)

                cv2.putText(display_frame, f"Ready: Press 's' to save", 
                           (x_min, max(20, y_min-10)), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (255, 0, 0), 2)
        else:
            cv2.putText(display_frame, "No hand detected", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 0, 255), 2)

        cv2.putText(display_frame, f"Count: {count}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 255, 0), 2)

        cv2.imshow(f'Data Collection - Letter {letter}', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            if hand_crop is not None:
                # Save the image
                filename = os.path.join(save_dir, f"{letter}_{count:04d}_{int(time.time())}.jpg")
                # Resize to 224x224 right now so it matches training data exactly
                resized_crop = cv2.resize(hand_crop, (224, 224))
                cv2.imwrite(filename, resized_crop)
                print(f"Saved: {filename}")
                count += 1
                
                # Flash effect on the display frame to confirm save
                cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], display_frame.shape[0]), (0, 255, 0), -1)
                cv2.imshow(f'Data Collection - Letter {letter}', display_frame)
                cv2.waitKey(100)
            else:
                print("Could not save: No hand clearly detected in frame.")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nFinished collecting data for '{letter}'. Total images: {count}")

if __name__ == "__main__":
    main()
