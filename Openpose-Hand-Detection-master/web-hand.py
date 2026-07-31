import cv2
import mediapipe as mp
import time

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8), # Index
    (5, 9), (9, 10), (10, 11), (11, 12), # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Pinky
]

# Global variable to store the latest result
latest_result = None

def print_result(result: mp.tasks.vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2,
    result_callback=print_result)

def draw_landmarks_on_image(rgb_image, detection_result):
    if detection_result is None or not detection_result.hand_landmarks:
        return rgb_image

    annotated_image = rgb_image.copy()
    height, width, _ = annotated_image.shape

    for hand_landmarks in detection_result.hand_landmarks:
        # Draw connections
        for connection in HAND_CONNECTIONS:
            start_idx, end_idx = connection
            start_point = hand_landmarks[start_idx]
            end_point = hand_landmarks[end_idx]
            cv2.line(annotated_image, 
                     (int(start_point.x * width), int(start_point.y * height)), 
                     (int(end_point.x * width), int(end_point.y * height)), 
                     (0, 255, 0), 2)
            
        # Draw points
        for landmark in hand_landmarks:
            cv2.circle(annotated_image, 
                       (int(landmark.x * width), int(landmark.y * height)), 
                       5, (0, 0, 255), -1)

    return annotated_image

print("Starting webcam. Press 'Esc' to exit.")
cap = cv2.VideoCapture(0)

with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert frame to RGB for mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Calculate timestamp
        timestamp_ms = int(time.time() * 1000)
        
        # Detect asynchronously
        landmarker.detect_async(mp_image, timestamp_ms)

        # Draw results on the frame
        annotated_image = draw_landmarks_on_image(frame, latest_result)

        # Flip horizontally and display
        cv2.imshow('MediaPipe Hand Detection - Webcam', cv2.flip(annotated_image, 1))

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
