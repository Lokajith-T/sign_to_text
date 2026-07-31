import cv2
import mediapipe as mp
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--image_path", default="sample.jpg", help="Process an image.")
args = parser.parse_args()

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

def draw_landmarks_on_image(rgb_image, detection_result):
    if not detection_result.hand_landmarks:
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

print(f"Attempting to process image: {args.image_path}")

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    num_hands=2,
    running_mode=VisionRunningMode.IMAGE)

with HandLandmarker.create_from_options(options) as landmarker:
    image = cv2.imread(args.image_path)
    if image is None:
        print(f"Could not read image from {args.image_path}. Please provide a valid --image_path.")
        exit(1)
        
    rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    results = landmarker.detect(mp_image)

    if not results.hand_landmarks:
        print("No hands detected.")
    else:
        print(f"Detected {len(results.hand_landmarks)} hand(s).")
        annotated_image = draw_landmarks_on_image(image, results)
        
        cv2.imshow('MediaPipe Hands - Image', annotated_image)
        print("Press any key on the image window to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
