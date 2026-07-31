import torch
import torch.nn as nn
from torchvision import models
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from torchvision import transforms
from PIL import Image

# Define model architecture
class SignLanguageModel(nn.Module):
    def __init__(self, num_classes=26, pretrained=False):
        super().__init__()
        self.model = models.resnet18(pretrained=pretrained)
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

def main():
    print("Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SignLanguageModel(num_classes=26)
    
    model_path = 'fine_tuned_model.pth'
    mediapipe_path = 'hand_landmarker.task'
    
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.to(device)
    model.eval()

    print(f"Model loaded successfully!")
    if 'val_acc' in checkpoint:
        print(f"Validation Accuracy: {checkpoint['val_acc']:.2f}%")

    print("Setting up MediaPipe...")
    # Setup MediaPipe hand detector
    try:
        base_options = python.BaseOptions(model_asset_path=mediapipe_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1
        )
        hands = vision.HandLandmarker.create_from_options(options)
    except Exception as e:
        print(f"Error loading MediaPipe task: {e}")
        return

    # Preprocessing transform
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Class mapping
    idx_to_class = {i: chr(65+i) for i in range(26)}  # A-Z

    print("Starting webcam...")
    # Capture from webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    timestamp_ms = 0
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Detect hand
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        try:
            results = hands.detect_for_video(mp_image, timestamp_ms)
        except Exception as e:
            print(f"Error in MediaPipe detection: {e}")
            break
            
        timestamp_ms += 33  # ~30 FPS
        
        if results.hand_landmarks:
            landmarks = results.hand_landmarks[0]
            
            # Extract hand region
            h, w = frame.shape[:2]
            x_coords = [lm.x * w for lm in landmarks]
            y_coords = [lm.y * h for lm in landmarks]
            
            x_min = max(0, int(min(x_coords)) - 40)
            y_min = max(0, int(min(y_coords)) - 40)
            x_max = min(w, int(max(x_coords)) + 40)
            y_max = min(h, int(max(y_coords)) + 40)
            
            if x_max > x_min and y_max > y_min:
                hand_crop = frame[y_min:y_max, x_min:x_max]
                
                # Preprocess and predict
                pil_image = Image.fromarray(cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB))
                tensor = preprocess(pil_image).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    outputs = model(tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    top_prob, top_idx = torch.max(probabilities, dim=1)
                    
                    predicted_class = idx_to_class[top_idx.item()]
                    confidence = top_prob.item() * 100
                    
                    # Display prediction
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    
                    # Draw the hand landmarks (the skeleton) manually on the frame
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
                        cv2.line(frame, (cx1, cy1), (cx2, cy2), (0, 255, 0), 2)
                        
                    for lm in landmarks:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

                    cv2.putText(frame, f"{predicted_class}: {confidence:.1f}%", 
                               (x_min, max(20, y_min-10)), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.9, (0, 255, 0), 2)
        
        cv2.imshow('ASL Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
