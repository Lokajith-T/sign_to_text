# Real-Time Sign Language Translator

A computer vision-based real-time American Sign Language (ASL) alphabet translator. This project uses Google's **MediaPipe** for robust hand tracking and landmark extraction, and a custom **PyTorch** ResNet18-based neural network for alphabet classification (A-Z).

## Features

- **Real-Time ASL Translation**: Live translation of hand gestures to text using your webcam.
- **Custom Dataset Collection**: Two data collection scripts (CLI and GUI based) allow you to easily capture and expand your own ASL dataset.
- **Easy Fine-Tuning**: A built-in PyTorch fine-tuning script to quickly train the base model on your newly collected custom gestures.
- **Skeleton Visualization**: Real-time rendering of the hand skeleton (landmarks and connections) over the video feed for clear visual feedback.

## Requirements

Ensure you have Python installed. You can install the required packages using:
```bash
pip install torch torchvision opencv-python mediapipe Pillow
```
*(Or install via `requirements.txt` if available)*

## Project Structure

- `app.py`: The main application. Opens the webcam, detects hand landmarks via MediaPipe, feeds the crop to the PyTorch model, and displays the predicted letter in real-time.
- `ui_collect_data.py`: A user-friendly graphical interface (built with Tkinter) to capture and save images of hand gestures for training.
- `collect_data.py`: A command-line alternative for collecting training data images.
- `fine_tune.py`: A script to retrain/fine-tune the PyTorch model on the data collected in the `custom_dataset` folder.
- `fine_tuned_model.pth` / `best_model.pth`: PyTorch model weight files.
- `hand_landmarker.task`: The MediaPipe pre-trained model for hand landmark detection.

## Usage

### 1. Running the Translator
To run the live translator:
```bash
python app.py
```
Press `q` to quit the application.

### 2. Collecting Your Own Data
If you want to train the model on your own hand or add new environments, use the GUI data collection tool:
```bash
python ui_collect_data.py
```
Enter the letter you want to record, position your hand in the green bounding box, and click the capture button to save images. They will be saved to the `custom_dataset/` directory.

### 3. Fine-Tuning the Model
Once you have collected images in your `custom_dataset/` folder, run the fine-tuning script to train the model on your new data:
```bash
python fine_tune.py
```
This will update and save a `fine_tuned_model.pth` file. `app.py` is configured to use this newly trained model automatically.

## Architecture

1. **Hand Detection**: `mediapipe.tasks.vision.HandLandmarker` finds the hand and extracts 21 3D landmarks.
2. **Cropping**: A dynamic bounding box is calculated around the detected hand, isolating it from the background.
3. **Classification**: The isolated hand image is resized to 224x224 and fed into a PyTorch ResNet18 model modified to output 26 classes corresponding to the A-Z alphabet.

## Author

- **Lokajith T**
