# AeroGestus
A modular gesture recognition system using MediaPipe landmarks and CLIP embeddings to control Windows media. Currently supports static poses and temporal motion sequences.

_Will add further control functionalities on future dates._

## Installation
1. Install Python 3.13 or higher.
2. Install dependencies:
> pip install torch torchvision mediapipe opencv-python pillow clip-by-openai customtkinter
3. Ensure gesture_recognizer.task is in the root directory. Can be downloaded from <https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer> if necessary.
4. Run 
> python app.py

## Build Executable
This is optional. You can run the project with the commands above.
1. Install PyInstaller:
> pip install pyinstaller
2. Run the build script: 
> python build.py
3. The executable will be generated in the `dist/ folder.`

_It is to be noted that the Build executable file currently spans 2.5GB due to the base size of CLIP model._
