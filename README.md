# Hanasam

**Hanasam** stands for **Hand as a Mouse**, which is a program that lets you use your hand to control your computer based on computer vision.

## Before you start
Hanasam is a project I wrote in 2024. The model is from Google's MediaPipe collection, and I did nearly nothing, so it's just a fun application of Google's hand landmarking model. You might find it useful in some very special cases.

## How to set it up
Install the necessary packages:

```bash
pip install -r requirements.txt
```

Then just run main.py, it's that simple!

## How to use it

After main.py starts running, you will first see a window displaying the camera stream and current landmarks.

Show one of your hands in front of your camera, and the cursor will move with your hand movement.

Pinch with your thumb and index finger to left click (hold to long click), or pinch with your thumb and middle finger to right click.

That's all the functionality of this program!
Hope this helps, or is at least fun.