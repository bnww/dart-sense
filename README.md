# Dart Sense

## Description
Dart Sense is an automatic dart scoring application. It uses a sophisticated deep learning-based computer vision model to track the landing coordinates of darts and 4 board calibration points in an image in order to predict the scores for each dart. This app can be used to score a game of darts in real time using video streamed from a smartphone.

## [Video Demonstration of app](https://www.youtube.com/watch?v=8a97sVmbqY0)

## Setup
1. Clone the repository: `git clone https://github.com/bnww/dart-sense.git`
2. Install the required dependencies: `npm install requirements.txt`
3. Download 'IP Webcam' on your smartphone so that you can stream video from your smartphone camera to the app for processing.

## Features
- Automatic scoring works on any dart board and with any darts!
- Stream from your smartphone camera
- Score tracking for various dart games (x01, cricket)
- Play solo or with up to 6 friends
- Customizable settings

![image output 1](images/0022_151123_2715.png)

Example images fed into the YOLO Object Detection model. 'YOLO model prediction' shows the YOLO predicted classes and respective probabilities. 'Predicted scores after transformation' shows the result after processing the YOLO results. The calibration points (corner of 20, 6, 3, and 11 segments) are used to transform the dart coordinates to a standardised reference frame, so that they can be attributed with scores. This allows accurate scoring regardless of the angle or distance of the camera to the board, so it can be used in any playing set-up!

![image output 2](images/d2_02_03_2021_2_DSC_0059.JPG)


## Usage
1. Run `GUI.py` to launch the app
2. Configure your game settings. **Important: replace Webcam IP with the IP shown when you start your IP Webcam server.**<br />
![app set up screen](images/set_up_screen.png)
3. Enjoy your game of darts!

