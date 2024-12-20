# Dart Sense

## Description
Dart Sense is an automatic dart scoring application. It uses a sophisticated deep learning-based computer vision model to track the landing coordinates of darts and 4 board calibration points in an image in order to predict the scores for each dart. This app can be used to score a game of darts in real time using video streamed from a smartphone.

A YOLOv8 object detection model was trained to detect dart positions and board calibration points. By comparing the coordinates of the calibration points in an image to the standardised dart board dimensions, you can then transform the image, allowing any set of coordinates to be mapped to a darts score. I developed this system for my Artifical Intelligence Msc Dissertation project. Since completing my studies, I have continued to work on this project, collecting more data to further improve the model and developing a fully functional GUI to run the system in real time, with added game logic to be able to play a game of darts with real players without the burden of scoring it yourself. Please see a video demonstration of the app below!

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

## Files
A small sample of images and respective labels can be found in the `data` folder.

`prepare_data.py` shows the pre-processing steps taken.

Training script in `training` folder. This would need altering before attempting to train on your own machine.

`accuracy.py` computes the accuracy metrics based on the input predictions and the ground truth data.

`get_scores.py` has all the logic for computing the score for a dart based on its coordinates.

`game_logic` sets the scoring, turn-taking logic and rules for a game of darts with multiple players.

`GUI.py` is the Tkinter app. It handles the graphics. This ties closely into `video_processing` which processes every frame of the video and extracts the best possible prediction for the current darts score.

## Methodology
### Data Collection and Pre-Processing
For this project, open source data from McNally et al. 2021 was used, consisting of ~16,000 images. This data was collected by the authors who played darts and took photos of the board after each throw. I collected a further 8,000 images in order to provide a greater variety of data for the object detection model to train on. These images were collected from Youtube videos, as well as my own playing set-up, allowing for an array of different playing configurations (e.g dart type, board type, colours, lighting, camera angles) to be sampled.

After collecting the images, further pre-processing was completed to ensure that the images were in the right form for the object detection model. All images were resized to 800x800 pixels in alignment with D1 and D2 from McNally et al. 2021. Furthermore, some of the lower resolution data from videos were sharpened to visually improve the visibility of edges within the images.

Finally, each data set was split into training (75%), validation (10%), and testing (15%) subsets using random sampling.

### Data labelling
All of the collected images required manual labelling for the object detection model. The program ’LabelImg’ was used for this. It has a GUI which shows the images and allows the user to draw bounding boxes for chosen classes, and then save the labels in the YOLO format. Each data set was loaded into the program and each image was labelled by hand. The labels were pre-processed to change the sizes of all the bounding boxes to 0.025 (2.5% of the image’s size) as this has been shown to work well for D1 and D2 (McNally et al. 2021).

Once a sufficiently accurate model had been trained, I was able to use this to make predictions for the rest of the unlabelled data, and then validate and change the annotations as necessary upon review.

### Training
You Only Look Once (YOLO) object detection was used to detect the landing positions of the
darts as well as 4 dart board calibration points. YOLOv8n was
used as this has the best benchmark results, and the nano size was used as it is important that the model is able to run in real time.

Model tuning is crucial in order to get the best results. Manual experiments and using YOLO’s tuning algorithm were used to find the best hyperparameters for training. The YOLOv8 genetic tuning algorithm was used to find the best hyperparameters for the model over many iterations. The model was tuned for 134 iterations each consisting of 25 epochs. For each iteration, the fitness of the hyperparameters is evaluated based on the performance of the model.

Comparing the F1 scores for a model using trained hyperparameters compared to the default values shows the benefits of tuning (shown below). D5 is an especially difficult dataset, taken from a unique video. No data from D5 was used to train the model, so the much improved performance on this data set is indicative of a model that can generalise effectively. This is likely due to the data augmentation being used in the tuned model. It used image rotation, scaling, translation and brightness/contrast adjustments.

<img src="images/hyperparameters.png" alt="alt text" width="50%">

### Score Prediction
In order to use the predicted dart locations from the YOLO model, the coordinates need to be contextualised with reference to some measure of the standardised dart board dimensions. The standardised dimensions of the dart board were used to generate coordinates for 4 calibration points around the board. The chosen points are the upper left corner of the double segments for the 20, 6, 3 and 11, illustrated in 3.2b. The YOLO model is trained to
detect these same calibration points on the images, so then the image can be transformed to the reference plane, and dart positions can be scored using the standardised board segment dimensions. This allows for darts to be accurately scored regardless of camera angle and any distortion effects in the image.

![alt text](images/transformation.png)

I tried another approach to the score prediction, which used a combination of Hough line detection and pixel colour analysis in order to segment the board. Lines were detected to delineate segments, then colour masks were used to find green and red segments. Following a line to the bullseye, the scoring multiplier was computed based on the number of green, red and other areas the line passes through. This method had many failure cases due to the colour of the darts themselves obscurring the board. This method is not used in the final app.

![alt text](images/1054_171023_48.png)

### Results
Results were evaluated based on the following custom metrics:
![alt text](images/accuracy_metrics.png)

The final model was trained for 100 epochs with the YOLOv8n (nano) architecture using
tuned hyperparameters and a batch size of 79.

Here are the final results on test data:
| Image PCS | Precision | Recall | Missed | Extra |
|-----------|-----------|--------|--------|-------|
| 0.884     | 0.977     | 0.942  | 0.040  | 0.004 |

The main failure cases are due to dart obfuscation. To an extent it becomes an impossible task, as even humans can't be 100% sure of a dart's score, and rely on moving in order to get a better viewpoint. I do believe that with more data to train the model this can imperove further and it will be able to exceed human performance on the task.

Here is an example where a dart is missed as the tip cannot be seen:
![alt text](images/1258_171023_581.png)
