# Human Fall Detection

## Introduction

This project is a human fall detection system based on the [MoveNet](https://tfhub.dev/google/movenet/multipose/lightning/1) model. The system is able to detect human fall in real-time using a video or webcam.

## Detection Method

- The system uses the MoveNet model to detect the keypoints of the human body
- The keypoints are then used to calculate:
  - The x and y coordinates of the nose that represent the head position
  - The aspect ratio of the body that represents the body position
  - The angle of centroid of the body with respect to the x-axis that represents the body orientation
- The head position and body position are then used to determine the fall detection
- The body orientation is then used to confirm the fall detection

## Usage

1. Clone the repository
2. Install the required packages
3. Change the input based on your needs

3.1 For video input, change the `queda.mp4` in VideoCapture to the path of your video file

```python
# Insert video file name here
cap = cv2.VideoCapture('queda.mp4')
```

3.2 For webcam input, use the following code

```python
# If using webcam, uncomment the line below and comment the line above
cap = cv2.VideoCapture(1)
```

4. Run the code
5. Press `esc` to exit the program

## References

- [MoveNet](https://tfhub.dev/google/movenet/multipose/lightning/1)
- [Video Fall Detection](https://github.com/EikeSan/video-fall-detection)
- [TensorFlow Multipose Estimation by Nicholas Renotte](https://www.youtube.com/watch?v=KC7nJtBHBqg&t=2056s)
- [Automatic Detection of Human Fall in Video by Vinay Vishwakarma, Chittaranjan Mandal, and Shamik Sural](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=7d1b7276c8119e12a7a6f4408bc95ba7b80ccbc9)
