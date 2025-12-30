# Human Fall Detection

## Initialization
# Importing the libraries
import cv2
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
import math

# Loading the detector
model = hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1")
movenet = model.signatures["serving_default"]


### Keypoints Drawing
# Draw the keypoints
def draw_keypoints(frame, keypoints, threshold=0.3):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (255, 0, 0), -1)


### Edge and Connection Drawing
edges = {
    (0, 1): "m",
    (0, 2): "c",
    (1, 3): "m",
    (2, 4): "c",
    (0, 5): "m",
    (0, 6): "c",
    (5, 7): "m",
    (7, 9): "m",
    (6, 8): "c",
    (8, 10): "c",
    (5, 6): "y",
    (5, 11): "m",
    (6, 12): "c",
    (11, 12): "y",
    (11, 13): "m",
    (13, 15): "m",
    (12, 14): "c",
    (14, 16): "c",
}


# Draw the skeleton
def draw_skeleton(frame, keypoints, edges, threshold=0.3):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        if (c1 > threshold) & (c2 > threshold):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)


### Boundaries Drawing


## Detections
### Human Detection
# Function to loop through each frame in the video
def loop_through_people(frame, keypoints_with_scores, edges, threshold):
    for person in keypoints_with_scores:
        if np.mean(person[:, 2]) < threshold:
            continue
            
        draw_skeleton(frame, person, edges, threshold)
        draw_keypoints(frame, person, threshold)
        
        # Determine basic posture
        base_posture = get_posture(frame, person, threshold)
        
        # Determine Binary Emergency State
        person_idx = keypoints_with_scores.tolist().index(person.tolist()) + 1
        binary_class = analyze_temporal_states(person_idx, base_posture)
        
        # Color & Label based on the 2 requested classes
        if binary_class == "FALL (EMERGENCY)":
            color = (0, 0, 255) # Bright Red
            display_label = "FALL (EMERGENCY)"
        else:
            color = (0, 255, 0) # Bright Green
            display_label = "NORMAL"

        # Get bounding box for label positioning
        y, x, c = frame.shape
        shaped = np.squeeze(np.multiply(person, [y, x, 1]))
        x1, y1, x2, y2 = x, y, 0, 0
        for kp in shaped:
            ky, kx, kp_conf = kp
            if kp_conf > threshold:
                x1, y1 = min(x1, kx), min(y1, ky)
                x2, y2 = max(x2, kx), max(y2, ky)
        
        # Draw box and label
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, display_label, (int(x1), int(y1) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


### Fall Detection
# Analysis purposes
# Record coordinates of nose for each person
y_coordinates = {
    1: [],
    2: [],
    3: [],
    4: [],
    5: [],
    6: [],
}

x_coordinates = {
    1: [],
    2: [],
    3: [],
    4: [],
    5: [],
    6: [],
}

# Record aspect ratio of bounding box for each person
aspect_ratios = {
    1: [],
    2: [],
    3: [],
    4: [],
    5: [],
    6: [],
}

# Record angle of bounding box for each person
angles = {
    1: [],
    2: [],
    3: [],
    4: [],
    5: [],
    6: [],
}

# Record frame when person falls for analysis
fall_frames = {i: [] for i in range(1, 7)}

# Record movement history for temporal analysis (last 100 frames)
movement_history = {i: [] for i in range(1, 7)}

def analyze_temporal_states(person_idx, current_posture):
    history = movement_history[person_idx]
    if len(history) < 30: # Need at least 1 second of data
        return "NORMAL" if current_posture != "FALLING" else "FALL (EMERGENCY)"

    # Calculate movement intensity over the last 30 frames
    recent_coords = [h[:2] for h in history[-30:]]
    dx = np.std([c[0] for c in recent_coords])
    dy = np.std([c[1] for c in recent_coords])
    movement_intensity = dx + dy

    # Binary Logic following the provided requirements:
    
    # FALL (EMERGENCY) Cases:
    # 1. Sudden rapid collapse (High velocity leading to flat posture)
    if current_posture == "FALLING":
        # Check if they were standing recently (rapid transition)
        rapid_drop = any(h[2] == "STANDING" for h in history[-20:])
        if rapid_drop or movement_intensity > 10.0:
            return "FALL (EMERGENCY)"
        
        # 2. No movement after falling (Fainted)
        if movement_intensity < 2.0:
            return "FALL (EMERGENCY)"
            
        # 3. Distress (Thrashing while down)
        if movement_intensity > 15.0:
            return "FALL (EMERGENCY)"

    # NORMAL Cases:
    # 1. Intentional Lying down (Slower transition Standing -> Sitting -> Lying)
    if current_posture == "FALLING":
        was_sitting = any(h[2] == "SITTING" for h in history[-40:])
        if was_sitting and movement_intensity < 8.0:
            return "NORMAL" # Intentional lying on bed/floor

    # Default to NORMAL for all other states (Standing, Sitting, Bending, etc.)
    return "NORMAL"


# Function to calculate aspect ratio of bounding box
def calculate_aspect_ratio(frame, keypoints_with_scores):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints_with_scores, [y, x, 1]))
    x1, y1, x2, y2 = x, y, 0, 0
    for kp in shaped:
        ky, kx, kp_conf = kp
        x1, y1 = min(x1, kx), min(y1, ky)
        x2, y2 = max(x2, kx), max(y2, ky)
    aspect_ratio = (x2 - x1) / (y2 - y1)
    return aspect_ratio


# Function to classify posture for each person
def get_posture(frame, keypoints, threshold=0.3):
    # Check if we have enough confident keypoints
    conf_scores = keypoints[:, 2]
    if np.mean(conf_scores) < threshold:
        return "UNKNOWN"

    aspect_ratio = calculate_aspect_ratio(frame, keypoints)
    angle = calculate_angle(frame, keypoints)

    # Classification Logic
    if aspect_ratio > 1.5 and angle < 45:
        return "FALLING"
    elif 0.8 < aspect_ratio <= 1.5:
        return "SITTING"
    else:
        return "STANDING"


# Function to detect fall and record data (kept for analysis)
def detect_fall(frame, keypoints_with_scores, frame_count):
    found_fall = False
    for i, person in enumerate(keypoints_with_scores):
        # Check confidence
        if np.mean(person[:, 2]) < 0.3:
            continue

        angle = calculate_angle(frame, person)
        angles[i + 1].append(angle)

        y_current = person[0][0] * frame.shape[0]
        x_current = person[0][1] * frame.shape[1]
        y_coordinates[i + 1].append(y_current)
        x_coordinates[i + 1].append(x_current)

        aspect_ratio = calculate_aspect_ratio(frame, person)
        aspect_ratios[i + 1].append(aspect_ratio)

        # Update Movement History for Temporal Analysis
        # Use nose coordinates for tracking
        movement_history[i + 1].append((x_current, y_current, get_posture(frame, person)))
        if len(movement_history[i + 1]) > 100: # Keep last 100 frames
            movement_history[i + 1].pop(0)

        # Record fall for analysis plot
        if aspect_ratio > 1.5 and angle < 45:
            fall_frames[i + 1].append(frame_count)
            found_fall = True
            
    return found_fall


## Fall Confirmation
# Function to calculate angle of centroid with respect to horizontal axis of bounding box per frame
def calculate_angle(frame, keypoints_with_scores):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints_with_scores, [y, x, 1]))
    x1, y1, x2, y2 = x, y, 0, 0
    for kp in shaped:
        ky, kx, kp_conf = kp
        x1, y1 = min(x1, kx), min(y1, ky)
        x2, y2 = max(x2, kx), max(y2, ky)
    centroid = (x1 + x2) / 2, (y1 + y2) / 2
    angle = math.atan2(centroid[1] - y1, centroid[0] - x1) * 180 / math.pi
    return angle


# If using webcam, use 0
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

# Get frame length of video (Note: may be 0 for webcams)
frame_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
total_frames_processed = 0

# Looping through the frames
while 1:
    try:
        ret, frame = cap.read()
        # Display number of frames
        cv2.putText(
            frame,
            f"Frame: {total_frames_processed}",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        # Status text
        cv2.putText(
            frame,
            "STATUS: MONITORING",
            (50, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

        # Load frame from the video into the detector
        img = frame.copy()
        # Adjust the size of the image according to the video resolution, as long as it is divisible by 32
        img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 160, 320)
        img = tf.cast(img, dtype=tf.int32)

        # Detecting objects in the image
        result = movenet(img)
        # Process the keypoints
        keypoints_with_scores = result["output_0"].numpy()[:, :, :51].reshape(6, 17, 3)

        # Render keypoints
        loop_through_people(frame, keypoints_with_scores, edges, 0.3)

        # Detecting fall for final alert
        curr_state = analyze_temporal_states(1, get_posture(frame, keypoints_with_scores[0]))
        if curr_state == "FALL (EMERGENCY)":
            cv2.putText(
                frame,
                "!!! FALL EMERGENCY !!!",
                (50, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 0, 255),
                3,
                cv2.LINE_AA,
            )

        # Displaying the frame
        cv2.imshow("frame", frame)
        total_frames_processed += 1

        # Pressing escape to exit
        if cv2.waitKey(33) == 27:
            break
    except:
        break

# Releasing the capture and destroying all windows
cap.release()
cv2.destroyAllWindows()

## Analysis
### Frames when fall is detected
# Plot the frame when fall was detected for each person
analysis_frames = total_frames_processed if frame_length <= 0 else frame_length

for i in range(1, 7):
    for j in range(analysis_frames):
        if j in fall_frames[i]:
            plt.scatter(j, i)
plt.show()

### Nose coordinates difference
# Plot the x and y coordinates of the nose of each person
plt.figure(figsize=(20, 10))
for i in range(1, 7):
    plt.subplot(2, 3, i)
    plt.plot(y_coordinates[i], label=f"Person {i}", color="red")
    plt.plot(x_coordinates[i], label=f"Person {i}", color="blue")
    plt.title(f"Person {i}")
    plt.xlabel("Frame")
    plt.ylabel("Coordinate")
plt.legend()
plt.show()

### Aspects ratio
# Plot the aspect ratio of each person
plt.figure(figsize=(20, 10))
for i in range(1, 7):
    plt.subplot(2, 3, i)
    plt.plot(aspect_ratios[i], label=f"Person {i}")
    plt.title(f"Person {i}")
    plt.xlabel("Frame")
    plt.ylabel("Aspect ratio")
plt.legend()
plt.show()

### Angles of centroid with respect to horizontal axis of bounding box
# Plot the angle of each person
plt.figure(figsize=(20, 10))
for i in range(1, 7):
    plt.subplot(2, 3, i)
    plt.plot(angles[i], label=f"Person {i}")
    plt.title(f"Person {i}")
    plt.xlabel("Frame")
    plt.ylabel("Angle")
plt.legend()
plt.show()

### First Person Analysis
# Since on the sample video, there is only one person, we can analyze the data for one person only.
# Plot the x and y coordinates of the nose for the first person
plt.figure(figsize=(12, 8))
plt.plot(y_coordinates[1], label=f"y-coordinate", color="red")
plt.plot(x_coordinates[1], label=f"x-coordinate", color="blue")
for i in range(analysis_frames):
    if i in fall_frames[1]:
        plt.scatter(i, 1)
plt.title(f"Nose coordinates")
plt.xlabel("Frame")
plt.ylabel("Coordinate")
plt.legend()
plt.show()

# When fall is detected, x coordinate tend to be higher than y coordinate. This is because when a person fall, the person will fall to the ground, which is at the bottom of the frame. This is shown in the graph above.

# Plot the aspect ratio for the first person
plt.figure(figsize=(12, 8))
plt.plot(aspect_ratios[1])
for i in range(analysis_frames):
    if i in fall_frames[1]:
        plt.scatter(i, 1)
plt.title(f"Apect ratio")
plt.xlabel("Frame")
plt.ylabel("Aspect ratio")
plt.show()

# When fall is detected, the aspect ratio tend to change drastically. This is because when a person fall, the bounding box will change drastically. This is shown in the graph above.

# Plot the angle for the first person
plt.figure(figsize=(12, 8))
plt.plot(angles[1])
for i in range(analysis_frames):
    if i in fall_frames[1]:
        plt.scatter(i, 1)
plt.title(f"Angle of centroid with respect to horizontal axis of bounding box")
plt.xlabel("Frame")
plt.ylabel("Angle")
plt.show()

# When fall is detected, the angle of centroid with respect to horizontal axis of bounding box tend to be lower than 45 degree. This is because when a person fall, the person will fall to the ground, which is at the bottom of the frame. This is shown in the graph above.
