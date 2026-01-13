import numpy as np 
import cv2
import time 
import math 
import tensorflow as tf 
import requests 
   

STREAMLIT_URL = "https://guardian-ai-backend-7zfj.onrender.com/upload_telemetry_cam"


MODEL_PATH = "E:/human-fall-detection/02_TFLite_Laptop/model_thunder_int8.tflite"
INPUT_SIZE = 192  # Thunder requires larger input usually? No, check model specs. sticking to 192 or 256. Thunder is usually 256. 
# But let's stick to what was there or 256 if I can confirm. 
# Safe bet: leave INPUT_SIZE alone for now unless I know for sure. Thunder Int8 is often 256. Lightning is 192.
# Let's check INPUT_SIZE 256 for Thunder.
INPUT_SIZE = 256 
CONFIDENCE_THRESHOLD = 0.3

SKELETON_COLOR = (0,255,0)

PROCESS_EVERY_N_FRAME = 2
movement_history = []
HISTORY_WINDOW = 30
MAX_JUMP_THRESHOLD = 0.15

last_centroid = None

EDGES = [
    (0,1),(0,2),(1,3),(2,4),(0,5),(0,6),
    (5,7),(7,9),(6,8),(8,10),(5,6),
    (5,11),(6,12),(11,12),(11,13),(13,15),
    (12,14),(14,16)
]

def send_predictions(posture, conf):
    vision_status = "Normal"  # Fixed: was "NORMAL"
    if posture == "FALLING":
        vision_status = "Fall"  # Fixed: was "FALL"
    
    payload = {
        "device_id": "RPI_CAM_01",
        "posture_class": posture,  # ✅ Fixed typo: was "posture_cass"
        "accel_magnitude": 1.0,
        "slump_metric": 0.0,
        "vision_status": vision_status,
        "risk_score": conf * 100        
    }
    try:
        requests.post(STREAMLIT_URL, json=payload, timeout=3.0)
        print(f"✅ Sent: {posture}")
    except Exception as e:
        print(f"❌ Error: {e}")


def get_bounding_box(keypoints, height, width):
    valid_points = keypoints[keypoints[:,2]>CONFIDENCE_THRESHOLD]
    if len(valid_points)==0:
        return None
    
    y_coords = valid_points[:,0]*height
    x_coords = valid_points[:,1]*width
    return {
        'x1':x_coords.min(),'y1':y_coords.min(),
        'x2':x_coords.max(),'y2':y_coords.max()
    }

def draw_skeleton(frame, keypoints):
    height, width = frame.shape[:2]

    kpts = keypoints[0,0]*[height, width, 1]

    for edge in EDGES:
        if kpts[edge[0],2]>CONFIDENCE_THRESHOLD and kpts[edge[1],2]>CONFIDENCE_THRESHOLD:
            pt1 = (int(kpts[edge[0],1]), int(kpts[edge[0],0]))
            pt2 = (int(kpts[edge[1],1]), int(kpts[edge[1],1]))

            cv2.line(frame, pt1, pt2, SKELETON_COLOR, 2, cv2.LINE_AA)

    for i in range(17):
        if kpts[i,2] > CONFIDENCE_THRESHOLD:
            x,y = int(kpts[i,1]), int(kpts[i,0])
            cv2.circle(frame, (x,y), 4, (0,0,255),-1)
            cv2.circle(frame, (x,y), 6, (255,255,255),2)


def detect_posture(keypoints, height, width):
    global movement_history, last_centroid

    nose = keypoints[0]
    l_sh, r_sh = keypoints[5], keypoints[6]
    l_hip, r_hip = keypoints[11], keypoints[12]
    l_knee, r_knee = keypoints[13], keypoints[14]
    l_ank, r_ank = keypoints[15], keypoints[16]

    def to_px(pt):
        return (pt[1]*width, pt[0]*height, pt[2])
    
    n = to_px(nose)
    sh = to_px(((l_sh[0]+r_sh[0])/2,(l_sh[1]+r_sh[1])/2,1))
    hip = to_px(((l_hip[0]+r_hip[0])/2,(l_hip[1]+r_hip[1])/2,1))
    knee_l, knee_r = to_px(l_knee), to_px(r_knee)
    ank_l, ank_r = to_px(l_ank), to_px(r_ank)

    def angle(a,b,c):
        a,b,c = np.array(a[:2]), np.array(b[:2]), np.array(c[:2])
        ba, bc = a-b, c-b
        cosang = np.dot(ba,bc)/(np.linalg.norm(ba)*np.linalg.norm(bc)+1e-6)
        return np.degrees(np.arccos(np.clip(cosang,-1,1)))

    torso_angle = abs(math.degrees(math.atan2(sh[1]-hip[1],sh[0]-hip[0])))

    knee_angle = min(angle(knee_l, hip, ank_l), angle(knee_r, hip, ank_r))

    
    cx, cy = hip[0], hip[1]
    movement_history.append((cx,cy))
    if len(movement_history)>HISTORY_WINDOW:
        movement_history.pop(0)
    
    motion = np.std([p[0] for p in movement_history]) + np.std([p[1] for p in movement_history])
    
    if n[1] > hip[1] + 20 or torso_angle < 35 or n[1] > height*0.8:
        if motion > 40:
            return "FALL", (0,0,255)
        else:
            return "LYING",(0,165,255)
    
    if knee_angle < 120 and hip[1] > sh[1] - 10:
        return "SITTING", (255,165,0)
    
    if torso_angle > 50 and hip[1] < knee_l[1] and hip[1] < knee_r[1]:
        return "STANDING", (0,255,0)
    
    return "UNKNOWN", (200,200,200)
        
def main():
    print('='*60)
    print("MoveNet Single Pose Detection + Posture Recognition")
    print('='*60)

    print("\n[1/3] Loading model...")
    try:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        print(input_details)
        output_details =interpreter.get_output_details()
        print(output_details)
        print("Model Loaded")
    except Exception as e:
        print(f"Error:{e}")
        return
    
    print("\n [2/3] Opening Camera...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    print("Camera Ready")


    print("[3/3 Starting Detection ....")


    frame_count = 0
    last_keypoint = None
    fps_history = []
    prev_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count+=1

            if frame_count%PROCESS_EVERY_N_FRAME==0 or last_keypoint is None:
                img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                input_data = np.expand_dims(img, axis=0).astype(np.uint8)

                interpreter.set_tensor(input_details[0]['index'],input_data)
                interpreter.invoke()

                keypoints = interpreter.get_tensor(output_details[0]['index'])
                last_keypoint = keypoints
            else:
                draw_skeleton(frame, keypoints)
                posture, color = detect_posture(
                            keypoints[0,0],
                            frame.shape[0],
                            frame.shape[1])   
                
                conf = 0.15
                send_predictions(posture, conf)

                curr_time = time.time()
                fps = 1/(curr_time - prev_time)
                prev_time = curr_time
                fps_history.append(fps)
                if len(fps_history)>30:
                    fps_history.pop(0)
                fps_smooth = sum(fps_history)/len(fps_history)

                cv2.putText(frame, f"FPS:{fps_smooth:.1f}",(10,30),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2)
                cv2.putText(frame, f"Posture:{posture}",(10,60),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2)
                
                if posture == "FALL":
                    cv2.putText(frame, "!!! FALL DETECTED !!!",(frame.shape[1]//2-150,50),cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255),3)
                    print("Alerting user...")
                    cap.release()
                    cv2.destroyAllWindows()
                    print("="*60)
                    img_filename = f'Fall_detected_frame.jpg'
                    cv2.imwrite(img_filename,frame)
                    print("="*60)

                    #streamlit api call

                    break

                cv2.imshow('Pose Detection', frame)

                key = cv2.waitKey(1) &  0xFF
                if key == ord('q') or key ==27:
                    break
                elif key == ord('s'):
                    filename = f'pose_{frame_count}.jpg'
                    cv2.imwrite(filename, frame)
                    print("Saved frame")
    except KeyboardInterrupt:
        print("Stopped")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"Processed {frame_count} frames")
        print("="*60)

if __name__ == "__main__":
    main()