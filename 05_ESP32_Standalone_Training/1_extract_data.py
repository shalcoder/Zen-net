
import cv2
import os

VIDEO_PATH = "../dataset/queda.mp4"
OUTPUT_FALL = "dataset/fall"
OUTPUT_NORMAL = "dataset/normal"

def extract_frames():
    cap = cv2.VideoCapture(VIDEO_PATH)
    count = 0
    print("Press 'f' to save as FALL, 'n' to save as NORMAL, 'SPACE' to skip. 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Resize to 96x96 (ESP32 Standard)
        # We use a small square resolution for MobileNet
        preview = cv2.resize(frame, (320, 240))
        model_input = cv2.resize(frame, (96, 96))
        
        cv2.imshow("Categorizer", preview)
        key = cv2.waitKey(0) & 0xFF # Wait indefinitely for key press
        
        if key == ord('f'):
            p = os.path.join(OUTPUT_FALL, f"frame_{count}.jpg")
            cv2.imwrite(p, model_input)
            print(f"Saved FALL: {p}")
            count += 1
        elif key == ord('n'):
            p = os.path.join(OUTPUT_NORMAL, f"frame_{count}.jpg")
            cv2.imwrite(p, model_input)
            print(f"Saved NORMAL: {p}")
            count += 1
        elif key == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    extract_frames()
