import cv2
import mediapipe as mp
import numpy as np
import time
import requests
import threading
import sys  # We need sys to read the hidden token from Node.js

# --- SYSTEM CONFIG ---
MODEL_PATH = 'pose_landmarker.task'
BASE_URL = "http://localhost:5000/api"
UPDATE_URL = f"{BASE_URL}/hunter/quest/update"
PROFILE_URL = f"{BASE_URL}/hunter/profile"
BATCH_LIMIT = 5

# --- AUTOMATIC AUTHENTICATION VIA NODE.JS ---
def get_token_from_node():
    # If Node.js triggered this script, it passed the token as the second argument
    if len(sys.argv) > 1:
        token = sys.argv[1]
        print("System Token received from the Web Server.")
        
        # Verify the token is real by pinging the backend
        try:
            headers = {'x-auth-token': token}
            response = requests.get(PROFILE_URL, headers=headers, timeout=3)
            if response.status_code == 200:
                hunter = response.json()
                print("\n" + "*"*40)
                print(f" IDENTITY VERIFIED: Welcome, {hunter['username']}.")
                print(f" CURRENT STATUS: Level {hunter['level']} [{hunter['rank']}]")
                print("*"*40 + "\n")
                return token
            else:
                print("\n[!] ACCESS DENIED: Token is invalid or expired.")
                sys.exit()
        except:
            print("\n[!] CRITICAL ERROR: Cannot connect to Backend Database.")
            sys.exit()
    else:
        print("\n[!] SYSTEM ERROR: No token provided. You must start the camera from the Web Dashboard.")
        sys.exit()

# Instantly grab the token and verify
TOKEN = get_token_from_node()

# ==========================================
# (Keep all your existing MediaPipe / OpenCV code below this line)
# ==========================================


# --- MEDIAPIPE CONFIG (TASKS API) ---
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_poses=1
)

SKELETON = [(11,12), (11,23), (12,24), (23,24), (11,13), (13,15), (12,14), (14,16), (23,25), (25,27), (24,26), (26,28)]

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle
def sync_to_system():
    try:
        headers = {'x-auth-token': TOKEN, 'Content-Type': 'application/json'}
        payload = {'completedSquats': 1} # Backend expects this!
        
        response = requests.post(UPDATE_URL, json=payload, headers=headers, timeout=2)
        if response.status_code == 200:
            print(f"--- [SYSTEM] 1 Squat Synced! Server: {response.json().get('msg')} ---")
        else:
            print(f"--- [SYSTEM ERROR] Sync Rejected: {response.text} ---")
    except Exception as e:
        print(f"--- [SYSTEM ERROR] Sync Failed: {e} ---")

with PoseLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    counter, stage, start_time = 0, "up", time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        h, w, _ = frame.shape
        frame = cv2.flip(frame, 1)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        timestamp = int((time.time() - start_time) * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp)

        if result and result.pose_landmarks:
            lm = result.pose_landmarks[0]
            
            # --- CALCULATE ANGLES ---
            l_angle = calculate_angle([lm[23].x, lm[23].y], [lm[25].x, lm[25].y], [lm[27].x, lm[27].y])
            r_angle = calculate_angle([lm[24].x, lm[24].y], [lm[26].x, lm[26].y], [lm[28].x, lm[28].y])
            is_symmetric = abs(l_angle - r_angle) < 25
            
            # Dynamic Colors: Red for incomplete/bad form, Cyan for goal reached
            l_color = (0, 255, 255) if l_angle < 105 else (0, 0, 255)
            r_color = (0, 255, 255) if r_angle < 105 else (0, 0, 255)
            if not is_symmetric: l_color = r_color = (0, 0, 255)

            # --- DRAW FULL SKELETON ---
            for s, e in SKELETON:
                # Assign specific colors to legs, white for others
                color = (255, 255, 255)
                if s in [23, 25]: color = l_color
                if s in [24, 26]: color = r_color
                cv2.line(frame, (int(lm[s].x*w), int(lm[s].y*h)), (int(lm[e].x*w), int(lm[e].y*h)), color, 2)

            # Draw Dots (Nodes)
            for i in range(11, 33):
                cv2.circle(frame, (int(lm[i].x*w), int(lm[i].y*h)), 5, (255, 0, 255), -1)

            # --- SQUAT LOGIC ---
           # --- SQUAT LOGIC ---
            if l_angle > 160 and r_angle > 160: stage = "up"
            if stage == "up" and l_angle < 100 and r_angle < 100 and is_symmetric:
                stage = "down"
                counter += 1
                # Trigger the sync safely in the background
                threading.Thread(target=sync_to_system, daemon=True).start()

            cv2.putText(frame, f"SQUATS: {counter}", (20, 50), 2, 1, (255, 255, 255), 2)
            if not is_symmetric: cv2.putText(frame, "IMBALANCED!", (20, 90), 2, 0.7, (0, 0, 255), 2)

        cv2.imshow('System: Squat Mission', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
cap.release()
cv2.destroyAllWindows()