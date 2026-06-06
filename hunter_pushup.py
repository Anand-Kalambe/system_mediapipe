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

SKELETON = [(11,12), (11,13), (13,15), (12,14), (14,16), (11,23), (12,24), (23,25), (24,26)]

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

def sync_to_system(count):
    try:
        # 1. Attach the token so the backend knows who you are!
        headers = {'x-auth-token': TOKEN, 'Content-Type': 'application/json'}
        # 2. Use the correct property name expected by the backend
        payload = {'completedPushups': count}
        
        # 3. Use UPDATE_URL, not API_URL
        response = requests.post(UPDATE_URL, json=payload, headers=headers, timeout=5)
        
        if response.status_code == 200:
            print(f"--- [SYSTEM] {count} Pushups Synced! Server: {response.json().get('msg')} ---")
        else:
            print(f"--- [SYSTEM ERROR] Sync Rejected: {response.text} ---")
    except Exception as e:
        print(f"--- [SYSTEM ERROR] Sync Failed: {e} ---")

with PoseLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    
    total_lifetime_reps = 0 
    local_batch_counter = 0  # Logic kept, UI removed
    stage = "up"
    start_time = time.time()

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
            
            # --- ANGLES ---
            l_arm = calculate_angle([lm[11].x, lm[11].y], [lm[13].x, lm[13].y], [lm[15].x, lm[15].y])
            r_arm = calculate_angle([lm[12].x, lm[12].y], [lm[14].x, lm[14].y], [lm[16].x, lm[16].y])
            back_angle = calculate_angle([lm[11].x, lm[11].y], [lm[23].x, lm[23].y], [lm[25].x, lm[25].y])
            
            # Form Colors
            arm_color = (0, 255, 255) if l_arm < 100 else (255, 255, 255)
            back_color = (0, 255, 0) if back_angle > 150 else (0, 0, 255)

            # --- DRAWING ---
            for s, e in SKELETON:
                color = back_color if (s in [11, 12] and e in [23, 24]) else (arm_color if s in [11,13,12,14] else (255,255,255))
                cv2.line(frame, (int(lm[s].x*w), int(lm[s].y*h)), (int(lm[e].x*w), int(lm[e].y*h)), color, 3)
            
            for i in range(11, 27):
                cv2.circle(frame, (int(lm[i].x*w), int(lm[i].y*h)), 5, (255, 0, 255), -1)

            # --- REFINED PUSH-UP LOGIC ---
            if l_arm > 155 and r_arm > 155:
                stage = "up"
            
            if stage == "up" and l_arm < 95 and r_arm < 95:
                if back_angle > 145:
                    stage = "down"
                    local_batch_counter += 1
                    total_lifetime_reps += 1
                    
                    # Batch Sync Trigger (Silent)
                    # Batch Sync Trigger (Silent)
                    if local_batch_counter >= BATCH_LIMIT:
                        threading.Thread(target=sync_to_system, args=(local_batch_counter,), daemon=True).start()
                        local_batch_counter = 0
                else:
                    cv2.putText(frame, "FIX BACK!", (w//2 - 50, h//2), 2, 1, (0, 0, 255), 2)

            # --- UI HUD (CLEAN) ---
            # Removed the rectangle and progress text as requested
            cv2.putText(frame, f"PUSHUPS: {total_lifetime_reps}", (20, 50), 2, 1.2, (0, 255, 255), 2)

        cv2.imshow('System: Push-up Mission', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()