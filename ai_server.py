import cv2
import mediapipe as mp
import numpy as np
import base64
import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MEDIAPIPE CONFIG ---
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'pose_landmarker.task')

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE, 
    num_poses=1
)
landmarker = PoseLandmarker.create_from_options(options)

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

SKELETON_FULL = [
    (11, 12),  # shoulder connection
    (11, 13), (13, 15),  # left arm
    (12, 14), (14, 16),  # right arm
    (11, 23), (12, 24),  # shoulders to hips
    (23, 24),  # hips connection
    (23, 25), (25, 27),  # left leg (hip-knee-ankle)
    (24, 26), (26, 28),  # right leg (hip-knee-ankle)
    (27, 29), (28, 30),  # heels
    (29, 31), (30, 32),  # toes
    (27, 31), (28, 32)   # foot outline
]

# HEALTH CHECK
@app.get("/")
def health_check():
    return {"status": "ok", "message": "System AI Active"}

# PUSHUPS

@app.websocket("/ws/pushups")
async def pushup_tracker(websocket: WebSocket):
    await websocket.accept()
    reps = 0
    stage = "up"

    try:
        while True:
            data = await websocket.receive_text()
            img_data = base64.b64decode(data.split(',')[1])
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            h, w, _ = frame.shape
            
            # Convert from OpenCV BGR to RGB for MediaPipe model accuracy
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = landmarker.detect(mp_image)

            if result and result.pose_landmarks:
                lm = result.pose_landmarks[0]
                
                # Calculate angles in pixel space to prevent aspect ratio distortion
                l_arm = calculate_angle(
                    [lm[11].x * w, lm[11].y * h], 
                    [lm[13].x * w, lm[13].y * h], 
                    [lm[15].x * w, lm[15].y * h]
                )
                r_arm = calculate_angle(
                    [lm[12].x * w, lm[12].y * h], 
                    [lm[14].x * w, lm[14].y * h], 
                    [lm[16].x * w, lm[16].y * h]
                )
                
                # Check visibility scores across the whole body side to determine side-view orientation
                l_vis = (lm[11].visibility + lm[13].visibility + lm[15].visibility + lm[23].visibility + lm[25].visibility + lm[27].visibility) / 6
                r_vis = (lm[12].visibility + lm[14].visibility + lm[16].visibility + lm[24].visibility + lm[26].visibility + lm[28].visibility) / 6
                
                is_side_view = abs(l_vis - r_vis) > 0.15
                
                if is_side_view:
                    if l_vis > r_vis:
                        arm_angle = l_arm
                        back_hip_angle = calculate_angle(
                            [lm[11].x * w, lm[11].y * h], 
                            [lm[23].x * w, lm[23].y * h], 
                            [lm[25].x * w, lm[25].y * h]
                        )
                        knee_angle = calculate_angle(
                            [lm[23].x * w, lm[23].y * h], 
                            [lm[25].x * w, lm[25].y * h], 
                            [lm[27].x * w, lm[27].y * h]
                        )
                    else:
                        arm_angle = r_arm
                        back_hip_angle = calculate_angle(
                            [lm[12].x * w, lm[12].y * h], 
                            [lm[24].x * w, lm[24].y * h], 
                            [lm[26].x * w, lm[26].y * h]
                        )
                        knee_angle = calculate_angle(
                            [lm[24].x * w, lm[24].y * h], 
                            [lm[26].x * w, lm[26].y * h], 
                            [lm[28].x * w, lm[28].y * h]
                        )
                    is_valid_form = back_hip_angle > 140 and knee_angle > 145
                else:
                    # Front/diagonal view: require both arms to bend, and verify both leg alignments
                    arm_angle = min(l_arm, r_arm)
                    l_back_hip = calculate_angle(
                        [lm[11].x * w, lm[11].y * h], 
                        [lm[23].x * w, lm[23].y * h], 
                        [lm[25].x * w, lm[25].y * h]
                    )
                    r_back_hip = calculate_angle(
                        [lm[12].x * w, lm[12].y * h], 
                        [lm[24].x * w, lm[24].y * h], 
                        [lm[26].x * w, lm[26].y * h]
                    )
                    l_knee = calculate_angle(
                        [lm[23].x * w, lm[23].y * h], 
                        [lm[25].x * w, lm[25].y * h], 
                        [lm[27].x * w, lm[27].y * h]
                    )
                    r_knee = calculate_angle(
                        [lm[24].x * w, lm[24].y * h], 
                        [lm[26].x * w, lm[26].y * h], 
                        [lm[28].x * w, lm[28].y * h]
                    )
                    is_valid_form = l_back_hip > 140 and r_back_hip > 140 and l_knee > 145 and r_knee > 145
                    back_hip_angle = (l_back_hip + r_back_hip) / 2
                    knee_angle = (l_knee + r_knee) / 2

                # Track posture/rep states
                if arm_angle > 150: 
                    stage = "up"
                if stage == "up" and arm_angle < 95 and is_valid_form:
                    stage = "down"
                    reps += 1

                # --- DRAWING PUSHUP SKELETON ---
                arm_color = (0, 255, 255) if arm_angle < 100 else (255, 255, 255)
                back_color = (0, 255, 0) if is_valid_form else (0, 0, 255)

                for s, e in SKELETON_FULL:
                    if (s in [11, 13, 12, 14] and e in [13, 15, 14, 16]):
                        color = arm_color
                    elif (s in [11, 12, 23, 24, 25, 26] and e in [23, 24, 25, 26, 27, 28]):
                        color = back_color
                    else:
                        color = (255, 255, 255)
                    cv2.line(frame, (int(lm[s].x*w), int(lm[s].y*h)), (int(lm[e].x*w), int(lm[e].y*h)), color, 3)
                
                for i in range(11, 33):
                    cv2.circle(frame, (int(lm[i].x*w), int(lm[i].y*h)), 5, (255, 0, 255), -1)
                
                # Visual warnings on HUD for poor posture
                if not is_valid_form:
                    if (is_side_view and back_hip_angle <= 140) or (not is_side_view and (l_back_hip <= 140 or r_back_hip <= 140)):
                        cv2.putText(frame, "FIX BACK ALIGNMENT!", (30, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    elif (is_side_view and knee_angle <= 145) or (not is_side_view and (l_knee <= 145 or r_knee <= 145)):
                        cv2.putText(frame, "STRAIGHTEN LEGS!", (30, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # ENCODE AND SEND BACK TO REACT
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
            frame_b64 = base64.b64encode(buffer).decode('utf-8')
            await websocket.send_json({"reps": reps, "frame": frame_b64})

    except WebSocketDisconnect:
        print("[SYSTEM]: Hunter disconnected.")


# SQUATS

@app.websocket("/ws/squats")
async def squat_tracker(websocket: WebSocket):
    await websocket.accept()
    reps = 0
    stage = "up"

    try:
        while True:
            data = await websocket.receive_text()
            img_data = base64.b64decode(data.split(',')[1])
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            h, w, _ = frame.shape
            
            # Convert from OpenCV BGR to RGB for MediaPipe model accuracy
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = landmarker.detect(mp_image)

            if result and result.pose_landmarks:
                lm = result.pose_landmarks[0]
                
                # Calculate angles in pixel space to prevent aspect ratio distortion
                l_angle = calculate_angle(
                    [lm[23].x * w, lm[23].y * h], 
                    [lm[25].x * w, lm[25].y * h], 
                    [lm[27].x * w, lm[27].y * h]
                )
                r_angle = calculate_angle(
                    [lm[24].x * w, lm[24].y * h], 
                    [lm[26].x * w, lm[26].y * h], 
                    [lm[28].x * w, lm[28].y * h]
                )
                
                l_back_hip = calculate_angle(
                    [lm[11].x * w, lm[11].y * h], 
                    [lm[23].x * w, lm[23].y * h], 
                    [lm[25].x * w, lm[25].y * h]
                )
                r_back_hip = calculate_angle(
                    [lm[12].x * w, lm[12].y * h], 
                    [lm[24].x * w, lm[24].y * h], 
                    [lm[26].x * w, lm[26].y * h]
                )
                
                l_vis = (lm[11].visibility + lm[23].visibility + lm[25].visibility + lm[27].visibility) / 4
                r_vis = (lm[12].visibility + lm[24].visibility + lm[26].visibility + lm[28].visibility) / 4
                
                is_side_view = abs(l_vis - r_vis) > 0.15
                
                if is_side_view:
                    if l_vis > r_vis:
                        knee_angle = l_angle
                        back_hip_angle = l_back_hip
                    else:
                        knee_angle = r_angle
                        back_hip_angle = r_back_hip
                    
                    is_symmetric = True  # Not applicable in side profile
                    is_valid_form = back_hip_angle > 65
                else:
                    knee_angle = (l_angle + r_angle) / 2
                    back_hip_angle = (l_back_hip + r_back_hip) / 2
                    is_symmetric = abs(l_angle - r_angle) < 25
                    is_valid_form = is_symmetric and l_back_hip > 65 and r_back_hip > 65

                if is_side_view:
                    if knee_angle > 160: 
                        stage = "up"
                    if stage == "up" and knee_angle < 100 and is_valid_form:
                        stage = "down"
                        reps += 1
                else:
                    if l_angle > 160 and r_angle > 160: 
                        stage = "up"
                    if stage == "up" and l_angle < 105 and r_angle < 105 and is_valid_form:
                        stage = "down"
                        reps += 1

                # --- DRAWING SQUAT SKELETON ---
                l_color = (0, 255, 255) if l_angle < 105 else (255, 255, 255)
                r_color = (0, 255, 255) if r_angle < 105 else (255, 255, 255)
                back_color = (0, 255, 0) if is_valid_form else (0, 0, 255)

                for s, e in SKELETON_FULL:
                    color = (255, 255, 255)
                    if s in [23, 25] and e in [25, 27]:
                        color = l_color if is_valid_form else (0, 0, 255)
                    elif s in [24, 26] and e in [26, 28]:
                        color = r_color if is_valid_form else (0, 0, 255)
                    elif s in [11, 12, 23] and e in [23, 24, 24]:
                        color = back_color
                    cv2.line(frame, (int(lm[s].x*w), int(lm[s].y*h)), (int(lm[e].x*w), int(lm[e].y*h)), color, 3)

                for i in range(11, 33):
                    cv2.circle(frame, (int(lm[i].x*w), int(lm[i].y*h)), 5, (255, 0, 255), -1)

                # Visual warning on HUD for poor posture
                if not is_valid_form:
                    if not is_symmetric and not is_side_view:
                        cv2.putText(frame, "IMBALANCED FORM!", (30, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    elif back_hip_angle <= 65:
                        cv2.putText(frame, "KEEP CHEST UP / FIX BACK!", (30, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # ENCODE AND SEND BACK TO REACT
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
            frame_b64 = base64.b64encode(buffer).decode('utf-8')
            await websocket.send_json({"reps": reps, "frame": frame_b64})

    except WebSocketDisconnect:
        print("[SYSTEM]: Hunter disconnected.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)