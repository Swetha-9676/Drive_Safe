import cv2
import time
import numpy as np
import mediapipe as mp
import pygame

# ==================== AUDIO ====================
pygame.mixer.init()
channel = pygame.mixer.Channel(0)

SOUND_PHASE1 = pygame.mixer.Sound("audio/phase1_warning.wav")
SOUND_PHASE2 = pygame.mixer.Sound("audio/phase2_warning.wav")
SOUND_PHASE3 = pygame.mixer.Sound("audio/phase3_danger.wav")

PHASE_AUDIO = {
    1: SOUND_PHASE1,
    2: SOUND_PHASE2,
    3: SOUND_PHASE3
}

# ==================== STATE ====================
current_phase = 0
alarm_locked = False

# ==================== COUNT ====================
drowsy_count = 0
MAX_DROWSY_COUNT = 3
drowsy_event_active = False

# ==================== MEDIAPIPE ====================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ==================== CAMERA ====================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# ==================== THRESHOLDS ====================
EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.60

PHASE1_TIME = 1.5
PHASE2_TIME = 3.0
PHASE3_TIME = 4.5

eye_start = None
yawn_start = None

# ==================== LANDMARKS ====================
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [13, 14, 78, 308]

# ==================== FUNCTIONS ====================
def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def compute_ear(eye, lm, w, h):
    p = [(lm[i].x * w, lm[i].y * h) for i in eye]
    return (euclidean(p[1], p[5]) + euclidean(p[2], p[4])) / \
           (2.0 * euclidean(p[0], p[3]))

def compute_mar(mouth, lm, w, h):
    p = [(lm[i].x * w, lm[i].y * h) for i in mouth]
    return euclidean(p[0], p[1]) / euclidean(p[2], p[3])

def play_phase(phase):
    channel.stop()
    if phase == 3:
        channel.play(PHASE_AUDIO[phase], loops=-1)  # 🔁 LOOP until 'S'
    else:
        channel.play(PHASE_AUDIO[phase])            # play once


# ==================== MAIN LOOP ====================
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    eyes_closed = False
    yawning = False

    if result.multi_face_landmarks:
        lm = result.multi_face_landmarks[0].landmark

        ear = (compute_ear(LEFT_EYE, lm, w, h) +
               compute_ear(RIGHT_EYE, lm, w, h)) / 2
        mar = compute_mar(MOUTH, lm, w, h)

        eyes_closed = ear < EAR_THRESHOLD
        yawning = mar > MAR_THRESHOLD

        for i in LEFT_EYE + RIGHT_EYE:
            cv2.circle(frame, (int(lm[i].x*w), int(lm[i].y*h)), 2, (0,255,0), -1)
        for i in MOUTH:
            cv2.circle(frame, (int(lm[i].x*w), int(lm[i].y*h)), 2, (0,0,255), -1)

    # ==================== TIMERS ====================
    if eyes_closed:
        eye_start = eye_start or time.time()
        eye_time = time.time() - eye_start
    else:
        eye_start = None
        eye_time = 0

    if yawning:
        yawn_start = yawn_start or time.time()
        yawn_time = time.time() - yawn_start
    else:
        yawn_start = None
        yawn_time = 0

    max_drowsy_time = max(eye_time, yawn_time)

    # ==================== COUNT ====================
    if max_drowsy_time > PHASE1_TIME:
        if not drowsy_event_active:
            drowsy_event_active = True
            if drowsy_count < MAX_DROWSY_COUNT:
                drowsy_count += 1
    else:
        drowsy_event_active = False

    # ==================== PHASE LOGIC ====================
    if drowsy_count >= MAX_DROWSY_COUNT:
        alarm_locked = True
        if current_phase != 3:
            current_phase = 3
            play_phase(3)

    elif max_drowsy_time > PHASE1_TIME:
        if current_phase == 0:
            current_phase = 1
            play_phase(1)
        elif current_phase == 1 and not channel.get_busy():
            current_phase = 2
            play_phase(2)
        elif current_phase == 2 and not channel.get_busy():
            current_phase = 3
            play_phase(3)

    else:
        if not alarm_locked:
            current_phase = 0
            channel.stop()

    # ==================== DISPLAY ====================
    cv2.putText(frame, f"Eye Closed: {eye_time:.1f}s", (30,90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
    cv2.putText(frame, f"Yawning: {yawn_time:.1f}s", (30,120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
    cv2.putText(frame, f"Phase: {current_phase}", (30,150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.putText(frame, f"Drowsy Count: {drowsy_count}/3", (30,180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255), 2)

    if alarm_locked:
        cv2.putText(frame, "CRITICAL ALERT! Press 'S' to stop",
                    (30,220), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0,0,255), 2)

    cv2.imshow("Driver Drowsiness Detection", frame)

    # ==================== KEYS ====================
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        channel.stop()
        current_phase = 0
        alarm_locked = False
        drowsy_count = 0
        drowsy_event_active = False
        eye_start = None
        yawn_start = None
        print("Alarm reset")

    if key == ord('q'):
        break

# ==================== CLEANUP ====================
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()



