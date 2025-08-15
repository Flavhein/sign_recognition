import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque

# === Load the trained model ===
model = tf.keras.models.load_model("sign_language_model.keras")

# === Constants ===
SEQ_LEN = 60
INPUT_DIM = 63  # 1 hand × 21 keypoints × 3 coords
BUFFER = deque(maxlen=SEQ_LEN)
CONFIDENCE_THRESHOLD = 0.5

EMOJI_PATH = ""  

# === MediaPipe Setup ===
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1)
mp_draw = mp.solutions.drawing_utils

# === Labels (must match training) ===
labels = ['hello','thanks','no','smart','need','find','fuck','me','home','yes']  # 10 classes

# === Helpers ===
def extract_relative_hand_keypoints(hand_results, face_results):
    face_center = np.array([0.0, 0.0, 0.0])
    if face_results.multi_face_landmarks:
        landmarks = face_results.multi_face_landmarks[0].landmark
        points = [landmarks[1], landmarks[152]]
        face_center = np.mean([[p.x, p.y, p.z] for p in points], axis=0)

    if hand_results.multi_hand_landmarks:
        hand = hand_results.multi_hand_landmarks[0]
        keypoints = []
        for lm in hand.landmark:
            keypoints.extend([
                lm.x - face_center[0],
                lm.y - face_center[1],
                lm.z - face_center[2]
            ])
        return np.array(keypoints)
    else:
        return np.zeros(INPUT_DIM)

def load_emoji(path, size=28):
    if not path:
        return None
    icon = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if icon is None:
        return None
    h, w = icon.shape[:2]
    scale = size / max(h, w)
    icon = cv2.resize(icon, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    return icon

def overlay_rgba(bg, fg, x, y):
    """Overlay RGBA fg on BGR bg at (x,y)."""
    fh, fw = fg.shape[:2]
    if x >= bg.shape[1] or y >= bg.shape[0]:
        return
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + fw, bg.shape[1]), min(y + fh, bg.shape[0])
    fg_x1, fg_y1 = x1 - x, y1 - y
    fg_crop = fg[fg_y1:fg_y1 + (y2 - y1), fg_x1:fg_x1 + (x2 - x1)]
    if fg_crop.shape[2] == 4:
        alpha = fg_crop[:, :, 3] / 255.0
        for c in range(3):
            bg[y1:y2, x1:x2, c] = (1 - alpha) * bg[y1:y2, x1:x2, c] + alpha * fg_crop[:, :, c]
    else:
        bg[y1:y2, x1:x2] = fg_crop[:, :, :3]

def draw_status(image, state, progress, emoji=None):
    """
    state: 'capturing', 'predicting', 'idle'
    progress: 0.0..1.0
    """
    h, w = image.shape[:2]
    pad = 12
    size = 28  # icon size if drawn
    x, y = pad, pad

    # Choose color per state
    colors = {
        'capturing': (0, 200, 0),    # green
        'predicting': (200, 100, 0), # blue-ish/orange? let's use blue instead:
    }
    colors['predicting'] = (200, 0, 0)  # red
    colors['idle'] = (128, 128, 128)    # gray

    # Try emoji first if provided and we are capturing; otherwise draw a circle
    if state == 'capturing' and emoji is not None:
        overlay_rgba(image, emoji, x, y)
        icon_right = x + emoji.shape[1]
        icon_center_y = y + emoji.shape[0] // 2
    else:
        color = colors[state]
        cv2.circle(image, (x + size//2, y + size//2), size//2, color, -1)
        icon_right = x + size
        icon_center_y = y + size//2

    # Progress bar (under the icon)
    bar_w, bar_h = 160, 10
    bar_x, bar_y = icon_right + 10, icon_center_y - bar_h // 2
    cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), 1)
    filled = int(bar_w * np.clip(progress, 0, 1))
    fill_color = (0, 200, 0) if state == 'capturing' else (200, 0, 0) if state == 'predicting' else (128, 128, 128)
    cv2.rectangle(image, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h), fill_color, -1)

    # Label
    label = f"{state.upper()}  {int(progress*100):>3d}%"
    cv2.putText(image, label, (bar_x, bar_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)

# === Load emoji once ===
EMOJI_ICON = load_emoji(EMOJI_PATH, size=28)

# === Webcam capture ===
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(rgb)
    face_results = face_mesh.process(rgb)

    # Draw hand landmarks
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Draw only the center face point (between eyes)
    if face_results.multi_face_landmarks:
        face = face_results.multi_face_landmarks[0]
        points = [face.landmark[1], face.landmark[152]]
        cx = int(np.mean([p.x for p in points]) * image.shape[1])
        cy = int(np.mean([p.y for p in points]) * image.shape[0])
        cv2.circle(image, (cx, cy), 5, (0, 255, 255), -1)

    # Get relative keypoints
    keypoints = extract_relative_hand_keypoints(hand_results, face_results)

    # Decide if we are "capturing" this frame (only when a hand is detected)
    hand_detected = hand_results.multi_hand_landmarks is not None
    if hand_detected:
        BUFFER.append(keypoints)

    progress = len(BUFFER) / float(SEQ_LEN)

    # Default status: idle (no hand)
    status = 'capturing' if hand_detected and len(BUFFER) < SEQ_LEN else 'predicting' if len(BUFFER) == SEQ_LEN else 'idle'
    draw_status(image, status, progress, emoji=EMOJI_ICON)

    if len(BUFFER) == SEQ_LEN:
        X = np.array(BUFFER).reshape(1, SEQ_LEN, INPUT_DIM)
        prediction = model.predict(X, verbose=0)[0]
        class_id = np.argmax(prediction)
        confidence = prediction[class_id]

        if confidence > CONFIDENCE_THRESHOLD:
            gesture = labels[class_id]
            cv2.putText(image, f"{gesture} ({confidence:.2f})", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            print(f"Prediction: {gesture} ({confidence:.2f})")
        else:
            cv2.putText(image, f"Low confidence ({confidence:.2f})", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2, cv2.LINE_AA)

        BUFFER.clear()  # reset for the next capture cycle

    cv2.imshow("Sign Language Recognition", image)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

