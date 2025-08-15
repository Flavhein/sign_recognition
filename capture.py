import cv2
import mediapipe as mp
import numpy as np
import time

mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh

hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)
face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1)
drawing = mp.solutions.drawing_utils

BUFFER = []
DURATION = 2  # seconds
FPS = 30
SEQ_LEN = DURATION * FPS

cap = cv2.VideoCapture(0)
print("Appuyez sur ESPACE pour capturer...")

recording = False
start_time = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    results_hands = hands.process(image)
    results_face = face_mesh.process(image)

    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Dessiner la main et le visage
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            drawing.draw_landmarks(image_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    if results_face.multi_face_landmarks:
        face_landmarks = results_face.multi_face_landmarks[0]
        # Use points 1 and 152 to calculate center
        points = [face_landmarks.landmark[i] for i in [1, 152]]
        cx = np.mean([p.x for p in points])
        cy = np.mean([p.y for p in points])
        cz = np.mean([p.z for p in points])
        face_center = np.array([cx, cy, cz])

        # Draw only the center point between the eyes
        h, w, _ = image_bgr.shape
        center_x = int(cx * w)
        center_y = int(cy * h)
        cv2.circle(image_bgr, (center_x, center_y), 5, (0, 255, 255), -1)  # Yellow dot
    else:
        face_center = np.array([0, 0, 0])


    # START capture on SPACEBAR
    key = cv2.waitKey(1)
    if key == 32 and not recording:
        print("Capture démarrée...")
        recording = True
        start_time = time.time()
        BUFFER = []

    # CAPTURE FRAME DATA
    if recording:
        elapsed = time.time() - start_time

        # Get face center
        if results_face.multi_face_landmarks:
            face_landmarks = results_face.multi_face_landmarks[0]
            # Utiliser les points 1 et 152 = entre les yeux approximativement
            points = [face_landmarks.landmark[i] for i in [1, 152]]
            cx = np.mean([p.x for p in points])
            cy = np.mean([p.y for p in points])
            cz = np.mean([p.z for p in points])
            face_center = np.array([cx, cy, cz])
        else:
            face_center = np.array([0, 0, 0])

        frame_data = []

        # Extract hand keypoints
        if results_hands.multi_hand_landmarks:
            hand_landmarks = results_hands.multi_hand_landmarks[0]
            for lm in hand_landmarks.landmark:
                rel_x = lm.x - face_center[0]
                rel_y = lm.y - face_center[1]
                rel_z = lm.z - face_center[2]
                frame_data.extend([rel_x, rel_y, rel_z])
        else:
            # pas de main → zéros
            frame_data.extend([0] * 21 * 3)  # pour 1 mains = 21 points

        BUFFER.append(frame_data)

        if elapsed >= DURATION:
            print("Capture terminée. Sauvegarde...")
            recording = False

            # ENREGISTRER en .npy
            filename = input("Nom du fichier (sans extension) : ")
            np.save(f"{filename}.npy", np.array(BUFFER))
            print(f"Fichier {filename}.npy enregistré.")

    cv2.imshow("Capture", image_bgr)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
