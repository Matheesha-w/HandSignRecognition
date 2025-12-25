import pickle
import cv2
import mediapipe as mp
import numpy as np


# Load trained model

model = pickle.load(open('model.p', 'rb'))['model']


# MediaPipe setup

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# Webcam

cap = cv2.VideoCapture(0)

CONFIDENCE_THRESHOLD = 0.75

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    H, W, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

   
    # NO HAND DETECTED (BONUS)
    
    if not results.multi_hand_landmarks:
        cv2.putText(
            frame,
            "No hand detected",
            (40, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

   
    # HAND DETECTED
  
    else:
        hand = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        x_, y_ = [], []
        data_aux = []

        for lm in hand.landmark:
            x_.append(lm.x)
            y_.append(lm.y)

        for lm in hand.landmark:
            data_aux.append(lm.x - min(x_))
            data_aux.append(lm.y - min(y_))

      
        # Prediction with confidence
        
        proba = model.predict_proba([np.array(data_aux)])
        max_confidence = np.max(proba)
        predicted_label = model.classes_[np.argmax(proba)]

        if max_confidence < CONFIDENCE_THRESHOLD:
            label = "UNKNOWN"
            color = (0, 0, 255)  # Red
        else:
            label = f"{predicted_label} ({max_confidence:.2f})"
            color = (0, 255, 0)  # Green

        
        # Bounding box
        
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) + 10
        y2 = int(max(y_) * H) + 10

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )

    cv2.imshow("Hand Sign Recognition", frame)

    # ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
