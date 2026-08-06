import cv2
import mediapipe as mp
from joblib import load
import os
from joblib import load

model = load("gesture_model.pkl")

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera.")
    exit()

prediction = ""

photo = None

if os.path.exists("photo.jpg.jpg"):
    photo = cv2.imread("photo.jpg.jpg")
    photo = cv2.resize(photo, (300, 300))
else:
    print("photo.jpg not found")

cv2.namedWindow("Sign Language Recognition", cv2.WINDOW_NORMAL)

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        hand = results.multi_hand_landmarks[0]

        mp_draw.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS
        )

        features = []

        for lm in hand.landmark:
            features.extend([lm.x, lm.y, lm.z])

        if len(features) == 63:
            prediction = model.predict([features])[0]

    if prediction == "PHOTO" and photo is not None:

        ph, pw = photo.shape[:2]

        x = (frame.shape[1] - pw) // 2
        y = 20

        frame[y:y + ph, x:x + pw] = photo

    h, w = frame.shape[:2]

    bar_height = 80

    cv2.rectangle(
        frame,
        (0, h - bar_height),
        (w, h),
        (25, 25, 25),
        -1
    )

    text = prediction

    font = cv2.FONT_HERSHEY_SIMPLEX

    scale = 1

    thickness = 2

    (text_w, text_h), _ = cv2.getTextSize(
        text,
        font,
        scale,
        thickness
    )

    x = (w - text_w) // 2

    y = h - 25

    cv2.putText(
        frame,
        text,
        (x, y),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )

    cv2.imshow("Sign Language Recognition", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
hands.close()
cv2.destroyAllWindows()
