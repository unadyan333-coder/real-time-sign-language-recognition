import cv2
import mediapipe as mp
import csv
import os
import time  

DATA_FILE = "hand_gestures_data.csv"
CURRENT_LABEL = "HELLO"
DELAY_BETWEEN_SAMPLES = 0.5 

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)


if not os.path.exists(DATA_FILE):
    header = []
    for i in range(21):
        header.extend([f"x{i}", f"y{i}", f"z{i}"])
    header.append("label")

    with open(DATA_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

print("=" * 40)
print(f"Collecting label : {CURRENT_LABEL}")
print("Press S to START automatic collection")
print("Press P to PAUSE automatic collection")
print("Press Q to quit")
print("=" * 40)



cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera.")
    exit()

is_collecting = False
last_sample_time = 0 
saved_count = 0        

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    row = []

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
        for lm in hand.landmark:
            row.extend([lm.x, lm.y, lm.z])

 
   
    current_time = time.time()
    
    if is_collecting:
      
      if current_time - last_sample_time >= DELAY_BETWEEN_SAMPLES:
            if len(row) == 63:
                row.append(CURRENT_LABEL)
                with open(DATA_FILE, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
                
                saved_count += 1
                last_sample_time = current_time
                print(f"Saved samples: {saved_count}")
            else:
              

                cv2.putText(frame, "WARNING: No hand detected!", (10, 140), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


    status_text = "COLLECTING..." if is_collecting else "PAUSED"
    status_color = (0, 255, 0) if is_collecting else (0, 165, 255)
    
    cv2.putText(frame, f"Label: {CURRENT_LABEL} ({status_text})", (10, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)
    
    cv2.putText(frame, f"Saved in this session: {saved_count}", (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    cv2.putText(frame, "S = Start    P = Pause    Q = Quit", (10, 105), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Hand Data Collection", frame)


    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        is_collecting = True
        last_sample_time = time.time()
        print("Automatic collection STARTED.")

    elif key == ord("p"):
        is_collecting = False
        print("Automatic collection PAUSED.")

    elif key == ord("q"):
        break

cap.release()
hands.close()
cv2.destroyAllWindows()
print(f"Done! Total samples saved this run: {saved_count}")
