import os
import cv2
import time

DATA_DIR = './data'
NUMBER_OF_CLASSES = 6
DATASET_SIZE = 100

os.makedirs(DATA_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)

for j in range(NUMBER_OF_CLASSES):
    class_dir = os.path.join(DATA_DIR, str(j))
    os.makedirs(class_dir, exist_ok=True)

    print(f'Collecting data for class {j}')
    print('Press Q when ready...')

    # Wait for user
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, 'Press Q to start capturing',
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Capture images
    for i in range(DATASET_SIZE):
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        cv2.imwrite(os.path.join(class_dir, f'{i}.jpg'), frame)
        cv2.imshow('frame', frame)
        cv2.waitKey(50)  # small delay

cap.release()
cv2.destroyAllWindows()
