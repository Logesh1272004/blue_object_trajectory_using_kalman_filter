import cv2
import mediapipe as mp

# Open the default camera (usually webcam)
cap = cv2.VideoCapture(0)

# Load the hand tracking module from MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image with the hand tracking model
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                # Extract the coordinates of each landmark (scaled to the image dimensions)
                height, width, _ = frame.shape
                cx, cy = int(landmark.x * width), int(landmark.y * height)
                
                # You can use the cx, cy coordinates to perform your tracking logic
                # (e.g., use Kalman filter for tracking as in your original code)

                # Draw a circle at each landmark
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)  # Reduced wait time for real-time display
    if key == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
