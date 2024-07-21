import cv2
from blue_detector import BlueDetector
from kalmanfilter import KalmanFilter

# Open the default camera (usually webcam)
cap = cv2.VideoCapture(0)

# Load detector
od = BlueDetector()

# Load Kalman filter to predict the trajectory
kf = KalmanFilter()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    blue_bbox = od.detect(frame)
    
    if blue_bbox is not None:
        x, y, x2, y2 = blue_bbox
        cx = int((x + x2) / 2)
        cy = int((y + y2) / 2)

        predicted = kf.predict(cx, cy)
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 4)  # Draw a green rectangle around the detected object
        cv2.circle(frame, (predicted[0], predicted[1]), 20, (255, 0, 0), 4)  # Draw a blue circle at the predicted position

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)  # Reduced wait time for real-time display
    if key == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
