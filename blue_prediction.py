def blue():    
    import cv2
    from blue_detector import BlueDetector
    from kalmanfilter import KalmanFilter

    # Initialize video capture
    cap = cv2.VideoCapture("Logesh.mp4")

    # Load detector
    od = BlueDetector()

    # Load Kalman filter to predict the trajectory
    kf = KalmanFilter()
    i=[]
    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        if ret is False:
            break

        # Detect the blue object
        blue_bbox = od.detect(frame)
        x, y, x2, y2 = blue_bbox
        cx = int((x + x2) / 2)
        cy = int((y + y2) / 2)

        # Predict the next position using the Kalman filter
        predicted = kf.predict(cx, cy)

        # Draw circles to visualize the detected and predicted positions
        cv2.circle(frame, (cx, cy), 20, (0, 0, 255), 4)  # Detected position in blue
        cv2.circle(frame, (predicted[0], predicted[1]), 20, (255, 0, 0), 4)  # Predicted position in red

        # Display the frame
        cv2.imshow("Frame", frame)

        # Print the predicted positions
        print(predicted)
        i.append(predicted) 
        # Check for the 'Esc' key to exit
        key = cv2.waitKey(150)
        if key == 27:
            break

    # Release the video capture object and close the window
    cap.release()
    cv2.destroyAllWindows()
    return i