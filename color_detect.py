import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while True: 
    _, frame = cap.read()

    threshImage = cv2.inRange(frame, np.array([17, 15, 100]), np.array([50, 56, 200]))

    colorImage = cv2.bitwise_and(frame, frame, mask = threshImage)

    cv2.imshow("Thresholded Image", threshImage)
    cv2.imshow("Color Detected", colorImage)
    cv2.imshow("Original", frame)

    if cv2.waitKey(33) == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
