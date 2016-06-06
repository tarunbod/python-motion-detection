import sys
import cv2
import numpy as np

args = sys.argv[1:]

video = cv2.VideoCapture(0)
lastFrame = None

while True:
    ret, frame = video.read()
    if not ret:
        break
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray, (5,5))

    if lastFrame is None:
        if not np.any(frame):
            continue
    else:
        diffImage = cv2.absdiff(lastFrame, gray)
        _, threshImage = cv2.threshold(diffImage, 42, 255, cv2.THRESH_BINARY)
        threshImage = cv2.dilate(threshImage, None, iterations=2)
        _, contours, _ = cv2.findContours(threshImage.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for contour in contours:
            if cv2.contourArea(contour) > 500:
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        cv2.imshow("Image", frame)
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    lastFrame = gray


cv2.destroyAllWindows()
video.release()
