import cv2

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
print("Webcam working?", ret)
cap.release()
