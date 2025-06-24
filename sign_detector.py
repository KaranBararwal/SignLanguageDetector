import cv2
import numpy as np
from tensorflow.keras.models import load_model

# load your trained model
model = load_model('asl_model.h5')

# label mapping : A-Z excluding 'J' and 'Z'
labels_map = [chr(i) for i in range(65,91) if i != 74] 

# preprocessing function for the hand ROI
def preprocess_image(image):
    gray = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY) # convert to grayscale
    resized = cv2.resize(gray , (28,28)) # resize to 28*28 (model expects this size)
    norm = resized / 255.0 #normalize pixel values
    reshaped = norm.reshape(1, 28, 28, 1) # reshape to match model input shape
    return reshaped

# open the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("Could not open webcam")

print("📷 Webcam active. Show your sign inside the box.")
print("👉 Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    # define region of interest (ROI) for the hand
    x1, y1, x2, y2 = 100,100,300,300
    roi = frame[y1:y2, x1:x2]
    input_image = preprocess_image(roi)

    # predict the sign
    prediction = model.predict(input_image)
    predicted_index = np.argmax(prediction)
    predicted_label = labels_map[predicted_index]

    # draw the roi and label on the frame
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f"Prediction: {predicted_label}" , (x1 , y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9 , (0, 255, 0), 2)
    
    # show the frame
    cv2.imshow("Sign Language Detector", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
