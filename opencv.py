import cv2
import numpy as np
from keras.models import load_model

# Load the saved model
loaded_model = load_model("model.h5")

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Start video capture (0 = default webcam)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Extract face region
        face = frame[y:y+h, x:x+w]

        # Preprocess the face for prediction
        face_resized = cv2.resize(face, (256, 256))
        face_resized = face_resized.astype("float32") / 255.0
        face_resized = np.expand_dims(face_resized, axis=0)

        # Prediction
        pred = loaded_model.predict(face_resized, verbose=0)
        class_idx = np.argmax(pred)

        # Label
        label = "Masked" if class_idx == 1 else "Unmasked"
        color = (0, 255, 0) if class_idx == 1 else (0, 0, 255)

        # Draw rectangle & label
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Show the output
    cv2.imshow("Mask Detection", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
