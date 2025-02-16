import cv2
import numpy as np
from keras.models import load_model

# Load the trained model
model = load_model('tomato.h5')

# Compile the model (optional, depending on your use case)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# List of class names
class_names = ['Tomato_disease', 'Tomato_healthy']

# Start capturing video from the camera
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Resize and preprocess the frame
    img = cv2.resize(frame, (224, 224))  # Resize to (128, 128)
    img_array = np.expand_dims(img, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image

    # Make a prediction
    prediction = model.predict(img_array)

    # Get the predicted class (index of the highest probability)
    predicted_index = np.argmax(prediction, axis=1)[0]  # Get the index of the class with the highest probability
    predicted_class = class_names[predicted_index]

    # Display the prediction on the frame
    cv2.putText(frame, f'Prediction: {predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Show the frame
    cv2.imshow('Real-Time Prediction', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
