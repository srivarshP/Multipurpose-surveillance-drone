import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load your trained model (assuming you saved it as my_model.h5)
model = load_model('tomato_disease_model.h5')

# Load test data using a generator (adjust the directory path as needed)
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    "C:/robotics/final-year project/AI MODEL/tomato/train",  # path to test data directory
    target_size=(224, 224),  # resize images if needed
    batch_size=32,
    class_mode='categorical'
)
# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy}")
print(f"Test Loss: {test_loss}")
model.save('final_model.keras')


