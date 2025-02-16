import os
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = 224
BATCH_SIZE = 32

data_dir = "C:/robotics/final-year project/AI MODEL/PlantVillage"
train_datagen = ImageDataGenerator(
    rescale=1./255,         
    rotation_range=40,       
    width_shift_range=0.2,  
    height_shift_range=0.2,  
    shear_range=0.2,         
    zoom_range=0.2,          
    horizontal_flip=True,    
    fill_mode='nearest'      
)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(IMG_SIZE, IMG_SIZE),  # Resize images
    batch_size=BATCH_SIZE,
    class_mode='categorical'           # Multi-class classification
)



