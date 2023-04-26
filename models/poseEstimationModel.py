import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Load images and position-orientation data
image_folder = "../data/rawdata/volunteer01/filtered_images"
image_files = sorted(os.listdir(image_folder))
images = [cv2.imread(os.path.join(image_folder, img)) for img in image_files]
positions_orientations = np.load("../data/rawdata/synthetic_positions_orientations.npy")

# Preprocess images outsorce this to preprocess py
image_size = (128, 128)  # Resize the images to a smaller size
images = []
for img_path in image_files:
    img = cv2.imread(os.path.join(image_folder, img_path))
    img_resized = cv2.resize(img, image_size)
    images.append(img_resized)

print("Number of images:", len(images))
print("Number of position-orientation data:", positions_orientations.shape[0])
# Split data into training and validation sets
images = np.array(images)
X_train, X_val, y_train, y_val = train_test_split(images, positions_orientations[:, 1:], test_size=0.2, random_state=42, shuffle=True)

print("X_train shape:", np.array(X_train).shape)
print("X_val shape:", np.array(X_val).shape)
print("y_train shape:", np.array(y_train).shape)
print("y_val shape:", np.array(y_val).shape)

# Create the neural network model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(*image_size, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='linear'))  # Output layer for position (3 values) and orientation (4 values)

# Compile and train the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_val, y_val))

model.save("ultrasound_pose_estimator.h5")