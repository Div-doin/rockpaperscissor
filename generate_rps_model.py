import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# A simple CNN model for rock-paper-scissors detection
# (Not trained, but has correct structure so your program will run)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(227, 227, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),

    Dense(4, activation='softmax')  # rock, paper, scissors, none
])

# Save it as .h5 so your game can load it
model.save("rock-paper-scissors-model.h5")

print("Model created successfully as rock-paper-scissors-model.h5")
