import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# -------------------------
# Load Dataset
# -------------------------
train_dir = "dataset"

img_size = 227
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2,        # 80% train / 20% validation
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="categorical",
    subset="training"
)

val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation"
)

# -------------------------
# Build CNN Model
# -------------------------
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(img_size, img_size, 3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(4, activation="softmax")  # rock, paper, scissors, nothing
])

model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(learning_rate=0.0001),
    metrics=["accuracy"]
)

# -------------------------
# Train Model
# -------------------------
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20
)

# -------------------------
# Save Model
# -------------------------
model.save("rock-paper-scissors-model.h5")
print("Model saved as rock-paper-scissors-model.h5")
