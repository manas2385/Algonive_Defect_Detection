import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# -------------------------------
# Data Generators
# -------------------------------
train_gen = ImageDataGenerator(rescale=1./255)
val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    'data/train',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

val_data = val_gen.flow_from_directory(
    'data/validation',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

# -------------------------------
# Debug: Check classes
# -------------------------------
print("\nClass Mapping:", train_data.class_indices)
num_classes = len(train_data.class_indices)
print("Number of Classes:", num_classes)

# -------------------------------
# CNN Model
# -------------------------------
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')  # Dynamic output layer
])

# -------------------------------
# Compile
# -------------------------------
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -------------------------------
# Train
# -------------------------------
model.fit(
    train_data,
    validation_data=val_data,
    epochs=5
)

# -------------------------------
# Save Model
# -------------------------------
model.save("defect_model.h5")

print("\nModel trained and saved successfully")