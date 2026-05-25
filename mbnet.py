import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import numpy as np
import os

# Suppress the oneDNN warning to clean up your console
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ==========================================
# 1. Load the Tiny Raw Data
# ==========================================
print("Loading raw MNIST data...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# ==========================================
# 2. Build the Preprocessing "Conveyor Belt"
# ==========================================
# Instead of doing the math right now, we just give TensorFlow the instructions
def preprocess(image, label):
    # 1. Add channel dimension: (28, 28) -> (28, 28, 1)
    image = tf.expand_dims(image, axis=-1)
    # 2. Convert to RGB: (28, 28, 1) -> (28, 28, 3)
    image = tf.image.grayscale_to_rgb(image)
    # 3. Resize to 96x96
    image = tf.image.resize(image, [96, 96])
    # 4. Normalize to [-1, 1]
    image = (image / 127.5) - 1.0
    return image, label

BATCH_SIZE = 64

print("Building data pipelines...")
# Create the training pipeline
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.cache().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Create the testing pipeline
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Free up the tiny amount of memory the original arrays used
del x_train, y_train, x_test, y_test

# ==========================================
# 3. Build the Base Model
# ==========================================
print("Building MobileNetV2...")
base_model = MobileNetV2(input_shape=(96, 96, 3), 
                         include_top=False, 
                         weights='imagenet')

base_model.trainable = False 

# ==========================================
# 4. Add the Custom "Deep" Head
# ==========================================
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'), 
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# ==========================================
# 5. Train the Head (Phase 1)
# ==========================================
print("\n--- PHASE 1: Training the Custom Head ---")
# Notice we just pass 'train_dataset' now instead of x and y arrays
model.fit(train_dataset, 
          epochs=2, 
          validation_data=test_dataset)

# ==========================================
# 6. Fine-Tuning: Unlock ALL Parameters (Phase 2)
# ==========================================
print("\n--- PHASE 2: Unlocking MobileNetV2 for Fine-Tuning ---")

base_model.trainable = True

# Freeze Batch Normalization Layers
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_accuracy', factor=0.5, patience=1, min_lr=1e-7)

history_fine = model.fit(train_dataset, 
                         epochs=5, 
                         validation_data=test_dataset,
                         callbacks=[lr_callback])

model.save("mnist_mobilenet_96x96.keras")
print("\nComplete! Model saved as 'mnist_mobilenet_96x96.keras'")
