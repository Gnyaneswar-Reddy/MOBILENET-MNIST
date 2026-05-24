import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import numpy as np

# Optional: Check if TensorFlow sees your GPU (will say 0 if you are using CPU)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# ==========================================
# 1. Load Data
# ==========================================
print("Loading MNIST data...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Slicing the data to save time (10,000 for training, 2,000 for testing)
# Feel free to remove the slicing if you want to train on all 60,000 images!
x_train, y_train = x_train[:10000], y_train[:10000]
x_test, y_test = x_test[:2000], y_test[:2000]

# ==========================================
# 2. Vectorized Preprocessing (No loops!)
# ==========================================
print("Preprocessing images (resizing and coloring)...")

# Add the missing channel dimension: (28, 28) -> (28, 28, 1)
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# Convert grayscale to RGB: (28, 28, 1) -> (28, 28, 3)
x_train_rgb = tf.image.grayscale_to_rgb(tf.convert_to_tensor(x_train))
x_test_rgb = tf.image.grayscale_to_rgb(tf.convert_to_tensor(x_test))

# Resize to 32x32 (MobileNetV2's minimum size for maximum speed)
x_train_resized = tf.image.resize(x_train_rgb, [32, 32])
x_test_resized = tf.image.resize(x_test_rgb, [32, 32])

# Normalize pixel values to be between [-1, 1] (MobileNetV2 expects this)
x_train_final = (x_train_resized / 127.5) - 1.0
x_test_final = (x_test_resized / 127.5) - 1.0

# ==========================================
# 3. Build the Transfer Learning Model
# ==========================================
print("Downloading and building MobileNetV2...")

# Load the base model without the top classification layer
base_model = MobileNetV2(input_shape=(32, 32, 3), 
                         include_top=False, 
                         weights='imagenet')

# Freeze the pre-trained weights so we don't ruin them
base_model.trainable = False 

# Attach our custom 10-digit classifier head
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# ==========================================
# 4. Train the Model
# ==========================================
print("Starting training...")
history = model.fit(x_train_final, y_train, 
                    epochs=3, 
                    batch_size=32, 
                    validation_data=(x_test_final, y_test))

print("Training complete!")

# ==========================================
# 5. Fine-Tuning Phase (CORRECTED)
# ==========================================
print("\nUnfreezing the top layers for fine-tuning...")

# Unfreeze the base model
base_model.trainable = True

# 1. THE FIX: Loop through and explicitly freeze ALL Batch Normalization layers
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

# 2. Freeze the bottom layers (we still only want to train the top 20)
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Recompile with a tiny learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

print("Starting fine-tuning...")
# Let it train for 8 epochs this time so it can actually learn the digits
history_fine = model.fit(x_train_final, y_train, 
                         epochs=8, 
                         batch_size=32, 
                         validation_data=(x_test_final, y_test))

# Save the properly trained model
model.save("mnist_mobilenet.keras")
print("Model saved successfully as 'mnist_mobilenet.keras'!")