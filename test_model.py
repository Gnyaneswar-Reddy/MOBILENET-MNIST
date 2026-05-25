import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Suppress the oneDNN CPU warning
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

print("Loading test data...")
# We only need the test dataset this time, so we use (_, _) to ignore the training data
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# ==========================================
# 1. Rebuild the Preprocessing Pipeline
# ==========================================
# We must process the test images the EXACT same way we trained them
def preprocess(image, label):
    image = tf.expand_dims(image, axis=-1)
    image = tf.image.grayscale_to_rgb(image)
    image = tf.image.resize(image, [96, 96])
    image = (image / 127.5) - 1.0
    return image, label

BATCH_SIZE = 64

print("Building test data pipeline...")
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ==========================================
# 2. Load the Saved Model
# ==========================================
print("Loading the trained model...")
# Make sure the filename matches what you saved it as!
model = tf.keras.models.load_model("mnist_mobilenet_96x96.keras")

# ==========================================
# 3. Overall Accuracy and Loss
# ==========================================
print("\n--- 1. OVERALL METRICS ---")
# evaluate() runs the whole dataset through and calculates the final loss/accuracy
loss, accuracy = model.evaluate(test_dataset, verbose=0)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Test Loss:     {loss:.4f}")

# ==========================================
# 4. Detailed Predictions & Stats
# ==========================================
print("\nGenerating predictions for advanced statistics...")
# predict() outputs raw probabilities. We want the actual digit guesses.
raw_predictions = model.predict(test_dataset)
predicted_digits = np.argmax(raw_predictions, axis=1)

print("\n--- 2. CLASSIFICATION REPORT ---")
# This compares the true labels (y_test) to our model's guesses (predicted_digits)
print(classification_report(y_test, predicted_digits, digits=4))

# ==========================================
# 5. Visual Confusion Matrix
# ==========================================
print("\n--- 3. GENERATING CONFUSION MATRIX ---")
cm = confusion_matrix(y_test, predicted_digits)

# Set up the matplotlib figure
plt.figure(figsize=(10, 8))

# Draw the heatmap using seaborn
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=[str(i) for i in range(10)],
            yticklabels=[str(i) for i in range(10)])

plt.xlabel('Predicted Digit (What the AI Guessed)', fontsize=12)
plt.ylabel('True Digit (The Actual Answer)', fontsize=12)
plt.title('MobileNetV2 Performance on MNIST', fontsize=16)

print("Opening heatmap window...")
plt.show()