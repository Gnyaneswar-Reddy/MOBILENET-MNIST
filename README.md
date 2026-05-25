# MNIST X MobileNetV2

An AI project that takes Google's powerful image-recognition model (MobileNetV2), customizes it, and trains it to read hand-written numbers (0-9) from the MNIST dataset with an incredible **98.93% accuracy**.

---

## 🔥 Smart Fixes We Added to Make It Work

* **Conveyor-Belt Data Loader:** Instead of loading all 60,000 images at once and crashing the computer's memory, we built a pipeline that streams and resizes small batches of images on the fly.
* **Smart Freezing:** We locked the core "brain" of the model at first so its pre-trained knowledge wouldn't get ruined by our new, untrained layers.
* **Saving the Color Tracks:** We kept the model's built-in normalization layers frozen the entire time to prevent it from getting confused by our black-and-white images.
* **Custom AI Head:** We attached a brand new, 2-layer decision-making network to the end of the model to handle the final 0-to-9 number guessing.

---

## 🧠 Why This Model is So Fast and Smart

* **Two-Step Filtering:** Instead of looking at shapes and colors all at once, the model separates these tasks to cut down the math workload by up to 90%.
* **Information Safeguard:** The model temporarily expands data into a larger mathematical space while filtering it so it doesn't accidentally erase subtle lines and curves.
* **Two-Phase Training:** We trained our new decision layers first, and only unlocked the core model once the final layers stopped making wild guesses.

---

## 🔮 Next Steps to Improve It Even Further

* **Image Distortions:** We plan to randomly tilt and zoom the training images so the AI can read messy or crooked handwriting.
* **Auto-Slowing Learning Rate:** We will add a feature that automatically shrinks our tuning steps as the model gets closer to a perfect score so it doesn't overshoot the target.
* **Early Stopping:** We will set up a trigger that automatically stops the training process the exact moment the AI stops improving to save time and prevent overfitting.

---
## 📊 Final Performance Metrics

### Overall Evaluation
* **Test Accuracy:** `98.93%`
* **Test Loss:** `0.0385`

### Classification Report Breakdown

```text
              precision    recall  f1-score   support

           0     0.9909    0.9980    0.9944       980
           1     0.9912    0.9947    0.9930      1135
           2     0.9875    0.9961    0.9918      1032
           3     0.9960    0.9802    0.9880      1010
           4     0.9888    0.9888    0.9888       982
           5     0.9683    0.9944    0.9812       892
           6     0.9968    0.9781    0.9874       958
           7     0.9951    0.9864    0.9907      1028
           8     0.9887    0.9897    0.9892       974
           9     0.9881    0.9861    0.9871      1009

    accuracy                         0.9893     10000
   macro avg     0.9891    0.9893    0.9892     10000
weighted avg     0.9894    0.9893    0.9893     10000
