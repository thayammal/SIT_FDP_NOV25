import tensorflow as tf
from tensorflow import keras
import numpy as np

# Suppress warnings and set log level for cleaner output
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# --- 1. Load Data (Simulating "Inbuilt" Image Data) ---
# MNIST is a classic computer vision dataset, often used as the "Hello World"
# for CNNs and is easily accessible via Keras.
print("Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# --- 2. Data Preprocessing (Image Processing Steps) ---

# Image Shape Check: MNIST images are 28x28 grayscale.
img_rows, img_cols = 28, 28
num_classes = 10

# Reshape the data to include a channel dimension (1 for grayscale).
# CNNs expect the input shape to be (samples, rows, cols, channels).
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

# Normalization: Convert pixel values from 0-255 to 0-1 (Floating point).
# This is a critical image preprocessing step for neural networks.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

print(f"Training data shape: {x_train.shape}")
print(f"Test data shape: {x_test.shape}")
print(f"Number of training samples: {x_train.shape[0]}")

# Convert class vectors to binary class matrices (One-Hot Encoding).
# e.g., digit 5 becomes [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# --- 3. Define the CNN Model Architecture ---

print("\nBuilding CNN model...")

model = keras.Sequential([
    # Convolutional Layer: 32 filters, 3x3 kernel, ReLU activation.
    # The first layer must define the input shape.
    keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),

    # Pooling Layer: Reduces dimensionality, makes the model more robust to shifts.
    keras.layers.MaxPooling2D(pool_size=(2, 2)),

    # Second Convolutional Layer
    keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),

    # Second Pooling Layer
    keras.layers.MaxPooling2D(pool_size=(2, 2)),

    # Flatten Layer: Converts the 2D feature maps into a 1D vector for the Dense layers.
    keras.layers.Flatten(),

    # Dropout Layer: Helps prevent overfitting during training.
    keras.layers.Dropout(0.5),

    # Dense (Fully Connected) Layer: Standard neural network layer.
    keras.layers.Dense(num_classes, activation='softmax')
])

# --- 4. Compile and Train the Model ---

# Compile: Defines the optimizer, loss function, and metrics.
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train: Fit the model to the training data.
batch_size = 128
epochs = 5 # Using a small number of epochs for quick demonstration
print("\nStarting model training (5 epochs)...")

history = model.fit(
    x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(x_test, y_test)
)

# --- 5. Evaluate the Model ---

print("\nEvaluating model performance on test data...")
score = model.evaluate(x_test, y_test, verbose=0)

print(f"Test Loss: {score[0]:.4f}")
print(f"Test Accuracy: {score[1]*100:.2f}%")

# --- 6. Prediction Example ---
# Pick a random test image to see the prediction
sample_idx = np.random.randint(0, x_test.shape[0])
sample_image = x_test[sample_idx]
true_label = np.argmax(y_test[sample_idx])

# The model expects a batch, so we add a dimension
sample_image_batch = np.expand_dims(sample_image, axis=0)

prediction = model.predict(sample_image_batch, verbose=0)
predicted_label = np.argmax(prediction)

print(f"\n--- Prediction Test ---")
print(f"Random sample index: {sample_idx}")
print(f"Actual Digit (True Label): {true_label}")
print(f"Predicted Digit (Model Output): {predicted_label}")
print(f"Prediction Confidence: {prediction[0][predicted_label]*100:.2f}%")

print("\nModel training and evaluation complete.")

import matplotlib.pyplot as plt

# --- After your prediction code for a sample image ---
sample_idx = np.random.randint(0, x_test.shape[0])
sample_image = x_test[sample_idx]
true_label = np.argmax(y_test[sample_idx])

sample_image_batch = np.expand_dims(sample_image, axis=0)
prediction = model.predict(sample_image_batch, verbose=0)
predicted_label = np.argmax(prediction)
confidence = prediction[0][predicted_label] * 100

# Visualize input image and prediction
plt.figure(figsize=(6,3))

# Show the input image
plt.subplot(1,2,1)
plt.imshow(sample_image.squeeze(), cmap='gray')
plt.title(f"Input Image\nTrue: {true_label}")
plt.axis('off')

# Show text output for prediction
plt.subplot(1,2,2)
plt.text(0.5, 0.5,
         f"Predicted: {predicted_label}\nConfidence: {confidence:.2f}%",
         fontsize=14, ha='center')
plt.axis('off')

plt.tight_layout()
plt.show()
