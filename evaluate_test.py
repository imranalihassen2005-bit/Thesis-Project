import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input

model_path = r"D:\SEMESTER 8 COURSES\Thesis\Final_Project\Models\multiclass_model_100_epochs.keras"
print("Loading model...")
model = tf.keras.models.load_model(model_path)

TEST_DIR = r"D:\SEMESTER 8 COURSES\Thesis\Thesis Project\Dataset\final\test"

val_test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

test_data = val_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

print("Evaluating on test set...")
loss, acc = model.evaluate(test_data)
print(f"Test Accuracy: {acc*100:.2f}%")
print(f"Test Loss: {loss:.4f}")

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

pred_probs = model.predict(test_data)
y_pred = np.argmax(pred_probs, axis=1)
y_true = test_data.classes
class_names = list(test_data.class_indices.keys())

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))
