import os
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

model_path = r"D:\SEMESTER 8 COURSES\Thesis\Final_Project\Models\multiclass_model_100_epochs.keras"
model = tf.keras.models.load_model(model_path)

test_normal_dir = r"D:\SEMESTER 8 COURSES\Thesis\Thesis Project\Dataset\final\test\Normal"
img_names = os.listdir(test_normal_dir)[:5]

print("Evaluating using App.py method:")
for img_name in img_names:
    img_path = os.path.join(test_normal_dir, img_name)
    img = Image.open(img_path).convert('RGB').resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    arr = preprocess_input(img_array)
    preds = model.predict(arr, verbose=0)[0]
    pred_class = ['Normal', 'Pneumonia', 'Tuberculosis'][np.argmax(preds)]
    print(f"  {img_name} -> {pred_class} (N:{preds[0]:.2f}, P:{preds[1]:.2f}, T:{preds[2]:.2f})")

from tensorflow.keras.preprocessing import image
print("\nEvaluating using keras image.load_img method:")
for img_name in img_names:
    img_path = os.path.join(test_normal_dir, img_name)
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    arr = preprocess_input(img_array)
    preds = model.predict(arr, verbose=0)[0]
    pred_class = ['Normal', 'Pneumonia', 'Tuberculosis'][np.argmax(preds)]
    print(f"  {img_name} -> {pred_class} (N:{preds[0]:.2f}, P:{preds[1]:.2f}, T:{preds[2]:.2f})")
