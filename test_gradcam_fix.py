import os
import numpy as np
import tensorflow as tf
from gradcam import preprocess_image

model_path = r"d:\SEMESTER 8 COURSES\Thesis\Final_Project\Models\multiclass_model_100_epochs.keras"
model = tf.keras.models.load_model(model_path)

test_normal_dir = r"D:\SEMESTER 8 COURSES\Thesis\Thesis Project\Dataset\final\test\Normal"
img_names = os.listdir(test_normal_dir)[:3]
classes = ['Normal', 'Pneumonia', 'Tuberculosis']

print("Testing gradcam.preprocess_image with Image.NEAREST fix:")
for img_name in img_names:
    img_path = os.path.join(test_normal_dir, img_name)
    arr = preprocess_image(img_path)
    preds = model.predict(arr, verbose=0)[0]
    pred_class = classes[np.argmax(preds)]
    confidence = float(preds[np.argmax(preds)]) * 100
    print(f"  {img_name} -> {pred_class} ({confidence:.2f}%) N:{preds[0]*100:.2f}% P:{preds[1]*100:.2f}% T:{preds[2]*100:.2f}%")
