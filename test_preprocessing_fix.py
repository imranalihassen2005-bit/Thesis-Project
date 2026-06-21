import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

model_path = r"d:\SEMESTER 8 COURSES\Thesis\Final_Project\Models\multiclass_model_100_epochs.keras"
print("Loading model...")
model = tf.keras.models.load_model(model_path)
classes = ['Normal', 'Pneumonia', 'Tuberculosis']

uploads = glob.glob(r"d:\SEMESTER 8 COURSES\Thesis\Final_Project\static\uploads\*.jpg") + \
          glob.glob(r"d:\SEMESTER 8 COURSES\Thesis\Final_Project\static\uploads\*.png")

print(f"Testing {len(uploads)} uploaded images using keras preprocessing...")
for img_path in uploads[:10]:
    print(f"\nEvaluating: {os.path.basename(img_path)}")
    
    # Correct Keras preprocessing (nearest interpolation)
    img = image.load_img(img_path, target_size=(224, 224))
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    
    preds = model.predict(arr, verbose=0)[0]
    pred_class = classes[np.argmax(preds)]
    confidence = float(preds[np.argmax(preds)]) * 100
    
    print(f"  Prediction: {pred_class} ({confidence:.2f}%)")
    print(f"  Probabilities: {dict(zip(classes, [f'{p*100:.2f}%' for p in preds]))}")
