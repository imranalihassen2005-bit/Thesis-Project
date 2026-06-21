import os
import glob
import numpy as np
import tensorflow as tf
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

model_path = r"d:\SEMESTER 8 COURSES\Thesis\Final_Project\Models\multiclass_model_100_epochs.keras"

print("Loading model...")
model = tf.keras.models.load_model(model_path)
classes = ['Normal', 'Pneumonia', 'Tuberculosis']

uploads = glob.glob(r"d:\SEMESTER 8 COURSES\Thesis\Final_Project\static\uploads\*.jpg") + \
          glob.glob(r"d:\SEMESTER 8 COURSES\Thesis\Final_Project\static\uploads\*.png")

from tensorflow.keras.applications.efficientnet import preprocess_input

for img_path in uploads[:5]:
    print(f"\nEvaluating: {os.path.basename(img_path)}")
    img = Image.open(img_path).convert('RGB').resize((224, 224), Image.NEAREST)

    # EfficientNet preprocessing (consistent with training)
    arr = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
    arr = preprocess_input(arr)
    preds = model.predict(arr, verbose=0)[0]

    pred_class = classes[np.argmax(preds)]
    confidence = float(preds[np.argmax(preds)]) * 100

    print(f"  Prediction: {pred_class} ({confidence:.2f}%)")
    print(f"  Probabilities: {dict(zip(classes, [f'{p*100:.2f}%' for p in preds]))}")
