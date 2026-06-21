"""
Test DenseNet121 model to see if it can complement the EfficientNetB0 model.
"""
import os
import glob
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from PIL import Image

print("=" * 60)
print("  DenseNet121 Model Diagnostic")
print("=" * 60)

# Load DenseNet model
model_path = r"d:\SEMESTER 8 COURSES\Thesis\Final_Project\Models\multiclass_densenet121.keras"
print("\n[1] Loading DenseNet121 model...")
model = tf.keras.models.load_model(model_path)

# Check output layer
output_layer = model.layers[-1]
print(f"  Output layer: {output_layer.name}, units={getattr(output_layer, 'units', '?')}, "
      f"activation={getattr(output_layer, 'activation', lambda: None).__name__}")

# Check what preprocessing it expects
print("\n[2] Model input shape:", model.input_shape)

# Test with synthetic inputs - try BOTH preprocessing methods
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess

classes = ['Normal', 'Pneumonia', 'Tuberculosis']

for preprocess_name, preprocess_fn in [("DenseNet preprocess", densenet_preprocess), 
                                         ("EfficientNet preprocess", effnet_preprocess)]:
    print(f"\n[3] Synthetic tests with {preprocess_name}:")
    
    # Black
    black = np.zeros((1, 224, 224, 3), dtype=np.float32)
    pred = model.predict(preprocess_fn(black.copy()), verbose=0)[0]
    print(f"  Black:  N={pred[0]*100:6.2f}%  P={pred[1]*100:6.2f}%  TB={pred[2]*100:6.2f}%  -> {classes[np.argmax(pred)]}")
    
    # White
    white = np.ones((1, 224, 224, 3), dtype=np.float32) * 255
    pred = model.predict(preprocess_fn(white.copy()), verbose=0)[0]
    print(f"  White:  N={pred[0]*100:6.2f}%  P={pred[1]*100:6.2f}%  TB={pred[2]*100:6.2f}%  -> {classes[np.argmax(pred)]}")
    
    # Gray
    gray = np.ones((1, 224, 224, 3), dtype=np.float32) * 128
    pred = model.predict(preprocess_fn(gray.copy()), verbose=0)[0]
    print(f"  Gray:   N={pred[0]*100:6.2f}%  P={pred[1]*100:6.2f}%  TB={pred[2]*100:6.2f}%  -> {classes[np.argmax(pred)]}")
    
    # Noise
    np.random.seed(42)
    noise = np.random.randint(0, 256, (1, 224, 224, 3)).astype(np.float32)
    pred = model.predict(preprocess_fn(noise.copy()), verbose=0)[0]
    print(f"  Noise:  N={pred[0]*100:6.2f}%  P={pred[1]*100:6.2f}%  TB={pred[2]*100:6.2f}%  -> {classes[np.argmax(pred)]}")

# Test with uploaded images using DenseNet preprocessing
print(f"\n[4] Uploaded images (DenseNet preprocess):")
uploads = glob.glob(r"d:\SEMESTER 8 COURSES\Thesis\Final_Project\static\uploads\*.jpg") + \
          glob.glob(r"d:\SEMESTER 8 COURSES\Thesis\Final_Project\static\uploads\*.png") + \
          glob.glob(r"d:\SEMESTER 8 COURSES\Thesis\Final_Project\static\uploads\*.jpeg")

for img_path in uploads[:10]:
    img = Image.open(img_path).convert('RGB').resize((224, 224))
    arr = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
    arr = densenet_preprocess(arr)
    preds = model.predict(arr, verbose=0)[0]
    pred_class = classes[np.argmax(preds)]
    print(f"  {os.path.basename(img_path):30s} -> {pred_class:12s}  "
          f"N={preds[0]*100:5.2f}%  P={preds[1]*100:5.2f}%  TB={preds[2]*100:5.2f}%")

print("\n" + "=" * 60)
print("  DenseNet diagnostic complete")
print("=" * 60)
