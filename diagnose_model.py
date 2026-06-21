"""
Diagnostic script to understand the model's behavior and class biases.
"""
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

print("=" * 60)
print("  TB-CAD Model Diagnostic Tool")
print("=" * 60)

# Load model
model_path = r"d:\SEMESTER 8 COURSES\Thesis\Final_Project\Models\multiclass_model_100_epochs.keras"
print("\n[1] Loading model...")
model = tf.keras.models.load_model(model_path)

# Print model summary
print("\n[2] Model Architecture Summary:")
model.summary()

# Check the output layer
print("\n[3] Output Layer Details:")
output_layer = model.layers[-1]
print(f"  Layer name: {output_layer.name}")
print(f"  Layer type: {type(output_layer).__name__}")
if hasattr(output_layer, 'activation'):
    print(f"  Activation: {output_layer.activation.__name__}")
if hasattr(output_layer, 'units'):
    print(f"  Units: {output_layer.units}")

# Check output layer weights/biases
print("\n[4] Output Layer Weights Analysis:")
weights = output_layer.get_weights()
if len(weights) >= 2:
    w, b = weights[0], weights[1]
    print(f"  Weight shape: {w.shape}")
    print(f"  Bias values: {b}")
    print(f"  Bias labels: Normal={b[0]:.4f}, Pneumonia={b[1]:.4f}, Tuberculosis={b[2]:.4f}")
    print(f"  Weight norms per class:")
    for i, name in enumerate(['Normal', 'Pneumonia', 'Tuberculosis']):
        print(f"    {name}: L2={np.linalg.norm(w[:, i]):.4f}, mean={np.mean(w[:, i]):.6f}")

# Test with synthetic inputs
print("\n[5] Testing with synthetic inputs:")
from tensorflow.keras.applications.efficientnet import preprocess_input

# Create a pure black image (like a blank/empty scan)
black = np.zeros((1, 224, 224, 3), dtype=np.float32)
black_pre = preprocess_input(black.copy())
pred_black = model.predict(black_pre, verbose=0)[0]
print(f"  Black image:  Normal={pred_black[0]*100:.2f}%  Pneumonia={pred_black[1]*100:.2f}%  TB={pred_black[2]*100:.2f}%")

# Create a pure white image
white = np.ones((1, 224, 224, 3), dtype=np.float32) * 255
white_pre = preprocess_input(white.copy())
pred_white = model.predict(white_pre, verbose=0)[0]
print(f"  White image:  Normal={pred_white[0]*100:.2f}%  Pneumonia={pred_white[1]*100:.2f}%  TB={pred_white[2]*100:.2f}%")

# Create a gray image (typical CXR background)
gray = np.ones((1, 224, 224, 3), dtype=np.float32) * 128
gray_pre = preprocess_input(gray.copy())
pred_gray = model.predict(gray_pre, verbose=0)[0]
print(f"  Gray image:   Normal={pred_gray[0]*100:.2f}%  Pneumonia={pred_gray[1]*100:.2f}%  TB={pred_gray[2]*100:.2f}%")

# Random noise
np.random.seed(42)
noise = np.random.randint(0, 256, (1, 224, 224, 3)).astype(np.float32)
noise_pre = preprocess_input(noise.copy())
pred_noise = model.predict(noise_pre, verbose=0)[0]
print(f"  Random noise: Normal={pred_noise[0]*100:.2f}%  Pneumonia={pred_noise[1]*100:.2f}%  TB={pred_noise[2]*100:.2f}%")

# Test with existing uploaded images
print("\n[6] Testing with uploaded images (if any):")
import glob
from PIL import Image

uploads = glob.glob(r"d:\SEMESTER 8 COURSES\Thesis\Final_Project\static\uploads\*.jpg") + \
          glob.glob(r"d:\SEMESTER 8 COURSES\Thesis\Final_Project\static\uploads\*.png") + \
          glob.glob(r"d:\SEMESTER 8 COURSES\Thesis\Final_Project\static\uploads\*.jpeg")

if not uploads:
    print("  No uploaded images found.")
else:
    for img_path in uploads[:10]:
        img = Image.open(img_path).convert('RGB').resize((224, 224))
        arr = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
        arr = preprocess_input(arr)
        preds = model.predict(arr, verbose=0)[0]
        pred_class = ['Normal', 'Pneumonia', 'Tuberculosis'][np.argmax(preds)]
        print(f"  {os.path.basename(img_path):30s} -> {pred_class:12s}  "
              f"N={preds[0]*100:5.2f}%  P={preds[1]*100:5.2f}%  TB={preds[2]*100:5.2f}%")

print("\n[7] Checking if model uses softmax or raw logits:")
# Test if output sums to ~1.0
test_input = np.random.rand(1, 224, 224, 3).astype(np.float32) * 255
test_pre = preprocess_input(test_input.copy())
test_out = model.predict(test_pre, verbose=0)[0]
print(f"  Sum of outputs: {sum(test_out):.6f} (should be ~1.0 for softmax)")

print("\n" + "=" * 60)
print("  Diagnostic complete")
print("=" * 60)
