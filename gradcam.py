"""
TB-CAD Clinical Suite — Grad-CAM Module
Generates Gradient-weighted Class Activation Maps for EfficientNetB0.
"""

import numpy as np
import tensorflow as tf
import cv2
import os
from PIL import Image


def find_last_conv_layer(model):
    """Find the last convolutional layer in the model."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        # EfficientNet wraps layers in a functional model
        if hasattr(layer, 'layers'):
            for sub_layer in reversed(layer.layers):
                if isinstance(sub_layer, tf.keras.layers.Conv2D):
                    return sub_layer.name
    # Fallback: try common EfficientNetB0 layer names
    for name in ['top_conv', 'block7a_project_conv', 'block6d_project_conv']:
        try:
            model.get_layer(name)
            return name
        except ValueError:
            continue
    return None


def generate_gradcam(model, img_array, class_index, layer_name=None):
    """
    Generate a Grad-CAM heatmap for the given image and class.
    
    Args:
        model: Loaded Keras model
        img_array: Preprocessed image array (1, 224, 224, 3)
        class_index: Index of the predicted class
        layer_name: Name of the convolutional layer to use (auto-detected if None)
    
    Returns:
        heatmap: Numpy array (224, 224) with values 0-255
    """
    if layer_name is None:
        layer_name = find_last_conv_layer(model)
        if layer_name is None:
            # If we still can't find it, try to use the EfficientNet backbone
            for layer in model.layers:
                if 'efficientnet' in layer.name.lower():
                    for sub_layer in reversed(layer.layers):
                        if 'conv' in sub_layer.name.lower() and len(sub_layer.output_shape) == 4:
                            layer_name = sub_layer.name
                            break
                    break
    
    if layer_name is None:
        raise ValueError("Could not find a convolutional layer for Grad-CAM")

    # Try to get the conv layer - might be nested in a sub-model
    try:
        conv_layer = model.get_layer(layer_name)
        grad_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=[conv_layer.output, model.output]
        )
    except ValueError:
        # Layer might be inside a nested model (e.g., EfficientNetB0 backbone)
        for layer in model.layers:
            if hasattr(layer, 'layers'):
                try:
                    conv_layer = layer.get_layer(layer_name)
                    grad_model = tf.keras.models.Model(
                        inputs=model.input,
                        outputs=[conv_layer.output, model.output]
                    )
                    break
                except ValueError:
                    continue
        else:
            raise ValueError(f"Could not find layer '{layer_name}' in model or sub-models")

    # Compute gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_index]

    # Get gradients of the loss with respect to conv layer output
    grads = tape.gradient(loss, conv_outputs)

    if grads is None:
        # Fallback: return a blank heatmap
        return np.zeros((224, 224), dtype=np.uint8)

    # Global average pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the conv output channels by their importance
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU and normalize
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    # Resize to image dimensions
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)

    return heatmap


def create_gradcam_overlay(original_img_path, heatmap, output_path, alpha=0.4):
    """
    Overlay the Grad-CAM heatmap on the original image and save.
    
    Args:
        original_img_path: Path to the original image
        heatmap: Grad-CAM heatmap array (224, 224)
        output_path: Path to save the overlaid image
        alpha: Transparency of the heatmap overlay
    
    Returns:
        output_path: Path to saved overlay image
    """
    # Load original image
    original = cv2.imread(original_img_path)
    if original is None:
        # Try with PIL
        pil_img = Image.open(original_img_path).convert('RGB')
        original = np.array(pil_img)
        original = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
    
    original = cv2.resize(original, (224, 224))

    # Apply colormap to heatmap
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay
    overlay = cv2.addWeighted(original, 1 - alpha, heatmap_colored, alpha, 0)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, overlay)

    return output_path


def preprocess_image(image_path):
    """
    Preprocess an image for model prediction.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        img_array: Preprocessed numpy array (1, 224, 224, 3)
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224), Image.NEAREST)
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    from tensorflow.keras.applications.efficientnet import preprocess_input
    return preprocess_input(img_array)


def preprocess_base64_frame(base64_data):
    """
    Preprocess a base64-encoded image frame (from webcam).
    
    Args:
        base64_data: Base64 encoded image string
    
    Returns:
        img_array: Preprocessed numpy array (1, 224, 224, 3)
    """
    import base64
    import io

    # Decode base64
    if ',' in base64_data:
        base64_data = base64_data.split(',')[1]
    
    img_bytes = base64.b64decode(base64_data)
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img = img.resize((224, 224), Image.NEAREST)
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    from tensorflow.keras.applications.efficientnet import preprocess_input
    return preprocess_input(img_array)
