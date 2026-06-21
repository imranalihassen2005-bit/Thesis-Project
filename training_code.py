import os
import shutil
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

print("Libraries loaded successfully")
print("TensorFlow:", tf.__version__)
SOURCE_DIR = r"D:\SEMESTER 8 COURSES\Thesis\Thesis Project\Dataset\processed"

print(os.listdir(SOURCE_DIR))

for cls in ["Normal", "Pneumonia", "Tuberculosis"]:
    print(cls, ":", len(os.listdir(os.path.join(SOURCE_DIR, cls))))
SOURCE_DIR = r"D:\SEMESTER 8 COURSES\Thesis\Thesis Project\Dataset\processed"
OUTPUT_DIR = r"D:\SEMESTER 8 COURSES\Thesis\Thesis Project\Dataset\final"

classes = ["Normal", "Pneumonia", "Tuberculosis"]

train_ratio = 0.70
val_ratio = 0.15
test_ratio = 0.15

random.seed(42)

for cls in classes:
    src_class_dir = os.path.join(SOURCE_DIR, cls)

    images = [
        img for img in os.listdir(src_class_dir)
        if img.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    random.shuffle(images)

    total = len(images)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    splits = {
        "train": images[:train_end],
        "validation": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split_name, split_images in splits.items():
        dst_class_dir = os.path.join(OUTPUT_DIR, split_name, cls)
        os.makedirs(dst_class_dir, exist_ok=True)

        for img in split_images:
            src = os.path.join(src_class_dir, img)
            dst = os.path.join(dst_class_dir, img)
            shutil.copy2(src, dst)

print("Dataset split completed successfully")
FINAL_DATASET_PATH = r"D:\SEMESTER 8 COURSES\Thesis\Thesis Project\Dataset\final"

for split in ["train", "validation", "test"]:
    print("\n", split.upper())

    for cls in ["Normal", "Pneumonia", "Tuberculosis"]:
        path = os.path.join(FINAL_DATASET_PATH, split, cls)
        print(cls, ":", len(os.listdir(path)))
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=10,
    zoom_range=0.05,
    horizontal_flip=True
)

val_test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_data = train_datagen.flow_from_directory(
    os.path.join(FINAL_DATASET_PATH, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

val_data = val_test_datagen.flow_from_directory(
    os.path.join(FINAL_DATASET_PATH, "validation"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

test_data = val_test_datagen.flow_from_directory(
    os.path.join(FINAL_DATASET_PATH, "test"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

print(train_data.class_indices)
labels = train_data.classes

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels),
    y=labels
)

class_weights = dict(enumerate(class_weights))

print("Class Weights:", class_weights)
base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.6)(x)

output = Dense(3, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()
# ==========================================
# TRAIN MODEL - 10 EPOCHS
# ==========================================

checkpoint_path = r"D:\SEMESTER 8 COURSES\Thesis\Thesis Project\models\best_multiclass_10.keras"

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),

    ModelCheckpoint(
        checkpoint_path,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max"
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=3,
        min_lr=1e-6
    )
]

history_10 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    class_weight=class_weights,
    callbacks=callbacks
)
MODEL_10_PATH = r"D:\SEMESTER 8 COURSES\Thesis\Thesis Project\models\multiclass_model_10_epochs.keras"

model.save(MODEL_10_PATH)

print("10 Epoch Model Saved Successfully")
print("Saved At:", MODEL_10_PATH)
# ==========================================
# LOAD 10 EPOCH MODEL
# ==========================================

from tensorflow.keras.models import load_model

MODEL_10_PATH = r"D:\SEMESTER 8 COURSES\Thesis\Thesis Project\models\multiclass_model_10_epochs.keras"

model_10 = load_model(MODEL_10_PATH)

print("10 Epoch Model Loaded Successfully")
# ==========================================
# 1. CLASS DISTRIBUTION PLOT
# ==========================================

classes = ["Normal", "Pneumonia", "Tuberculosis"]

counts = [
    len(os.listdir(os.path.join(SOURCE_DIR, "Normal"))),
    len(os.listdir(os.path.join(SOURCE_DIR, "Pneumonia"))),
    len(os.listdir(os.path.join(SOURCE_DIR, "Tuberculosis")))
]

plt.figure(figsize=(8, 5))
bars = plt.bar(classes, counts)

plt.title("Class Distribution of Final Dataset")
plt.xlabel("Class")
plt.ylabel("Number of Images")

for bar in bars:
    plt.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 100,
        str(bar.get_height()),
        ha="center"
    )

plt.grid(axis="y")
plt.show()
# ==========================================
# 2. 10 EPOCH ACCURACY PLOT
# ==========================================

plt.figure(figsize=(8, 5))
plt.plot(history_10.history["accuracy"], label="Training Accuracy")
plt.plot(history_10.history["val_accuracy"], label="Validation Accuracy")

plt.title("10 Epoch Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()
# ==========================================
# 3. 10 EPOCH LOSS PLOT
# ==========================================

plt.figure(figsize=(8, 5))
plt.plot(history_10.history["loss"], label="Training Loss")
plt.plot(history_10.history["val_loss"], label="Validation Loss")

plt.title("10 Epoch Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()
# ==========================================
# 5. 10 EPOCH CONFUSION MATRIX
# ==========================================

test_data.reset()

pred_probs_10 = model_10.predict(test_data)
y_pred_10 = np.argmax(pred_probs_10, axis=1)
y_true = test_data.classes

class_names = list(test_data.class_indices.keys())

cm_10 = confusion_matrix(y_true, y_pred_10)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm_10,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title("10 Epoch Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.show()
# ==========================================
# 6. 10 EPOCH TP, TN, FP, FN PER CLASS
# ==========================================

tp_list = []
tn_list = []
fp_list = []
fn_list = []

for i, class_name in enumerate(class_names):
    TP = cm_10[i, i]
    FP = cm_10[:, i].sum() - TP
    FN = cm_10[i, :].sum() - TP
    TN = cm_10.sum() - (TP + FP + FN)

    tp_list.append(TP)
    tn_list.append(TN)
    fp_list.append(FP)
    fn_list.append(FN)

x = np.arange(len(class_names))
width = 0.2

plt.figure(figsize=(10, 6))
plt.bar(x - 1.5*width, tp_list, width, label="TP")
plt.bar(x - 0.5*width, tn_list, width, label="TN")
plt.bar(x + 0.5*width, fp_list, width, label="FP")
plt.bar(x + 1.5*width, fn_list, width, label="FN")

plt.xticks(x, class_names)
plt.title("10 Epoch TP, TN, FP, FN per Class")
plt.xlabel("Class")
plt.ylabel("Count")
plt.legend()
plt.grid(axis="y")
plt.show()
# ==========================================
# 7. CLASSIFICATION REPORT
# ==========================================

report_10 = classification_report(
    y_true,
    y_pred_10,
    target_names=class_names
)

print("10 Epoch Classification Report")
print(report_10)
# ==========================================
# SHOW 10 RANDOM TEST IMAGES WITH PREDICTIONS
# ==========================================

import random
from tensorflow.keras.preprocessing import image

test_dir = os.path.join(FINAL_DATASET_PATH, "test")

all_test_images = []

# Collect all test images
for class_name in class_names:
    class_folder = os.path.join(test_dir, class_name)

    for img_name in os.listdir(class_folder):
        if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
            all_test_images.append(
                (os.path.join(class_folder, img_name), class_name)
            )

# Select 10 random images
random_samples = random.sample(all_test_images, 10)

# Plot size
plt.figure(figsize=(20, 12))

for i, (img_path, actual_label) in enumerate(random_samples):

    # Load image
    img = image.load_img(img_path, target_size=IMG_SIZE)

    # Convert to array
    img_array = image.img_to_array(img)

    # Preprocess
    img_preprocessed = preprocess_input(
        np.expand_dims(img_array, axis=0)
    )

    # Predict
    prediction = model_10.predict(img_preprocessed, verbose=0)[0]

    predicted_index = np.argmax(prediction)

    predicted_label = class_names[predicted_index]

    confidence = prediction[predicted_index] * 100

    # Plot image
    plt.subplot(2, 5, i + 1)

    plt.imshow(img, cmap="gray")

    plt.axis("off")

    plt.title(
        f"Actual: {actual_label}\n"
        f"Predicted: {predicted_label}\n"
        f"Confidence: {confidence:.2f}%"
    )

plt.tight_layout()

plt.show()
# ==========================================
# PHASE 2 - FINE TUNING
# ==========================================

# Unfreeze top EfficientNet layers
base_model.trainable = True

# Freeze early layers
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Recompile with VERY small learning rate
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("Fine-tuning enabled.")
# ==========================================
# CONTINUE TRAINING TO 25 EPOCHS
# ==========================================

history_25 = model.fit(
    train_data,
    validation_data=val_data,
    initial_epoch=10,
    epochs=25,
    class_weight=class_weights,
    callbacks=callbacks
)
MODEL_25_PATH = r"D:\SEMESTER 8 COURSES\Thesis\Thesis Project\models\multiclass_model_25_epochs.keras"

model.save(MODEL_25_PATH)

print("25 Epoch Model Saved Successfully")
print("Saved At:", MODEL_25_PATH)
# ==========================================
# LOAD 25 EPOCH MODEL
# ==========================================

from tensorflow.keras.models import load_model

MODEL_25_PATH = r"D:\SEMESTER 8 COURSES\Thesis\Thesis Project\models\multiclass_model_25_epochs.keras"

model_25 = load_model(MODEL_25_PATH)

print("25 Epoch Model Loaded Successfully")
# ==========================================
# 25 EPOCH ACCURACY PLOT
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(history_25.history["accuracy"], label="Training Accuracy")

plt.plot(history_25.history["val_accuracy"], label="Validation Accuracy")

plt.title("25 Epoch Model Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.grid(True)

plt.show()
# ==========================================
# 25 EPOCH LOSS PLOT
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(history_25.history["loss"], label="Training Loss")

plt.plot(history_25.history["val_loss"], label="Validation Loss")

plt.title("25 Epoch Model Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.show()
# ==========================================
# 25 EPOCH VALIDATION ACCURACY / VALIDATION LOSS
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(history_25.history["val_accuracy"], label="Validation Accuracy")

plt.plot(history_25.history["val_loss"], label="Validation Loss")

plt.title("25 Epoch Validation Accuracy and Validation Loss")

plt.xlabel("Epoch")

plt.ylabel("Value")

plt.legend()

plt.grid(True)

plt.show()
# ==========================================
# 25 EPOCH CONFUSION MATRIX
# ==========================================

test_data.reset()

pred_probs_25 = model_25.predict(test_data)

y_pred_25 = np.argmax(pred_probs_25, axis=1)

y_true = test_data.classes

class_names = list(test_data.class_indices.keys())

cm_25 = confusion_matrix(y_true, y_pred_25)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm_25,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title("25 Epoch Confusion Matrix")

plt.xlabel("Predicted Label")

plt.ylabel("Actual Label")

plt.show()
# ==========================================
# 25 EPOCH TP, TN, FP, FN PER CLASS
# ==========================================

tp_list = []
tn_list = []
fp_list = []
fn_list = []

for i, class_name in enumerate(class_names):

    TP = cm_25[i, i]

    FP = cm_25[:, i].sum() - TP

    FN = cm_25[i, :].sum() - TP

    TN = cm_25.sum() - (TP + FP + FN)

    tp_list.append(TP)
    tn_list.append(TN)
    fp_list.append(FP)
    fn_list.append(FN)

x = np.arange(len(class_names))

width = 0.2

plt.figure(figsize=(10, 6))

plt.bar(x - 1.5*width, tp_list, width, label="TP")

plt.bar(x - 0.5*width, tn_list, width, label="TN")

plt.bar(x + 0.5*width, fp_list, width, label="FP")

plt.bar(x + 1.5*width, fn_list, width, label="FN")

plt.xticks(x, class_names)

plt.title("25 Epoch TP, TN, FP, FN per Class")

plt.xlabel("Class")

plt.ylabel("Count")

plt.legend()

plt.grid(axis="y")

plt.show()
# ==========================================
# 25 EPOCH CLASSIFICATION REPORT
# ==========================================

report_25 = classification_report(
    y_true,
    y_pred_25,
    target_names=class_names
)

print("25 Epoch Classification Report")

print(report_25)
# ==========================================
# SHOW 10 RANDOM TEST IMAGES - 25 EPOCH MODEL
# ==========================================

import random
from tensorflow.keras.preprocessing import image

test_dir = os.path.join(FINAL_DATASET_PATH, "test")

all_test_images = []

for class_name in class_names:

    class_folder = os.path.join(test_dir, class_name)

    for img_name in os.listdir(class_folder):

        if img_name.lower().endswith((".png", ".jpg", ".jpeg")):

            all_test_images.append(
                (os.path.join(class_folder, img_name), class_name)
            )

random_samples = random.sample(all_test_images, 10)

plt.figure(figsize=(20, 12))

for i, (img_path, actual_label) in enumerate(random_samples):

    img = image.load_img(img_path, target_size=IMG_SIZE)

    img_array = image.img_to_array(img)

    img_preprocessed = preprocess_input(
        np.expand_dims(img_array, axis=0)
    )

    prediction = model_25.predict(
        img_preprocessed,
        verbose=0
    )[0]

    predicted_index = np.argmax(prediction)

    predicted_label = class_names[predicted_index]

    confidence = prediction[predicted_index] * 100

    plt.subplot(2, 5, i + 1)

    plt.imshow(img, cmap="gray")

    plt.axis("off")

    plt.title(
        f"Actual: {actual_label}\n"
        f"Predicted: {predicted_label}\n"
        f"Confidence: {confidence:.2f}%"
    )

plt.tight_layout()

plt.show()
# ==========================================
# CONTINUE TRAINING TO 50 EPOCHS
# ==========================================

checkpoint_path = r"D:\SEMESTER 8 COURSES\Thesis\Thesis Project\models\best_multiclass_50.keras"

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=7,
        restore_best_weights=True
    ),

    ModelCheckpoint(
        checkpoint_path,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max"
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=3,
        min_lr=1e-7
    )
]

history_50 = model.fit(
    train_data,
    validation_data=val_data,
    initial_epoch=25,
    epochs=50,
    class_weight=class_weights,
    callbacks=callbacks
)
MODEL_50_PATH = r"D:\SEMESTER 8 COURSES\Thesis\Thesis Project\models\multiclass_model_50_epochs.keras"

model.save(MODEL_50_PATH)

print("50 Epoch Model Saved Successfully")
print("Saved At:", MODEL_50_PATH)
from tensorflow.keras.models import load_model

MODEL_50_PATH = r"D:\SEMESTER 8 COURSES\Thesis\Thesis Project\models\multiclass_model_50_epochs.keras"

model = load_model(MODEL_50_PATH)

print("50 Epoch Model Loaded Successfully")
checkpoint_path = r"D:\SEMESTER 8 COURSES\Thesis\Thesis Project\models\best_multiclass_75.keras"

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=7,
        restore_best_weights=True
    ),

    ModelCheckpoint(
        checkpoint_path,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max"
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=3,
        min_lr=1e-7
    )
]

hhistory_75 = model.fit(
    train_data,
    validation_data=val_data,
    initial_epoch=50,
    epochs=75,
    class_weight=class_weights,
    callbacks=callbacks
)
MODEL_PATH = r"D:\SEMESTER 8 COURSES\Thesis\Thesis Project\models\best_multiclass_75.keras"

model = load_model(MODEL_PATH)

print("Best 75 model loaded successfully")
import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
for i, layer in enumerate(model.layers):
    print(i, layer.name, layer.trainable)
# ==========================================
# STRONGER FINE-TUNING SETUP
# ==========================================

for layer in model.layers:
    layer.trainable = True

# Keep Batch Normalization frozen for stability
for layer in model.layers:
    if "bn" in layer.name.lower() or "batch_normalization" in layer.name.lower():
        layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=5e-6),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("Model prepared for stronger fine-tuning")
import gc
import tensorflow as tf
from tensorflow.keras import backend as K

K.clear_session()
gc.collect()

print("Session cleared")
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

MODEL_PATH = r"D:\SEMESTER 8 COURSES\Thesis\Thesis Project\models\best_multiclass_75.keras"

model = load_model(MODEL_PATH, compile=False)

model.compile(
    optimizer=Adam(learning_rate=5e-6),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("Model reloaded and recompiled")
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

FINAL_DATASET_PATH = r"D:\SEMESTER 8 COURSES\Thesis\Thesis Project\Dataset\final"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=10,
    zoom_range=0.05,
    horizontal_flip=True
)

val_test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_data = train_datagen.flow_from_directory(
    os.path.join(FINAL_DATASET_PATH, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

val_data = val_test_datagen.flow_from_directory(
    os.path.join(FINAL_DATASET_PATH, "validation"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

labels = train_data.classes

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels),
    y=labels
)

class_weights = dict(enumerate(class_weights))

print("Data generators recreated")
print("Class Weights:", class_weights)
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

checkpoint_path = r"D:\SEMESTER 8 COURSES\Thesis\Thesis Project\models\best_multiclass_100.keras"

callbacks_100 = [
    ModelCheckpoint(
        checkpoint_path,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max"
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4,
        min_lr=1e-7
    )
]

history_100 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=25,
    class_weight=class_weights,
    callbacks=callbacks_100
)


MODEL_100_PATH = r"D:\SEMESTER 8 COURSES\Thesis\Thesis Project\models\multiclass_model_100_epochs.keras"

model.save(MODEL_100_PATH)

print("100 Epoch Model Saved Successfully")
print("Saved At:", MODEL_100_PATH)
from tensorflow.keras.models import load_model

MODEL_PATH = r"D:\SEMESTER 8 COURSES\Thesis\Thesis Project\models\best_multiclass_100.keras"

model = load_model(MODEL_PATH)

print("Model loaded successfully")
test_loss, test_accuracy = model.evaluate(test_data)

print("Test Accuracy:", test_accuracy)
print("Test Loss:", test_loss)
test_data.reset()

pred_probs = model.predict(test_data)
y_pred = np.argmax(pred_probs, axis=1)
y_true = test_data.classes

class_names = list(test_data.class_indices.keys())

print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title("Final Model Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.show()
# ==========================================
# SELECT IMAGE FROM COMPUTER
# ==========================================

from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Hide tkinter root window
Tk().withdraw()

# Open file picker
img_path = askopenfilename(
    title="Select Chest X-Ray Image",
    filetypes=[
        ("Image Files", "*.png *.jpg *.jpeg *.bmp")
    ]
)

print("Selected Image:")
print(img_path)
import os
import shutil

OLD_PATH = r"D:\SEMESTER 8 COURSES\Thesis\Thesis Project\Dataset\final"
NEW_PATH = r"D:\SEMESTER 8 COURSES\Thesis\Thesis Project\Dataset\final_multiclass_effnetb2"

classes = ["Normal", "Pneumonia", "Tuberculosis"]
splits = ["train", "validation", "test"]

for split in splits:
    for cls in classes:
        old_folder = os.path.join(OLD_PATH, split, cls)
        new_folder = os.path.join(NEW_PATH, split, cls)

        os.makedirs(new_folder, exist_ok=True)

        for img in os.listdir(old_folder):
            src = os.path.join(old_folder, img)
            dst = os.path.join(new_folder, img)

            if not os.path.exists(dst):
                shutil.copy2(src, dst)

print("New multiclass folder created successfully.")
for split in splits:
    print("\n", split.upper())
    for cls in classes:
        path = os.path.join(NEW_PATH, split, cls)
        print(cls, ":", len(os.listdir(path)))
import numpy as np
import tensorflow as tf

from tensorflow.keras.applications import EfficientNetB2
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

IMG_SIZE = (260, 260)
BATCH_SIZE = 32
DATASET_PATH = NEW_PATH
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=15,
    zoom_range=0.10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True
)

val_test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_data = train_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

val_data = val_test_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "validation"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

test_data = val_test_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "test"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

print(train_data.class_indices)
labels = train_data.classes

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels),
    y=labels
)

class_weights = dict(enumerate(class_weights))
print("Class Weights:", class_weights)
base_model = EfficientNetB2(
    weights="imagenet",
    include_top=False,
    input_shape=(260, 260, 3)
)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
x = Dense(512, activation="relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

output = Dense(3, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()
MODEL_PATH = r"D:\SEMESTER 8 COURSES\Thesis\Thesis Project\models\best_multiclass_effnetb2_260.keras"

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=7,
        restore_best_weights=True
    ),

    ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max"
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=3,
        min_lr=1e-7
    )
]
history1 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    class_weight=class_weights,
    callbacks=callbacks
)
base_model.trainable = True

for layer in base_model.layers[:-100]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history2 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=40,
    class_weight=class_weights,
    callbacks=callbacks
)
# ==========================================
# COMPARE ALL SAVED MODELS
# ==========================================

import os
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import classification_report, confusion_matrix

# Paths
MODEL_DIR = r"D:\SEMESTER 8 COURSES\Thesis\Thesis Project\models"
DATASET_PATH = r"D:\SEMESTER 8 COURSES\Thesis\Thesis Project\Dataset\final"

# Saved models
model_files = [
    "best_multiclass_10.keras",
    "best_multiclass_50.keras",
    "best_multiclass_75.keras",
    "best_multiclass_100.keras",
    "best_multiclass_effnetb2_260.keras",
    "multiclass_model_10_epochs.keras",
    "multiclass_model_25_epochs.keras",
    "multiclass_model_50_epochs.keras",
    "multiclass_model_75_epochs.keras",
    "multiclass_model_100_epochs.keras"
]

# Data generator
val_test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

# Test data for old EfficientNetB0 models
test_data_224 = val_test_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "test"),
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

# Test data for EfficientNetB2 model
test_data_260 = val_test_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "test"),
    target_size=(260, 260),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

results = []

for model_file in model_files:
    model_path = os.path.join(MODEL_DIR, model_file)

    if not os.path.exists(model_path):
        print("Model not found:", model_file)
        continue

    print("\n====================================")
    print("Testing model:", model_file)
    print("====================================")

    model = tf.keras.models.load_model(model_path)

    # Choose correct image size
    if "effnetb2_260" in model_file:
        data = test_data_260
    else:
        data = test_data_224

    data.reset()

    loss, acc = model.evaluate(data, verbose=0)

    data.reset()
    pred_probs = model.predict(data, verbose=0)
    pred_classes = np.argmax(pred_probs, axis=1)

    true_classes = data.classes
    class_names = list(data.class_indices.keys())

    report = classification_report(
        true_classes,
        pred_classes,
        target_names=class_names,
        output_dict=True
    )

    cm = confusion_matrix(true_classes, pred_classes)

    results.append({
        "Model": model_file,
        "Test Accuracy": acc,
        "Test Loss": loss,
        "Normal Precision": report["Normal"]["precision"],
        "Normal Recall": report["Normal"]["recall"],
        "Pneumonia Precision": report["Pneumonia"]["precision"],
        "Pneumonia Recall": report["Pneumonia"]["recall"],
        "Tuberculosis Precision": report["Tuberculosis"]["precision"],
        "Tuberculosis Recall": report["Tuberculosis"]["recall"],
        "Macro F1": report["macro avg"]["f1-score"],
        "Weighted F1": report["weighted avg"]["f1-score"]
    })

    print("Accuracy:", acc)
    print("Loss:", loss)
    print("\nClassification Report:")
    print(classification_report(
        true_classes,
        pred_classes,
        target_names=class_names
    ))

    print("Confusion Matrix:")
    print(cm)

# Convert results to table
results_df = pd.DataFrame(results)

# Sort by accuracy
results_df = results_df.sort_values(by="Test Accuracy", ascending=False)

print("\n\n========== FINAL MODEL COMPARISON ==========")
display(results_df)

print("\nBest model by accuracy:")
print(results_df.iloc[0]["Model"])