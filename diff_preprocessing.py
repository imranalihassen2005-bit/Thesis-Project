import os
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image

test_normal_dir = r"D:\SEMESTER 8 COURSES\Thesis\Thesis Project\Dataset\final\test\Normal"
img_name = os.listdir(test_normal_dir)[0]
img_path = os.path.join(test_normal_dir, img_name)

# Method 1
img1 = Image.open(img_path).convert('RGB').resize((224, 224))
arr1 = np.array(img1, dtype=np.float32)

# Method 2
img2 = image.load_img(img_path, target_size=(224, 224))
arr2 = image.img_to_array(img2)

print("Arr1 shape:", arr1.shape, "dtype:", arr1.dtype)
print("Arr2 shape:", arr2.shape, "dtype:", arr2.dtype)

diff = np.abs(arr1 - arr2)
print("Max diff:", np.max(diff))
print("Mean diff:", np.mean(diff))

# Let's check if the interpolation is the ONLY difference
img3 = image.load_img(img_path, target_size=(224, 224), interpolation='bicubic')
arr3 = image.img_to_array(img3)

diff3 = np.abs(arr1 - arr3)
print("Max diff with bicubic:", np.max(diff3))
print("Mean diff with bicubic:", np.mean(diff3))
