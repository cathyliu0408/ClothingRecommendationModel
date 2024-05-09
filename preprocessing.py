# Modify the existing preprocessing script to include the color feature extraction
import pandas as pd
import numpy as np
import cv2

def extract_color_features(image, bins=16, mask=None):
    # Calculate histograms for each channel
    hist_features = []
    for i in range(3):  # RGB channels
        hist = cv2.calcHist([image], [i], mask, [bins], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        hist_features.extend(hist)
    
    # Calculate average color in the masked area
    if mask is not None:
        masked_pixels = image[mask == 255]
        average_color = np.mean(masked_pixels, axis=0)
    else:
        average_color = np.mean(image, axis=(0, 1))
    
    # Concatenate histogram features and average color
    feature_vector = np.concatenate([hist_features, average_color])
    return feature_vector

path = 'data/vali_modified.csv'
df = pd.read_csv(path)

image_path_array = df['image_path'].to_numpy()
x1 = df['x1'].to_numpy().astype(np.float32)
y1 = df['y1'].to_numpy().astype(np.float32)
x2 = df['x2'].to_numpy().astype(np.float32)
y2 = df['y2'].to_numpy().astype(np.float32)

features_list = []

for i in range(len(image_path_array)):
    path = image_path_array[i]
    img = cv2.imread(path)
    if img is None:
        continue
    h, w = img.shape[:2]
    
    # Create a mask for the bounding box
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.rectangle(mask, (int(x1[i] * w), int(y1[i] * h)), (int(x2[i] * w), int(y2[i] * h)), 255, -1)
    
    # Extract color features
    color_features = extract_color_features(img, bins=16, mask=mask)
    features_list.append(color_features)

# Convert the list of features to a DataFrame
feature_df = pd.DataFrame(features_list)
feature_df.to_csv('data/vali_features.csv', index=False)
print(feature_df.head())

