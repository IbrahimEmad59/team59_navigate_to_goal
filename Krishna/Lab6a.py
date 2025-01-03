#!/usr/bin/env python3
import cv2
import numpy as np
import os
import csv
import math
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from skimage.feature import hog, local_binary_pattern
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
import pickle

# Directory where images are located
imageDirectory = '/home/ibrahim/ros2_ws/src/team59_navigate_to_goal/Krishna/2024F_imgs/'
imageType = '.png'

def crop_image_to_sign(img, debug=False):
    """Improved cropping function to focus on the sign with better masking and contour selection."""
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define color ranges in HSV for blue, green, and red
    color_ranges = {
        "blue": ((100, 10, 10), (160, 230, 230)),
        "green": ((40, 100, 50), (90, 230, 230)),
        "red": ((0, 175, 175), (20, 230, 230)),
        "red2": ((155, 100, 100), (180, 230, 230))
    }

    # Create masks for each color and combine them
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for color, (lower, upper) in color_ranges.items():
        lower_bound = np.array(lower, dtype=np.uint8)
        upper_bound = np.array(upper, dtype=np.uint8)
        current_mask = cv2.inRange(hsv, lower_bound, upper_bound)
        mask |= current_mask

        # Debug: Display individual color masks
        if debug:
            cv2.imshow(f"Mask for {color}", current_mask)

    # Apply morphological operations (close and open) to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Close small holes inside the objects
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # Remove noise around the objects

    # Apply Gaussian blur to smooth the mask
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Debug: Display cleaned mask
    if debug:
        cv2.imshow("Cleaned Mask with Morphological Operations and Blurring", mask)
        cv2.waitKey(0)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the contour closest to the center of the image
        img_center = (img.shape[1] // 2, img.shape[0] // 2)  # (x_center, y_center)
        closest_contour = None
        min_distance = float('inf')

        for contour in contours:
            # Calculate the center of the contour
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                distance = math.sqrt((cX - img_center[0])**2 + (cY - img_center[1])**2)
                if distance < min_distance:
                    min_distance = distance
                    closest_contour = contour

        # If a valid contour is found, crop the image around it
        if closest_contour is not None and cv2.contourArea(closest_contour) >= 500:  # Area threshold
            x, y, w, h = cv2.boundingRect(closest_contour)
            cropped_img = img[y:y+h, x:x+w]

            # Debugging: Display cropped image
            if debug:
                cv2.imshow("Cropped Image", cropped_img)
                cv2.waitKey(0)

            # Resize to uniform dimensions (e.g., 64x64 for training)
            processed_image = cv2.resize(cropped_img, (64, 64), interpolation=cv2.INTER_AREA)

            return processed_image

    return None  # Return None if no valid contour is found

def convert_to_cmyk(image):
    """Convert an image from BGR to simulated CMYK color space."""
    bgr = image.astype(np.float32) / 255.0  # Normalize BGR values to [0, 1]
    k = 1 - np.max(bgr, axis=2)  # Key (Black channel)
    c = (1 - bgr[..., 2] - k) / (1 - k + 1e-8)  # Cyan channel
    m = (1 - bgr[..., 1] - k) / (1 - k + 1e-8)  # Magenta channel
    y = (1 - bgr[..., 0] - k) / (1 - k + 1e-8)  # Yellow channel
    
    cmyk = np.stack((c, m, y, k), axis=2)
    return cmyk.astype(np.uint8)

def extract_HOG(image):
    winSize = (64, 64) 
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 9
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    return hog.compute(image).flatten()

def extract_features(image, debug=False):
    """Extract HOG features, CMYK histograms, LBP features, and directional features."""
    # Validate input image
    if image is None or image.size == 0:
        raise ValueError("Invalid input image: Image is None or empty")
    
    # Resize the image to a standard size (if not already resized)
    resized_img = cv2.resize(image, (64, 64))

    # Debugging: display resized image 
    if debug:
        cv2.imshow("Resized Image", resized_img)
        cv2.waitKey(0)  # Wait for a key press to display each image

    # Convert to CMYK and grayscale for HOG
    cmyk_img = convert_to_cmyk(resized_img)
    gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

    # Convert to CMYK and calculate histograms for each channel
    hist_c = extract_HOG(cv2.resize(cmyk_img[..., 0], (64, 64)))
    hist_m = extract_HOG(cv2.resize(cmyk_img[..., 1], (64, 64)))
    hist_y = extract_HOG(cv2.resize(cmyk_img[..., 2], (64, 64)))
    hist_k = extract_HOG(cv2.resize(cmyk_img[..., 3], (64, 64)))

    gray_features = extract_HOG(cv2.resize(gray_img, (64, 64)))

    # Concatenate all features: HOG, CMYK histograms, and gray feature
    features = np.concatenate((gray_features, hist_c, hist_m, hist_y, hist_k)).astype(np.float32)
    return features
    
# Load the labels
with open(imageDirectory + 'labels.txt', 'r') as f:
    reader = csv.reader(f)
    lines = list(reader)

# Shuffle data and split into training and test sets
random.shuffle(lines)
train_lines = lines[:math.floor(len(lines)//2)]
test_lines = lines[math.floor(len(lines)//2):]

# Prepare the training data
train_data = []
train_labels = []

for line in train_lines:
    img_path = os.path.join(imageDirectory, line[0] + imageType)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Image not loaded: {img_path}")
        continue

    # Crop the image to focus on the sign
    cropped_img = crop_image_to_sign(img)
    
    # Skip if no valid sign was detected
    if cropped_img is None:
        print(f"No valid sign detected in image: {img_path}")
        continue
    
    # Extract features
    try:
        features = extract_features(cropped_img)
        train_data.append(features)

        # Get the label from the CSV file
        label = int(line[1])
        train_labels.append(label)
    except Exception as e:
        print(f"Error extracting features for {img_path}: {e}")
        continue

# Convert lists to numpy arrays
train_data = np.array(train_data)
train_labels = np.array(train_labels)

# Standardize the feature vectors
scaler = MinMaxScaler().fit(train_data)
train_data = scaler.transform(train_data)

# Prepare the test data
test_data = []
test_labels = []

for line in test_lines:
    img_path = os.path.join(imageDirectory, line[0] + imageType)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Image not loaded: {img_path}")
        continue

    # Crop the image to focus on the sign
    cropped_img = crop_image_to_sign(img)
    
    # Skip if no valid sign was detected
    if cropped_img is None:
        print(f"No valid sign detected in image: {img_path}")
        continue
    
    # Extract features
    try:
        features = extract_features(cropped_img)
        test_data.append(features)

        # Get the label from the CSV file
        label = int(line[1])
        test_labels.append(label)
    except Exception as e:
        print(f"Error extracting features for {img_path}: {e}")
        continue

# Convert lists to numpy arrays
test_data = np.array(test_data)
test_labels = np.array(test_labels)

# Standardize the test data using the same scaler
test_data = scaler.transform(test_data)

selector = SelectKBest(score_func=f_classif, k=8820)
train_data = selector.fit_transform(train_data, train_labels)
test_data = selector.transform(test_data)

# Train an SVM classifier
svm_clf = SVC(kernel='linear', C=100.0)  # Adjust C as needed
svm_clf.fit(train_data, train_labels)

# Evaluate the classifier
predictions = svm_clf.predict(test_data)

# Calculate accuracy and confusion matrix
accuracy = accuracy_score(test_labels, predictions)
conf_matrix = confusion_matrix(test_labels, predictions)

# save the iris classification model as a pickle file
model_pkl_file = "/home/ibrahim/ros2_ws/src/team59_navigate_to_goal/Krishna/sign_classifier_model.pkl"  

with open(model_pkl_file, 'wb') as file:  
    pickle.dump(svm_clf, file)

# Calculate accuracy
print("\n\nTotal accuracy: {:.2f}%".format(accuracy * 100))
print("Confusion Matrix:\n", conf_matrix)