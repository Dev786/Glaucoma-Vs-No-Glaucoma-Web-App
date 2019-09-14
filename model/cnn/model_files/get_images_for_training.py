import cv2
import os
import numpy as np


def get_features_and_labels():
    glucauma = "../images/glaucoma"
    non_glucauma = "../images/non_glaucoma"
    positive_images = os.listdir(glucauma)
    negative_images = os.listdir(non_glucauma)
    features = []
    labels = []

    for image_name in positive_images:
        image_path = glucauma+"/"+image_name
        image = cv2.imread(image_path)
        reshaped_image = np.array(cv2.resize(
            image, (128, 128), interpolation=cv2.INTER_LINEAR))
        features.append(reshaped_image)
        labels.append([0, 1])

    for image_name in negative_images:
        image_path = non_glucauma+"/"+image_name
        image = cv2.imread(image_path)
        reshaped_image = np.array(cv2.resize(
            image, (128, 128), interpolation=cv2.INTER_LINEAR))
        features.append(reshaped_image)
        labels.append([1, 0])

    features = np.array(features)
    labels = np.array(labels)

    random_indices = np.random.choice(labels.shape[0], labels.shape[0])
    return features[random_indices], labels[random_indices]
