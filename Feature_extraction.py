# This is the code for feature extraction module
# The module will only take in images that are cut and calibrated by the first two modules
import cv2
import numpy as np


def feature_extraction(image_directory='Images/Artificial_Saliva/train/0_1.jpg'):
    image = cv2.imread(image_directory)
    B, G, R = cv2.split(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(B)
    x, y = mask.shape
    cv2.circle(mask, (x // 2, y // 2), x // 2, 255, -1)
    hsi = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    H, S, I = cv2.split(hsi)
    mean_B = cv2.mean(B, mask)[0]
    mean_G = cv2.mean(G, mask)[0]
    mean_R = cv2.mean(R, mask)[0]
    mean_gray = cv2.mean(gray, mask)[0]
    mean_H = cv2.mean(H, mask)[0]
    mean_S = cv2.mean(S, mask)[0]
    average_color = np.mean(image[mask > 0], axis=0)
    difference = np.subtract(image, average_color)
    squared_difference = np.square(difference)
    mse = np.mean(squared_difference[mask > 0])
    return [mean_R, mean_G, mean_B, mean_gray, mean_H, mean_S, mse]

