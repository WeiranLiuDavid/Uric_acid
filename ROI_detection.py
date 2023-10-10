# This is the code to detect the ROI in the image
import cv2
import numpy as np


def ROI_Detection(image_directory='Full_image/0/0_1.jpg'):
    image = cv2.imread(image_directory)
    # First we cut the image
    image = image[540:1990, 240:1740]
    # The next step is to use the threshold to detect the ROI
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Then use the morphology to remove the noise
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    # Then use the morphology to fill the holes
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    # Now we find the contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))
    # Now we calculate the area of each contour, the center circle should have the largest area
    # The area of the standard-color-circles is between 16000 and 21000
    area_list = []
    for contour in contours:
        area_list.append(cv2.contourArea(contour))
    center_circle = contours[area_list.index(max(area_list))]
    circle_list = []
    for area in area_list:
        if 16000 < area < 21000:
            circle_list.append(contours[area_list.index(area)])
    if len(circle_list) != 24:
        print('The number of circles is not 24')
    else:
        # We will use min-enclosing circle to get the center and radius of each circle
        center_list = []
        radius_list = []
        for circle in circle_list:
            (x, y), radius = cv2.minEnclosingCircle(circle)
            center = (int(x), int(y))
            radius = int(radius)
            center_list.append(center)
            radius_list.append(radius)
        (x, y), radius = cv2.minEnclosingCircle(center_circle)
        center = (int(x), int(y))
        radius = int(radius)
    # Now we draw the center circle and the standard-color-circles
    cv2.circle(image, center, radius, (0, 255, 0), 2)
    for i in range(24):
        cv2.circle(image, center_list[i], radius_list[i], (0, 255, 0), 2)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    return circle_list, center_circle


if __name__ == '__main__':
    ROI_Detection()


