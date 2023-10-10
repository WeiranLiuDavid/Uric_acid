import numpy as np
import cv2
import json
from ROI_detection import ROI_Detection
from sklearn.linear_model import LinearRegression

# The json_reader is the function to read a json file containing the coordinates of the circles
# The json file is generated by the VIA annotation tool
# The ROI detection module can work in most cases, but to achieve a high accuracy, we recommend to use the VIA annotation tool to annotate the ROI
def json_reader(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
        for data_value in data.values():
            regions = data_value['regions']
            circle_list = []
            for i in range(24):
                shape = regions[i]['shape_attributes']
                cx, cy, r = shape['cx'], shape['cy'], int(shape['r'])
                circle_list.append([cx, cy, r])
            center_circle = regions[24]['shape_attributes']
            cx, cy, r = center_circle['cx'], center_circle['cy'], int(center_circle['r'])
            return circle_list, [cx, cy, r]


def BGR_reader(image, circle_list, center_circle):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    color_list = []
    for i in range(24):
        mask = np.zeros_like(gray)
        cv2.circle(mask,  (circle_list[i][0], circle_list[i][1]), circle_list[i][2], 255, -1)
        mean = cv2.mean(image, mask)[:3]
        color_list.append(mean)
    return color_list


def color_calibration(image_directory='Full_image/0/0_1.jpg', json_file='Full_image/Coordinates.json', use_coordinates=True):
    if use_coordinates:
        circle_list, center_circle = json_reader(json_file)
    else:
        circle_list, center_circle = ROI_Detection(image_directory)
    image = cv2.imread(image_directory)
    color_list = BGR_reader(image, circle_list, center_circle)
    template = np.load('images/template.npy')[:24]
    color_calibration_model = LinearRegression()
    color_calibration_model.fit(color_list, template)
    x, y, r = center_circle
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = np.zeros_like(gray)
    # We use the lower half the center circle to extract features
    cv2.circle(mask, (x, y + (r // 2)), r // 2, 255, -1)
    masked_and_cropped = cv2.bitwise_and(image, image, mask=mask)[y:y + r, x - r // 2:x + r // 2, :]
    h, w, _ = masked_and_cropped.shape
    for i in range(h):
        for j in range(w):
            if masked_and_cropped[i, j].all() == 0:
                pass
            else:
                masked_and_cropped[i, j] = color_calibration_model.predict([masked_and_cropped[i, j]])
    return masked_and_cropped
