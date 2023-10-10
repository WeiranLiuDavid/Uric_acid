# This is the code for feature analysis module
# It relies on machine-learning models to predict the label of the sample
import pickle
import numpy as np
from Feature_extraction import feature_extraction
from sklearn.preprocessing import PolynomialFeatures

# The following are the min and max values of the features in the training set
max_red =  168.54322595255618
max_green =  201.95131064938008
max_blue =  201.5735152578837
max_gray =  194.97780671460114
max_H =  96.14793636415105
max_S =  81.77619883372022
max_MSE =  210.4655289406963
min_red =  133.16826133725868
min_green =  182.78939025080678
min_blue =  188.890788654249
min_gray =  181.0992470135311
min_H =  79.13338617448905
min_S =  40.46877653852687
min_MSE =  18.912832871012753

def linear_regression(feature_array):
    R, G, B, gray, H, S, MSE = feature_array
    model = pickle.load(open('models/linear_regression.sav', 'rb'))
    prediction = model.predict(np.array([S]).reshape(1, -1))
    return prediction.reshape(-1)


def multiple_regression(feature_array):
    R, G, B, gray, H, S, MSE = feature_array
    model = pickle.load(open('models/multiple_regression.sav', 'rb'))
    prediction = model.predict(np.array([R, G, B, gray, S, MSE]).reshape(1, -1))
    return prediction.reshape(-1)


def polynomial_regression(feature_array):
    R, G, B, gray, H, S, MSE = feature_array
    model = pickle.load(open('models/polynomial_regression.sav', 'rb'))
    feature = np.array([H]).reshape(1, -1)
    poly = PolynomialFeatures(degree=5)
    feature = poly.fit_transform(feature)
    prediction = model.predict(feature)
    return prediction.reshape(-1)


def multiple_poly_regression(feature_array):
    R, G, B, gray, H, S, MSE = feature_array
    model = pickle.load(open('models/multi_poly_regression.sav', 'rb'))
    feature = np.array([R, G, gray, S, MSE]).reshape(1, -1)
    poly = PolynomialFeatures(degree=2)
    feature = poly.fit_transform(feature)
    prediction = model.predict(feature)
    return prediction.reshape(-1)


def SVR(feature_array):
    R, G, B, gray, H, S, MSE = feature_array
    model = pickle.load(open('models/SVR.sav', 'rb'))
    # The SVR model is trained with the scaled features
    # So we need to scale the features before we feed them into the model
    R = (R - min_red) / (max_red - min_red)
    G = (G - min_green) / (max_green - min_green)
    B = (B - min_blue) / (max_blue - min_blue)
    gray = (gray - min_gray) / (max_gray - min_gray)
    H = (H - min_H) / (max_H - min_H)
    S = (S - min_S) / (max_S - min_S)
    MSE = (MSE - min_MSE) / (max_MSE - min_MSE)
    feature = np.array([G, B, gray, S, MSE]).reshape(1, -1)

    prediction = model.predict(feature)
    return prediction.reshape(-1)


def decision_tree(feature_array):
    R, G, B, gray, H, S, MSE = feature_array
    model = pickle.load(open('models/decision_tree.sav', 'rb'))
    # The decision tree model is trained with the scaled features
    G = (G - min_green) / (max_green - min_green)
    B = (B - min_blue) / (max_blue - min_blue)
    H = (H - min_H) / (max_H - min_H)
    S = (S - min_S) / (max_S - min_S)
    MSE = (MSE - min_MSE) / (max_MSE - min_MSE)
    feature = np.array([G, B, H, S, MSE]).reshape(1, -1)
    prediction = model.predict(feature)
    return prediction.reshape(-1)


def random_forest(feature_array):
    R, G, B, gray, H, S, MSE = feature_array
    model = pickle.load(open('models/random_forest.sav', 'rb'))
    # The random forest model is trained with the scaled features
    # So we need to scale the features before we feed them into the model
    gray = (gray - min_gray) / (max_gray - min_gray)
    H = (H - min_H) / (max_H - min_H)
    S = (S - min_S) / (max_S - min_S)
    feature = np.array([gray, H, S]).reshape(1, -1)
    prediction = model.predict(feature)
    return prediction.reshape(-1)


def main(model='decision_tree', image_directory='Images/Artificial_Saliva/train/0_1.jpg'):
    if model == 'linear_regression':
        return linear_regression(feature_extraction(image_directory))
    elif model == 'multiple_regression':
        return multiple_regression(feature_extraction(image_directory))
    elif model == 'polynomial_regression':
        return polynomial_regression(feature_extraction(image_directory))
    elif model == 'multiple_poly_regression':
        return multiple_poly_regression(feature_extraction(image_directory))
    elif model == 'SVR':
        return SVR(feature_extraction(image_directory))
    elif model == 'decision_tree':
        return decision_tree(feature_extraction(image_directory))
    elif model == 'random_forest':
        return random_forest(feature_extraction(image_directory))
    else:
        raise ValueError('Please enter a valid model name')


if __name__ == '__main__':
    print(main(model='decision_tree', image_directory='Images/Artificial_Saliva/train/0_1.jpg'))

