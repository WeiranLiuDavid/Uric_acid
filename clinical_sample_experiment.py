# This is the code for testing clinical samples
# We recommend to test 10 times for each sample to make sure the result is accurate
from Feature_analysis import linear_regression, multiple_regression, polynomial_regression, multiple_poly_regression, SVR, decision_tree, random_forest

from Feature_extraction import feature_extraction
import numpy as np
import os


def clinical_sample_test():
    # we will be using decision tree model as an example
    # you can change the model to any model you want
    samples = os.listdir('Images/Clinical_Samples')
    if '.DS_Store' in samples:
        samples.remove('.DS_Store')
    # sort the samples
    samples.sort(key=lambda x: int(x.split('_')[0]))
    sample_dict = {}
    for sample in samples:
        sample_id = sample.split('_')[0]
        if sample_id not in sample_dict:
            sample_dict[sample_id] = [sample]
        else:
            sample_dict[sample_id].append(sample)
    for id in sample_dict:
        print(id)
        sample_list = sample_dict[id]
        feature_list = []
        for sample in sample_list:
            feature = feature_extraction('Images/Clinical_Samples/' + sample)
            feature_list.append(feature)
        feature_list = np.array(feature_list)
        average_feature = np.mean(feature_list, axis=0)
        # We will pick the five samples that are the closest to the average
        # And then use the average of these five samples to predict the result
        distance_list = []
        for feature in feature_list:
            distance_list.append(np.linalg.norm(feature - average_feature))
        distance_list = np.array(distance_list)
        average = np.mean(feature_list[np.argsort(distance_list)][:5], axis=0)
        # Now we have the average feature, we can use it to predict the result
        prediction = decision_tree(average)
        print(prediction)


if __name__ == '__main__':
    clinical_sample_test()