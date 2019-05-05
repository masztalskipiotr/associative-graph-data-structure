import math
import numpy as np
import pandas as pd
from sklearn import datasets
from scipy import stats


def calculate_atr2atr_weight(a1, a2, r):
    return 1 - (abs(a1 - a2) / r)


def calculate_atr2obs_weight(atr_count):
    return 1 / atr_count


def get_similarity(agds, idx=None, test_features=None):
    similarity_rating = np.zeros(150)

    for k, feature in enumerate(agds[:-1]):
        i = 0
        attr_idx = 0
        attributes = [feature[0] for feature in feature.keys()]

        if idx:
            attr_similarity = np.zeros(len(attributes))
            for attr, observations in feature.items():
                if idx in observations:
                    attr_idx = i
                    break
                i += 1
        elif test_features:
            attr_similarity = np.zeros(len(attributes)+1)
            min_diff = math.inf
            for i, at in enumerate(attributes):
                diff = abs(test_features[k] - at)
                if diff < min_diff:
                    min_diff = diff
                # the attributes list is sorted so if the diff > min that
                # means the previous element was the minimum
                elif diff > min_diff:
                    attr_idx = i-1
                    attributes.insert(attr_idx, test_features[k])
                    break

        # assign self similarity
        attr_similarity[attr_idx] = 1
        # print(attr_similarity, 'length: ', len(attr_similarity))

        # calculate similarity of the lower part of the attribute table
        for j in range(attr_idx-1, -1, -1):
            weight = calculate_atr2atr_weight(a1=attributes[j], a2=attributes[j+1],
                                              r=max(attributes)-min(attributes))
            attr_similarity[j] = attr_similarity[j+1] * weight

        # calculate similarity of the higher part of the attribute table
        for j in range(attr_idx+1, len(attributes)):
            weight = calculate_atr2atr_weight(a1=attributes[j], a2=attributes[j-1],
                                              r=max(attributes)-min(attributes))
            attr_similarity[j] = attr_similarity[j-1] * weight

        if test_features:
            attr_similarity = np.delete(attr_similarity, attr_idx)

        # last feature is the class, and there are no connections between classes
#         if k == len(agds)-1:
#             attr_similarity = [0 for x in attr_similarity]
#             attr_similarity[attr_idx] = 1

        # update similarity rating for observations
        for n, observations in enumerate(feature.values()):
            for obs in observations:
                similarity_rating[obs] += attr_similarity[n] * 1/5

        # print(attr_similarity, 'length: ', len(attr_similarity))
    return similarity_rating


def get_topk_similar(similarity, k):
    similar = np.argsort(similarity)
    similar = np.flip(similar)
    return similar[:k]


def classify(agds, features, k):
    ''' Classifies an observation of given features using
    a k nearest neigbours approach
    '''
    similarity = get_similarity(agds, test_features=features)
    topk = get_topk_similar(similarity, k)
    neighbour_classes = []
    for obs in topk:
        for key, value in agds[-1].items():
            if obs in value:
                neighbour_classes.append(key[0])
    # print(neighbour_classes)
    return stats.mode(neighbour_classes).mode


if __name__ == "__main__":

    iris = datasets.load_iris()
    features = iris.data
    labels = iris.target

    agds = []

    for i in range(len(features[1, :])):
        unique, counts = np.unique(features[:, i], return_counts=True)
        # zips an array of unique features with an array of their counts
        features_tuple = zip(unique, counts)
        # creates an array of arrays containing indices of feature occurances
        # (observations)
        indices = [np.where(features[:, i] == el) for el in unique]
        indices = [i[0] for i in indices]
        # connects features and their counts with their corresponding observations
        feature_agds = zip(features_tuple, indices)
        agds.append(dict(feature_agds))

    # complete the agds with a labels dictionary
    unique, counts = np.unique(labels, return_counts=True)
    agds.append(dict(zip(zip(unique, counts),
                         [np.where(labels == el)[0] for el in unique])))

    similarity = get_similarity(agds, idx=120)
    topk = get_topk_similar(similarity, 20)
    print(topk)
