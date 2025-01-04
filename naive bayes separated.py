# This code is added separately because this one did not get used at all

import pandas as pd #type: ignore
import numpy as np #type: ignore

def probability(x, y):
    num_features = x.shape[1]
    classes = np.unique(y)
    class_probabilities = {}

    for c in classes:
        class_probabilities[c] = []
        for i in range(num_features):
            feature_values = np.unique(x[:, i])
            probabilities = {}
            for value in feature_values:
                probabilities[value] = np.sum((x[:, i] == value) * (y == c)) / np.sum(y == c)
            class_probabilities[c].append(probabilities)
    return class_probabilities

def class_probabilities(x, y):
    classes = np.unique(y)
    p_classes = {}
    class_probabilities_dict = probability(x, y)
    
    for c in classes:
        p_classes[c] = np.sum(y == c) / len(y)

    return p_classes, class_probabilities_dict

def predict(x_test, p_classes, class_probabilities_dict):
    classes = list(p_classes.keys())
    predictions = np.zeros(len(x_test), dtype=int)

    for idx, x_single in enumerate(x_test):
        class_scores = {}
        for c in classes:
            p_c_given_x = p_classes[c]
            for i, value in enumerate(x_single):
                if value in class_probabilities_dict[c][i]:
                    p_c_given_x *= class_probabilities_dict[c][i][value]
                else:
                    p_c_given_x *= 1e-10
            class_scores[c] = p_c_given_x
        predictions[idx] = max(class_scores, key=class_scores.get)

    return predictions

def accuracy(y_true, y_pred):
    correct_predictions = np.sum(y_true == y_pred)
    total_predictions = len(y_true)
    return correct_predictions / total_predictions if total_predictions > 0 else 0.0

if __name__ == "__main__":
    # Toy data
    x = np.array([[1, 2], [1, 3], [1, 4], [2, 5], [2, 1], [3, 2]])
    y = np.array([1, 1, 0, 0, 2, 1])
    p_classes, class_probabilities_dict = class_probabilities(x, y)
    x_test = np.array([[1, 2], [1, 3], [1, 4], [2, 5], [2, 1], [3, 2]])
    y_test = np.array([1, 1, 0, 0, 2, 1])
    predicted_classes = predict(x_test, p_classes, class_probabilities_dict)
    print(f"Predicted classes for the new test cases: {predicted_classes}")
    print(f"Accuracy: {accuracy(y_test, predicted_classes)}")
