from sklearn import datasets
from sklearn import metrics
from sklearn import linear_model
from sklearn import model_selection
import matplotlib.pyplot as plt
import numpy as np
import random


def main():
    plt.close("all")
    digits = datasets.load_digits()
    print(digits.DESCR)
    print()

    targets = set(digits.target)
    print(targets)

    for digit in targets:
        plt.figure()
        for i in range(3):
            for j in range(3):
                plt.subplot(3, 3, i * 3 + j + 1)
                index = random.randint(0, sum(digits.target == digit) - 1)
                plt.imshow(digits.data[digits.target == digit][index].reshape(8, 8))

    train_data, test_data, train_target, test_target = model_selection.train_test_split(digits.data, digits.target,
                                                                                        test_size=0.2)
    clf = linear_model.Perceptron()
    clf.fit(train_data, train_target)
    prediction = clf.predict(test_data)

    score = metrics.accuracy_score(test_target, prediction)
    print("Perceptron accuracy score: ", score)

    plt.figure()
    for i in range(3):
        for j in range(3):
            plt.subplot(3, 3, i * 3 + j + 1)
            plt.imshow(clf.coef_.reshape(10, 8, 8)[i * 3 + j, :, :])

    plt.figure()
    for i in range(3):
        for j in range(3):
            plt.subplot(3, 3, i * 3 + j + 1)
            index = random.randint(0, sum(prediction != test_target) - 1)
            plt.imshow(test_data[prediction != test_target][index].reshape(8, 8))

    plt.show()


main()
