import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import tree
from sklearn import metrics
from sklearn import neighbors
from sklearn import linear_model
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
import time

data = pd.read_csv("product_images.csv")


# Waleed Akhtar
# R00186742


def preprocessing():
    print("-------------------------------- TASK 1 --------------------------------")
    label = data["label"]
    sneaker = data["label"] == 0  # Getting the sneakers from the spreadsheet
    sneakerCount = data["label"][sneaker].count()
    ankleBoot = data["label"] == 1  # Getting the ankle boots from the spreadsheet
    ankleBootCount = data["label"][ankleBoot].count()

    feature = data.drop("label", axis=1)  # Separating the label column from the spreadsheet to only get the pixel values

    print(f"Sneaker Count: {sneakerCount}")
    print(f"Ankle Boot Count: {ankleBootCount}")

    # Get the sneaker data from the spreadsheet and print it
    plt.imshow(np.array(feature.iloc[label[sneaker].index[0]]).reshape(28, 28))
    plt.show()

    # Get the ankle boot data from the spreadsheet and print it
    plt.imshow(np.array(feature.iloc[label[ankleBoot].index[0]]).reshape(28, 28))
    plt.show()


def evaluation(classifer, sampleValue):
    print("-------------------------------- TASK 2 --------------------------------")

    label = data["label"]
    feature = data.drop("label", axis=1)

    # Paraterising the data
    paraterisedFeature = feature.sample(sampleValue)  # Passes in values when the function is called
    parateristedLabel = label[paraterisedFeature.index]

    # Creating lists to store times
    fitTimeList = []
    predictTimeList = []
    accuracyList = []

    kf = model_selection.KFold()
    for train, test in kf.split(paraterisedFeature, parateristedLabel):
        featureTrainFold = paraterisedFeature.iloc[train]
        featureTestFold = paraterisedFeature.iloc[test]
        labelTrainFold = parateristedLabel.iloc[train]
        labelTestFold = parateristedLabel.iloc[test]

        # I start a timer before running the classifier to calculate the processing time it took
        startTimeFit = time.time()
        classifer.fit(featureTrainFold, labelTrainFold)  # The classifier it is running is passed when the function is
        # called
        finishTimeFit = time.time()  # Get the time when it finishes running the classifier
        fitTimeCalculation = finishTimeFit - startTimeFit  # Then I calculate the difference between the two times and
        # add it to my list
        fitTimeList.append(fitTimeCalculation)

        # I do the same with predict
        startTimePredict = time.time()
        prediction = classifer.predict(featureTestFold)
        finishTimePredict = time.time()
        predicttimeCalculation = finishTimePredict - startTimePredict
        predictTimeList.append(predicttimeCalculation)

        # I get the accuracy score and then store it in my list
        accuracy = metrics.accuracy_score(labelTestFold, prediction)
        accuracyList.append(accuracy)

        # Running the confusion matrix
        matrix = confusion_matrix(labelTestFold, prediction)

        # Getting the averages
        averageFitTime = sum(fitTimeList) / len(fitTimeList)
        averagePredictiomTime = sum(predictTimeList) / len(predictTimeList)
        averageAccuracy = sum(accuracyList) / len(accuracyList)

        print(f"Classifier used: {classifer}")
        print(f"Sample Sized used: {sampleValue}")
        print(f"Matrix: {matrix}")

        print("-------------------------------- Times --------------------------------")
        print("Training Time")
        print(f"Training Time (Minimum): {min(fitTimeList)}")
        print(f"Training Time (Maximum): {max(fitTimeList)}")
        print(f"Training Time (Average): {averageFitTime}")

        print("Prediction Time")
        print(f"Prediction Time (Minimum): {min(predictTimeList)}")
        print(f"Prediction Time (Maximum): {max(predictTimeList)}")
        print(f"Prediction Time (Average): {averagePredictiomTime}")

        print("Prediction Accuracy")
        print(f"Prediction Accuracy Time (Minimum): {min(accuracyList)}")
        print(f"Prediction Accuracy Time (Maximum): {max(accuracyList)}")
        print(f"Prediction Accuracy Time (Average): {averageAccuracy}")

        return sampleValue, averageAccuracy, averagePredictiomTime, averageFitTime


def perceptron():
    print("-------------------------------- TASK 3 --------------------------------")
    classiferPerceptron = linear_model.Perceptron()  # I set the classifier to Perceptron
    sampleValueList = []  # Create an empty list to store the sample values
    averageGitTimeList = []  # An empty list to store the average training times
    averagePredictionList = []  # Empty list to store average prediction times
    averageAccuracyList = []  # Empty list to store average accuracy
    sampleList = [200, 500, 1000, 2000, 5000]  # my various number of samples

    # For loop which goes over the list of samples and runs task 2 over and over again with my different sample values
    for sampleValue in sampleList:
        sampleValue, averageAccuracy, averagePredictiomTime, averageFitTime = evaluation(classiferPerceptron,
                                                                                         sampleValue)
        # I add all the values to my lists
        sampleValueList.append(sampleValue)
        averageGitTimeList.append(averageFitTime)
        averagePredictionList.append(averagePredictiomTime)
        averageAccuracyList.append(averageAccuracy)

    # Getting the average values from the times
    calculatedAverageFitTime = sum(averageGitTimeList) / len(averageGitTimeList)
    caculatedAveragePrediction = sum(averagePredictionList) / len(averagePredictionList)
    calculatedAverageAccuracy = sum(averageAccuracyList) / len(averageAccuracyList)

    print("\n\nMean prediction accuracy for Perceptron")
    print(f"Average Training Time: {calculatedAverageFitTime}")
    print(f"Average Prediction Time: {caculatedAveragePrediction}")
    print(f"Average Accuracy: {calculatedAverageAccuracy}")
    print(f"Best Accuracy: {max(averageAccuracyList)}")

    # Then I plot them on a line diagram
    xAxis = sampleValueList
    yAxis = averageGitTimeList

    plt.title("Line Plot between Input Data Vs Runtime (Perceptron)")
    plt.xlabel("Sample values")
    plt.ylabel("Average Accuracy Times")
    plt.plot(xAxis, yAxis, color='black')
    plt.show()


def supportVectorMachine():
    print("-------------------------------- TASK 4 --------------------------------")
    sampleValueList = []
    averageGitTimeList = []
    averagePredictionList = []
    averageAccuracyList = []
    sampleList = [200, 500, 1000, 2000, 5000]
    gammaList = [1e-3, 1e-5, 1e-7, 1e-10]  # This is my different gamma values

    # Nested for loop which goes through the sample list and then the gamma list
    for sampleValue in sampleList:
        for gameValue in gammaList:
            classiferSVM = svm.SVC(gamma=gameValue)  # Using radial basis function
            sampleValue, averageAccuracy, averagePredictiomTime, averageFitTime = evaluation(classiferSVM,
                                                                                             sampleValue)
            sampleValueList.append(sampleValue)
            averageGitTimeList.append(averageFitTime)
            averagePredictionList.append(averagePredictiomTime)
            averageAccuracyList.append(averageAccuracy)

    calculatedAverageFitTime = sum(averageGitTimeList) / len(averageGitTimeList)
    caculatedAveragePrediction = sum(averagePredictionList) / len(averagePredictionList)
    calculatedAverageAccuracy = sum(averageAccuracyList) / len(averageAccuracyList)

    print("\n\nMean prediction accuracy for Support Vector Machine")
    print(f"Average Training Time: {calculatedAverageFitTime}")
    print(f"Average Prediction Time: {caculatedAveragePrediction}")
    print(f"Average Accuracy: {calculatedAverageAccuracy}")
    print(f"Best Accuracy: {max(averageAccuracyList)}")

    xAxis = sampleValueList
    yAxis = averageGitTimeList

    plt.title("Line Plot between Input Data Vs Runtime (SVM)")
    plt.xlabel("Sample values")
    plt.ylabel("Average Accuracy Times")
    plt.plot(xAxis, yAxis, color='black')
    plt.show()


def kNearestNeighbours():
    print("-------------------------------- TASK 5 --------------------------------")
    sampleValueList = []
    averageGitTimeList = []
    averagePredictionList = []
    averageAccuracyList = []
    sampleList = [200, 500, 1000, 2000, 5000]
    kList = [1, 3, 5, 8, 10]  # List for different k values

    for sampleValue in sampleList:
        for kValues in kList:
            # Passing values from Klist and using default metric
            classifierKNN = neighbors.KNeighborsClassifier(n_neighbors=kValues, metric="minkowski")
            sampleValue, averageAccuracy, averagePredictiomTime, averageFitTime = evaluation(classifierKNN,
                                                                                             sampleValue)
            sampleValueList.append(sampleValue)
            averageGitTimeList.append(averageFitTime)
            averagePredictionList.append(averagePredictiomTime)
            averageAccuracyList.append(averageAccuracy)

    calculatedAverageFitTime = sum(averageGitTimeList) / len(averageGitTimeList)
    caculatedAveragePrediction = sum(averagePredictionList) / len(averagePredictionList)
    calculatedAverageAccuracy = sum(averageAccuracyList) / len(averageAccuracyList)

    print("\n\nMean prediction accuracy for k-nearest Neighbours")
    print(f"Average Training Time: {calculatedAverageFitTime}")
    print(f"Average Prediction Time: {caculatedAveragePrediction}")
    print(f"Average Accuracy: {calculatedAverageAccuracy}")
    print(f"Best Accuracy: {max(averageAccuracyList)}")

    xAxis = sampleValueList
    yAxis = averageGitTimeList

    plt.title("Line Plot between Input Data Vs Runtime (k-nearest Neighbours)")
    plt.xlabel("Sample values")
    plt.ylabel("Average Accuracy Times")
    plt.plot(xAxis, yAxis, color='black')
    plt.show()


def decisionTree():
    print("-------------------------------- TASK 6 --------------------------------")
    classifierDecisionTree = tree.DecisionTreeClassifier()
    sampleValueList = []
    averageGitTimeList = []
    averagePredictionList = []
    averageAccuracyList = []
    sampleList = [200, 500, 1000, 2000, 5000]

    for sampleValue in sampleList:
        sampleValue, averageAccuracy, averagePredictiomTime, averageFitTime = evaluation(classifierDecisionTree,
                                                                                         sampleValue)
        sampleValueList.append(sampleValue)
        averageGitTimeList.append(averageFitTime)
        averagePredictionList.append(averagePredictiomTime)
        averageAccuracyList.append(averageAccuracy)

    calculatedAverageFitTime = sum(averageGitTimeList) / len(averageGitTimeList)
    caculatedAveragePrediction = sum(averagePredictionList) / len(averagePredictionList)
    calculatedAverageAccuracy = sum(averageAccuracyList) / len(averageAccuracyList)

    print("\n\nMean prediction accuracy for Decision trees")
    print(f"Average Training Time: {calculatedAverageFitTime}")
    print(f"Average Prediction Time: {caculatedAveragePrediction}")
    print(f"Average Accuracy: {calculatedAverageAccuracy}")
    print(f"Best Accuracy: {max(averageAccuracyList)}")

    xAxis = sampleValueList
    yAxis = averageGitTimeList

    plt.title("Line Plot between Input Data Vs Runtime (Decision tree)")
    plt.xlabel("Sample values")
    plt.ylabel("Average Accuracy Times")
    plt.plot(xAxis, yAxis, color='black')
    plt.show()


def main():
    classiferPerceptron = linear_model.Perceptron()

    preprocessing()  # Task 1
    evaluation(classiferPerceptron, 200)  # Task 2
    perceptron()  # Task 3
    supportVectorMachine()  # Task 4
    kNearestNeighbours()  # Task 5
    decisionTree()  # Task 6


main()
