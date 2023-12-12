import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from AR import AR
from CNN import CNN, absolute_error_statistics

# Plot the difference between the predicted and actual values
def plot(data, train, test, predictions):
    padding = len(data.index[len(train):]) - len(predictions)
    plt.figure(figsize=(30,7))
    plt.plot(data.index[len(train) - 100:len(train)], train[len(train) - 100:], color='b', label='train')
    plt.plot(data.index[len(train):], test, color='g', label='ground truth')
    plt.plot(data.index[len(train) + padding:], predictions, color='r', label='predicted')
    plt.xticks([])
    plt.title("Difference between test and predicted data")
    plt.legend(loc='best')
    plt.show()

# Dispatcher function to train and evaluate the models
def dispatch(file, quantity, method):
    if method.upper() == 'CNN':
        cnn = CNN(file, quantity)
        print("---------------------------Training the CNN model---------------------------")
        # Train the model for 200 epochs
        for epoch in range(200):
            loss = cnn.train()
            if epoch % 10 == 0:
                print(f"Error: {loss}")
        
        error, predicted, actual = cnn.evaluate()
        print(f"MSE on test dataset: {error}")
        stat = absolute_error_statistics(np.array(actual), np.array(predicted))
        df = pd.DataFrame(stat)
        df = df.describe()
        print(f"\nMean absolute error on test data: {df[0]['mean']}, median error: {df[0]['50%']}")
        plot(cnn.data, cnn.train_data, cnn.test_data, predicted)

    else:
        print("---------------------------Training the AR model----------------------------")
        ar = AR(file, quantity)
        error, predicted, actual = ar.evaluate()
        print(f"MSE on test dataset: {error}")
        stat = absolute_error_statistics(np.array(actual), np.array(predicted))
        df = pd.DataFrame(stat)
        df = df.describe()
        print(f"\nMean absolute error on test data: {df[0]['mean']}, median error: {df[0]['50%']}")
        plot(ar.data, ar.train, ar.test, predicted)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help="Input file to process", required=True)
    parser.add_argument('-q', '--quantity', help="Input column", required=True, nargs='+')
    parser.add_argument('-m', '--method', help="Input method, either CNN or AR", required=True)
    args = parser.parse_args()
    
    file = args.input
    # For when the input quantity contains space
    quantity = ' '.join(args.quantity)
    method = args.method

    # Only two methods supported
    if method.upper() not in ['CNN', 'AR']:
        raise Exception('Unsupported method, method shoud be either CNN, or AR.')
    
    dispatch(file, quantity, method)