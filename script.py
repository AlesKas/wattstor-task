import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from AR import AR
from CNN import CNN, absolute_error_statistics

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

def dispatch(file, quantity, method):
    if method.upper() == 'CNN':
        cnn = CNN(file, quantity)
        for epoch in range(200):
            loss = cnn.train()
            if epoch % 10 == 0:
                print(f"Loss: {loss}")
        
        error, predicted, actual = cnn.evaluate()
        print(f"Error on test dataset: {error}")
        stat = absolute_error_statistics(np.array(actual), np.array(predicted))
        df = pd.DataFrame(stat)
        df = df.describe()
        print(f"Absolute error on test data:")
        print(df)
        plot(cnn.data, cnn.train_data, cnn.test_data, predicted)

    else:
        ar = AR(file, quantity)
        error, predicted, actual = ar.evaluate()
        print(f"Error on test dataset: {error}")
        stat = absolute_error_statistics(np.array(actual), np.array(predicted))
        df = pd.DataFrame(stat)
        df = df.describe()
        print(f"Absolute error on test data:")
        print(df)
        plot(ar.data, ar.train, ar.test, predicted)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help="Input file to process", required=True)
    parser.add_argument('-q', '--quantity', help="Input column", required=True, nargs='+')
    parser.add_argument('-m', '--method', help="Input method, either CNN or AR", required=True)
    args = parser.parse_args()
    
    file = args.input
    quantity = ' '.join(args.quantity)
    method = args.method

    if method.upper() not in ['CNN', 'AR']:
        raise Exception('Unsupported method, method shoud be either CNN, or AR.')
    
    dispatch(file, quantity, method)