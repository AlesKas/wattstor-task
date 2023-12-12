import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error

class AR():
    def __init__(self, input_file, quantity) -> None:
        self.data = pd.read_csv(input_file, delimiter=';')
        self.data['Time'] = pd.to_datetime(self.data['Time'], utc=True)
        self.data.set_index('Time')

        if quantity not in self.data.columns:
            raise Exception(f"{quantity} column not in data.")

        self.fill_missing_data()

        # Values obtained from the jupyyer notebook
        self.lag_info = {
            'Consumption': {'order': 1, 'lags': 96}, 
            'Grid consumption': {'order': 1, 'lags': 101}, 
            'PV generation': {'order': 1, 'lags': 11}, 
            'Battery charging' : {'lags' : 1},
            'Battery discharging': {'order': 1, 'lags': 11}, 
            'Grid backflow' : {'lags' : 1}
        }

        # Split the data to train and test datasets
        self.X = self.data[quantity].values
        size = int(len(self.X) * 0.80)
        self.train, self.test = self.X[0:size], self.X[size:len(self.X)]
  
        # Obtain the optimal number of lags and fit the model
        self.number_of_lags = self.lag_info[quantity]['lags']
        self.model = AutoReg(self.train, lags=self.number_of_lags)
        self.model_fit = self.model.fit()

    def fill_missing_data(self):
        missing = {}
        columns = [i for i in self.data.columns if i not in ['Time']]
        for column in columns:
            missing[column] = self.data[self.data[column].isna()].index.to_list()

        for key, value in missing.items():
            if len(value) == 0:
                continue
            start_index = value[0] - 1
            end_index = value[-1] + 1

            start_value = self.data[key][start_index]
            end_value = self.data[key][end_index]

            fill_data = float((end_value - start_value) / (len(value) + 1))
            new_value = start_value + fill_data
            for item in value:
                self.data.at[item, key] = new_value
                new_value += fill_data

    def evaluate(self):
        coef = self.model_fit.params
        history = self.train[len(self.train)-self.number_of_lags:]
        history = [history[i] for i in range(len(history))]
        predictions = list()
        for t in range(len(self.test)):
            length = len(history)
            lag = [history[i] for i in range(length-self.number_of_lags,length)]
            yhat = coef[0]
            for d in range(self.number_of_lags):
                yhat += coef[d+1] * lag[self.number_of_lags-d-1]
                obs = self.X[t]
            predictions.append(yhat)
            history.append(obs)
        error = mean_squared_error(self.test, predictions)
        return error, predictions, self.test