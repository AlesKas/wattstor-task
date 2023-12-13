import pytest

import pandas as pd

from utils import fill_missing_data

def test_fill_missing_data():
    data = pd.read_csv('./data/SG.csv', delimiter=';')
    data['Time'] = pd.to_datetime(data['Time'], utc=True)
    data.set_index('Time')
    data = fill_missing_data(data)

    for column in data.columns:
        assert len(data[data[column].isna()]) == 0