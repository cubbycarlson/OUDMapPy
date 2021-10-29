import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import json

df = pd.read_csv('data.csv')
df.head()
data = df[['State', 'Year', 'Month', 'Indicator', 'Data Value']]
# data = data[data["Indicator"].isin(["Number of Deaths", "Number of Drug Overdose Deaths"])]
data = data[data["Indicator"].isin(["Number of Drug Overdose Deaths"])]

states = data['State'].unique()
years = data['Year'].unique()
months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

death_time_series = {
    'dates': [],
    'state_list': []
}

for state in states:
    s = [state]
    death_time_series['state_list'].append(state)
    death_time_series[state] = []
    for year in years:
        y = [year]
        for month in months:
            m = [month]
            row = data.query(f'State in {s} & Year in {y} & Month in {m}')
            if (len(row['Data Value'].values) > 0):
                overdose_deaths = float(row['Data Value'].values[0].replace(',',''))
                death_time_series[state].append(overdose_deaths)
                if(state == 'CA'): #arbitrary
                    death_time_series['dates'].append([month, str(year)])

print(death_time_series)
with open('time_series.json', 'w') as fp:
    json.dump(death_time_series, fp)