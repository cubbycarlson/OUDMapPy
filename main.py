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
                if(state == 'CA'): # arbitrary
                    death_time_series['dates'].append([month, str(year)])

print(death_time_series)
with open('time_series.json', 'w') as fp:
    json.dump(death_time_series, fp)
    
from pmdarima.arima import auto_arima, ADFTest
from matplotlib import pyplot as plt
import json
import numpy as np
import pandas as pd

with open("time_series.json") as f:
    data = json.load(f)
states = data['state_list']

n = len(data['dates']) + 1

idx = np.arange(n - 1)

for state in states:
    df = pd.DataFrame({"date": idx, "deaths": data[state]})
    df.set_index('date', inplace=True)
    adf_test = ADFTest(alpha = 0.05)
    n_test = 6
    n_train = n - n_test
    train = df[:n_train]
    test = df[-n_test:]
    plt.plot(train)
    plt.plot(test)
    arima_model = auto_arima(
        train,
        start_p=0,
        d=1,
        start_q=0,
        max_p=5,
        max_d=5,
        max_q=5,
        start_P=0,
        D=1,
        start_Q=0,
        max_P=5,
        max_D=5,
        max_Q=5,
        m=12,
        seasonal=True,
        error_action='warn',
        trace=True,
        suppress_warnings=True,
        stepwise=True,
        random_state=20,
        n_fits=20
    )

    arima_model.summary()

    # prediction = pd.DataFrame(arima_model.predict(n_periods=n_test), index=test.index)
    n_projections = 24
    projection_idx = np.arange(n_projections) + n_train
    projection = pd.DataFrame(
        arima_model.predict(n_periods=n_projections),
        index=projection_idx
    )

    prediction.columns=['predicted_deaths']
    prediction

    plt.plot(train,label="actual (training data)")
    plt.plot(test,label='actual (not used in training)')
    plt.plot(projection, label='projection')
    plt.legend(loc='upper left')
    plt.title(state)
    plt.show()

    projection_array = projection.to_numpy().flatten()
    arima_order = arima_model.to_dict()['order']
    arima_seasonal_order = arima_model.to_dict()['seasonal_order']

    projection_data = {}
    projection_data['deaths'] = projection_array.tolist()
    projection_data['indices'] = projection_idx.tolist()
    projection_data['arima_order'] = arima_order
    projection_data['arima_seasonal_order'] = arima_seasonal_order

    with open(f'projections/{state}.json', 'w') as fp:
        json.dump(projection_data, fp)
        print(projection_data, 'SAVED')