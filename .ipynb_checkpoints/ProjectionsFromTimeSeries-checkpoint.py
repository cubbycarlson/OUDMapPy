#consider turning off graphical displays when actually running this

from pmdarima.arima import auto_arima, ADFTest
from matplotlib import pyplot as plt
import json
import numpy as np
import pandas as pd
import os

with open("time_series.json") as f:
    data = json.load(f)
states = data['state_list']

n = len(data['dates']) + 1  # of dates for which there is data available
idx = np.arange(n - 1)  # index of each date

n_projections = 24

months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November",
          "December"]
years = ["2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022"]

dates = []
i = 0
max_i = 98
for year in years:
    for month in months:
        if i < max_i:
            dates.append({"month": month, "year": year})
        i += 1

for state in states:
    time_series = {}
    time_series['dates'] = dates
    time_series['actuals'] = data[state]
    time_series['projections'] = {}
    time_series['projection_metadata'] = {}

    print('STARTING PROJECTION CREATION FOR', state)
    df = pd.DataFrame({"date": idx, "deaths": data[state]})
    df.set_index('date', inplace=True)
    adf_test = ADFTest(alpha=0.05)
    n_test = 0
    n_train = n - n_test
    # test = df[-n_test:]
    train = df[:n_train]
    # plt.plot(train)
    # plt.plot(test)
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
    # arima_model.summary()

    projections_idx = np.arange(n_projections) + n_train - 1
    p, c_i = arima_model.predict(n_periods=n_projections, return_conf_int=True)
    c_i_l_b = c_i[:, 0]  # lower bound of confidence interval
    c_i_h_b = c_i[:, 1]  # higher bound of confidence interval

    projection = pd.DataFrame(p, index=projections_idx)
    c_i_lower_bound = pd.DataFrame(c_i_l_b, index=projections_idx)
    c_i_higher_bound = pd.DataFrame(c_i_h_b, index=projections_idx)

    plt.plot(train, label="actual (training data)")
    # plt.plot(test,label='actual (not used in training)')
    plt.plot(projection, label='projection')
    plt.plot(c_i_lower_bound, label='lower bound')
    plt.plot(c_i_higher_bound, label='higher bound')
    plt.legend(loc='upper left')
    plt.title(state)
    plt.show()

    projection_array = projection.to_numpy().flatten()
    arima_order = arima_model.to_dict()['order']
    arima_seasonal_order = arima_model.to_dict()['seasonal_order']

    time_series['projections']['predicted_death'] = projection_array.tolist()
    time_series['projections']['confidence_intervals'] = {
        "l_bound": c_i_l_b.tolist(),
        "h_bound": c_i_h_b.tolist()
    }
    time_series['projections']['indices'] = projections_idx.tolist()
    time_series['projection_metadata']['arima_order'] = arima_order
    time_series['projection_metadata']['arima_seasonal_order'] = arima_seasonal_order
    time_series['projection_metadata']['training_data_indices'] = [0, n]

    filename = f"projections/{state}.json"
    with open(filename, 'r') as f:
        to_be_deletd = json.load(f)
    os.remove(filename)

    with open(filename, 'w') as f:
        json.dump(time_series, f)
