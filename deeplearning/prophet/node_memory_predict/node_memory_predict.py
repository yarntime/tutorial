#!/usr/bin/env python

import pandas as pd
import numpy as np
from fbprophet import Prophet
import matplotlib.pyplot as plt

data_df = pd.read_csv("grafana_data_export.csv")
data_df["y"] = np.log(data_df["y"])
print(data_df.head())
print(data_df.tail())

m = Prophet()
m.fit(data_df)

# make prediction
data_future = m.make_future_dataframe(periods=600, freq="S")
print(data_future.tail())
pred_res = m.predict(data_future)
print(pred_res[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

# visualization
m.plot(pred_res)

m.plot_components(pred_res)

plt.show()
