import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import warnings
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
import datetime
from dateutil.relativedelta import relativedelta


# Funtion To Check if series is stationary
def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = timeseries.rolling(window=12,center=False).mean()
    rolstd = timeseries.rolling(window=12,center=False).std()

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=True)
    
    #Perform Dickey-Fuller test:
    print 'Results of Dickey-Fuller Test:'
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print dfoutput
    return



dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m')
data = pd.read_csv('DayCareChildren.csv', parse_dates=['Month'], index_col='Month',date_parser=dateparse)

ts = data['#Children']
ts_log = np.log(ts)

# This Code is Not Needed
'''
ts_log_diff = ts_log - ts_log.shift(1)
ts_log_diff.dropna(inplace=True)
ts_log_seasonal_first_diff = ts_log_diff - ts_log_diff.shift(12)
ts_log_seasonal_first_diff.dropna(inplace=True)
'''

#test for stationary
test_stationarity(ts_log)

# Apply the model
mod = SARIMAX(ts_log, order=(0,1,0), seasonal_order=(1,1,1,12))
results = mod.fit()



#future year predictions
start = datetime.datetime.strptime("1961-01-01", "%Y-%m-%d")
date_list = [start + relativedelta(months=x) for x in range(0,12)]
future = pd.DataFrame(index=date_list, columns= data.columns)
data = pd.concat([data, future])

ts_log_forecast = results.predict(start = 144, end =168 , dynamic=True)
ts_forecast = np.exp(ts_log_forecast)


# Display the graph
plt.plot(ts_forecast, color='green')
plt.plot(ts, color='blue')
plt.show()