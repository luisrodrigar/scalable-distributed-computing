#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 16:23:47 2022

@author: Sara Dovalo del Río, Alejandra Estrada Sanz and Luis Ángel Rodríguez García
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

########################################
## Part I: Study of the data
########################################

## Lecture of the data
data_1 = pd.read_csv("data/tripdata_2017-01.csv")
data_2 = pd.read_csv("data/tripdata_2017-02.csv")
df = pd.concat([data_1, data_2], axis=0)
(n, p) = df.shape
df.head()
df.info()

## Check for missing values
df.isna().sum().sort_values()

## Type of variables 
print(df.columns)
df.dtypes


## Correlation
correlation=df.corr()
plt.figure(figsize = (15,10))
sns.heatmap(correlation, annot = True)
plt.show()

## Some interest aspects about data
########################################

# Journey data
t1 = df['tpep_dropoff_datetime'].min()
t2 = df['tpep_dropoff_datetime'].max()
print('Dates from', t1, 'to', t2)

## The method of payment used on the longest trip
df_group = df.groupby('payment_type')
max_distance = int(df_group.mean('trip_distance')['trip_distance'].idxmax())
print('The payment with the highest average distance is {}'.format(max_distance))
df.payment_type


## Average distance depending on the part of the day
df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
df['pu_hour'] = df['tpep_pickup_datetime'].dt.hour
df['pu_day'] = df['tpep_pickup_datetime'].dt.dayofyear
df['pu_wday'] = df['tpep_pickup_datetime'].dt.dayofweek
df['pu_month'] = df['tpep_pickup_datetime'].dt.month

morning = df[(df['pu_hour'] >= 7) & (df['pu_hour'] < 9)]
afternoon = df[(df['pu_hour'] >= 9) & (df['pu_hour'] < 16)]
evening = df[(df['pu_hour'] >= 16) & (df['pu_hour'] < 18)]
night = df[(df['pu_hour'] >= 18) & (df['pu_hour'] <= 23)]
latenight =df[(df['pu_hour'] >=0 ) & (df['pu_hour'] < 7)]

print("7am and 9 am average distance :",morning.trip_distance.mean(),
"9am and 4 P.M average distance :",afternoon.trip_distance.mean(),
"4 PM and 6 PM average distance : ",evening.trip_distance.mean(),
"6 pm and 11pm average distance :",night.trip_distance.mean(),
"11pm and 7 am average distance :",latenight.trip_distance.mean(),)

## We can see that the distances are greater at night, this may be due to
## that in many cities there is no public transport at this time,
## therefore people choose to travel by taxi.
## Latenight trip has an effect on distance


# Frequency of passenger numbers
passengers = df['passenger_count'].value_counts().sort_index()
passengers.plot(kind = 'bar',logy = True)
plt.xlabel('Number of passengers')
plt.ylabel('Frequency')
plt.title('Frequency of passenger numbers (logScaler')
plt.show()

# Trip distance depending on the number of passengers
passenger_vs_distance = data.groupby('passenger_count').count()['trip_distance']
passenger_vs_distance.plot()
plt.title('Trip distance depending on the number of passengers')

## There is no correlation between group size and distance

## Hola wapos k tl estáis? yo bien jodida



