import pandas as pd
import numpy as np

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

train = pd.read_csv('Data/winequality-red.csv')

# Handling missing value
data = train.select_dtypes(include=[np.number]).interpolate().dropna()

data_correlation = data.corr(method='pearson')
unstack_data = data_correlation.unstack()
sorted_data = unstack_data.sort_values(kind='quicksort')

print(sorted_data[0:3])


target_data = train.quality
features_data = data.drop(['quality'], axis=1)

features_train, features_test, target_train, target_test = train_test_split(features_data, target_data, random_state=42, test_size=.33)


lr = linear_model.LinearRegression()
model = lr.fit(features_train, target_train)

# Evaluate the performance
print("R^2 is: \n", model.score(features_test, target_test))
predictions = model.predict(features_test)

print('RMSE is: \n', mean_squared_error(target_test, predictions))

