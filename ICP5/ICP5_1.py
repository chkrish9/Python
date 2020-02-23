import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

train = pd.read_csv('Data/train.csv')

# Scatter Plot before removing outlier
garage_area = train['GarageArea']
sales_price = train['SalePrice']
plt.scatter(garage_area, sales_price, alpha=.75, color='b')
plt.xlabel('Garage Area')
plt.ylabel('Sale Price')
plt.title('Linear Regression Model')
plt.show()

# Removing outlier by using zscore
# Z score is the relationship between MEAN and Standard Deviation.
data = pd.concat([train['GarageArea'], train['SalePrice']], axis=1)
z = np.abs(stats.zscore(data))
threshold = 3
data = data[(z < 3).all(axis=1)]
print(data)

# Scatter Plot after removing outlier
garage_area = data['GarageArea']
sales_price = data['SalePrice']
plt.scatter(garage_area, sales_price, alpha=.75, color='b')
plt.xlabel('Garage Area')
plt.ylabel('Sale Price')
plt.title('Linear Regression Model')
plt.show()
