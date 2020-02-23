import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

train = pd.read_csv('train.csv')
x = train['GarageArea']
y = train['SalePrice']
garage_area = train['GarageArea']
sales_price = train['SalePrice']
plt.scatter(garage_area, sales_price, alpha=.75, color='b')
plt.xlabel('GarageArea')
plt.ylabel('SalePrice')
plt.title('Linear Regression Model')
plt.show()
data = pd.concat([train['GarageArea'], train['SalePrice']], axis=1)
print(x)
z = np.abs(stats.zscore(data))
threshold = 3
data = data[(z < 3).all(axis=1)]
print(data)
garage_area = data['GarageArea']
sales_price = data['SalePrice']
plt.scatter(garage_area, sales_price, alpha=.75, color='b')
plt.xlabel('GarageArea')
plt.ylabel('SalePrice')
plt.title('Linear Regression Model')
plt.show()