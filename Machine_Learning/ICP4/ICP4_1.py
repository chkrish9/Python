import pandas as pd

train_df = pd.read_csv('./Data/train_preprocessed.csv')
test_df = pd.read_csv('./Data/test_preprocessed.csv')


print(train_df.corr(method ='pearson'))

print(train_df.corr(method ='kendall'))

print(train_df.corr())

