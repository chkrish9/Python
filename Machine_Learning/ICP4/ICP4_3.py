import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

glass_df = pd.read_csv('./Data/glass.csv')
features = ["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]
target = "Type"
features_train, features_test, target_train, target_test = train_test_split(glass_df[features], glass_df[target], train_size=0.8, test_size=0.2, random_state=0)

svc = SVC()
target_pred = svc.fit(features_train, target_train).predict(features_test)
accuracy = accuracy_score(target_test, target_pred)
print(accuracy)