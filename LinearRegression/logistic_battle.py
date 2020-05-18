import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import LinearRegression as lr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load in candy data
df = pd.read_csv("candy-data.csv")
df = df.drop(columns = ['competitorname', 'winpercent'])

X = df.iloc[:, 1:]
y = df.iloc[:, 0][:, np.newaxis]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

my_regressor = lr.LinearRegression(X_train, y_train).fit()
sklearn_regressor = LinearRegression().fit(X_train, y_train)

my_train_accuracy = my_regressor.score()
sklearn_train_accuracy = sklearn_regressor.score(X_train, y_train)

# TODO: still have to implement score method within my regressor
my_test_accuracy = my_regressor.score(X_test, y_test)