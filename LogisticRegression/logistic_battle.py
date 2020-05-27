import numpy as np
import pandas as pd
import seaborn as sns
import LogisticRegression as lr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.datasets import make_classification

X, y = make_classification(n_samples=5000, n_features=2, n_redundant=0, n_informative=1, 
                           n_clusters_per_class=1, random_state = 2)
print(f"This dataset contains {X.shape[0]} entries and {X.shape[1]} features")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

my_regressor = lr.LogisticRegression(X_train, y_train).fit()
sklearn_regressor = LogisticRegression().fit(X_train, y_train)

my_train_accuracy = my_regressor.score()
sklearn_train_accuracy = sklearn_regressor.score(X_train, y_train)

my_test_accuracy = my_regressor.score(X_test, y_test)
sklearn_test_accuracy = sklearn_regressor.score(X_test, y_test)

result = pd.DataFrame([[my_train_accuracy, sklearn_train_accuracy],
              [my_test_accuracy, sklearn_test_accuracy]],
             ['Training Acc.', 'Test Acc.'],    
             ['Our\'s', 'Sklearn\'s'])

print(result)