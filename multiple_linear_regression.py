"""
This is a pretty straight forward, easy peasy multiple linear regression model. The dataset is retrieved from the UCI Machine Learning repository and contains 2014 Facebook post metrics from over 500 posts from an international cosmetics company. It has multiple independent and dependent variables that we can use. In this case, I used the metrics to predict Lifetime Engaged Users as a baseline (although any other predicted variable can be used).

### Importing the libraries
These are the three go to libraries for most ML.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""### Importing the dataset
I imported the dataset through Pandas dataframe and used iloc to assign the variables. Remember that the name of the dataset has to be updated for diff usecases AND it must be in the same folder as your .py file or uploaded on Jupyter Notebooks or Google Collab.
"""

dataset = pd.read_csv('dataset_Facebook.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
# print(dataset.info())
# ^ this is just to see the missing entries in our data. As an alternative you can also use conditional formatting on the csv.

"""### Missing Data
For this dataset there were a total of 3 columns with missing data. For index 8 and 9 (last index is not included hence '8:10') I used the 'median' to impute to compensate for outliers. For index 6 I used 'most_frequent' because it was a binary datapoint.
"""

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
imputer.fit(X[:, 8:10])
X[:, 8:10] = imputer.transform(X[:, 8:10])

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputer.fit(X[:, 6:7])
X[:, 6:7] = imputer.transform(X[:, 6:7])

"""### Encoding categorical data
Index 1 had categorical data that had to be converted using OneHotEncoding.This must be done AFTER imputing missing values since OneHotEncoding automatically makes the encoded column the first index which displaces all the rest.
"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

"""### Splitting the dataset into the Training set and Test set
Because of the large dataset and many variables for a simple algorothim I used a 90/10 split. The random state is tuned to 3 for consistency sakes.
"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state = 3)

"""### Training the Multiple Linear Regression model on the Training set
After making an instance of the LinearRegression object, I it to fit the training data and train the model.
"""

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

"""### Predicting the Test set results
By using the concatenate function I display the predicted values and  actual values in a side by side 2D array through '(len(y_wtv), 1))' for easy viewing.
"""

y_pred = regressor.predict(X_test)
np.set_printoptions(precision=0, suppress=True)
#print(np.concatenate((y_pred.astype(int).reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

"""### Evaluating Model Performance
We use two metrics to evaluate our model performance, r^2 being the more superior. These are both simple to understand and are covered in one of my Medium articles! In this model we acheivced a nearly .80 r^2 which means 80% of our data can be predicted by our model.
"""

from sklearn.metrics import r2_score, mean_squared_error as mse
print("r^2: " + str(r2_score(y_test, y_pred)))
print("MSE: " + str(mse(y_test, y_pred)))