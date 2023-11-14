# Ex-07-Feature-Selection
## AIM
To Perform the various feature selection techniques on a dataset and save the data to a file. 

# Explanation
Feature selection is to find the best set of features that allows one to build useful models.
Selecting the best features helps the model to perform well. 

# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature selection techniques to all the features of the data set
### STEP 4
Save the data to the file

# CODE
## titanic_dataset.csv :
```
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from google.colab import files
upload = files.upload()
df = pd.read_csv('titanic_dataset.csv')
df
```
![image](https://github.com/Anuayshh/expt7/assets/127651217/e3ae46da-ab1d-4c34-91ab-94c56aed7963)

```
df.isnull().sum()
```
![image](https://github.com/Anuayshh/expt7/assets/127651217/d7983870-1d4e-41d6-afa5-9963af473680)

```
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'].astype(str))
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
df[['Age']] = imputer.fit_transform(df[['Age']])
print("Feature selection")
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
selector = SelectKBest(chi2, k=3)
X_new = selector.fit_transform(X, y)
print(X_new)
```
![image](https://github.com/Anuayshh/expt7/assets/127651217/00ab032c-503a-4029-8134-09bec5b1ec05)

```
df_new = pd.DataFrame(X_new, columns=['Pclass', 'Age', 'Fare'])
df_new['Survived'] = y.values
df_new.to_csv('titanic_transformed.csv', index=False)
print(df_new)
```
![image](https://github.com/Anuayshh/expt7/assets/127651217/bb9ad2f8-1141-44fb-a9bb-3f40697e3857)

```
from google.colab import files
uploaded = files.upload()
df = pd.read_csv("CarPrice.csv")
df
```
![image](https://github.com/Anuayshh/expt7/assets/127651217/24c2c2fa-7f57-4249-bfd2-e1261b285bf3)

```
df = df.drop(['car_ID', 'CarName'], axis=1)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['fueltype'] = le.fit_transform(df['fueltype'])
df['aspiration'] = le.fit_transform(df['aspiration'])
df['doornumber'] = le.fit_transform(df['doornumber'])
df['carbody'] = le.fit_transform(df['carbody'])
df['drivewheel'] = le.fit_transform(df['drivewheel'])
df['enginelocation'] = le.fit_transform(df['enginelocation'])
df['enginetype'] = le.fit_transform(df['enginetype'])
df['cylindernumber'] = le.fit_transform(df['cylindernumber'])
df['fuelsystem'] = le.fit_transform(df['fuelsystem'])
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
print("Univariate Selection")
selector = SelectKBest(score_func=f_regression, k=10)
X_train_new = selector.fit_transform(X_train, y_train)
mask = selector.get_support()
selected_features = X_train.columns[mask]
model = ExtraTreesRegressor()
model.fit(X_train, y_train)
importance = model.feature_importances_
indices = np.argsort(importance)[::-1]
selected_features = X_train.columns[indices][:10]
df_new = pd.concat([X_train[selected_features], y_train], axis=1)
df_new.to_csv('CarPrice_new.csv', index=False)
print(df_new)
```
![image](https://github.com/Anuayshh/expt7/assets/127651217/81c8cc1a-cead-45d9-943d-b969015dac6b)

# RESULT:
Thus, the various feature selection techniques have been performed on a given dataset successfully.
