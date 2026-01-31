import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("/Users/eazdanmostafarafin/All Work/Clinexa/Dataset/CBC Report.csv")



# Data Preprocessing
data = df.drop(columns = ['Serial','Date'])


# Handle missing values by filling them with the mean of the column
columns_impute = ['ESR', 'Lymphocyte', 'Monocyte', 'Eosinophil', 'Basophil', 'RBC','Neutrophil']
imputer = SimpleImputer(strategy='mean', missing_values=np.nan)
imputer = imputer.fit(data[columns_impute])
data[columns_impute] = imputer.transform(data[columns_impute])

# Encoding target column
labelencoder = LabelEncoder()
data['Result'] = labelencoder.fit_transform(data['Result'])

# One-hot encoding Gender Column
data = pd.get_dummies(data, columns=['Gender'], drop_first=True)


# Split the data into features and target variable

X = data.drop(['Result'], axis=1)
y = data['Result']

X_train, temp_X, y_train, temp_y = train_test_split(X, y, test_size=0.1, random_state=23, stratify=y)
X_test, X_cv, y_test, y_cv = train_test_split(temp_X, temp_y, test_size=0.5, random_state=23, stratify=temp_y)

# Feature Scaling
# features standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_cv_scaled = scaler.transform(X_cv)

print("Shape of scaled training data:", X_train_scaled.shape, y_train.shape)
print("Shape of scaled testing data:", X_test_scaled.shape, y_test.shape)
print("Shape of scaled cross-validation data:", X_cv_scaled.shape, y_cv.shape)

print(data.head())