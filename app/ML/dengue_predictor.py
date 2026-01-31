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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

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

X_train, temp_X, y_train, temp_y = train_test_split(X, y, test_size=0.3, random_state=23, stratify=y)
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


# Logistic Regression Model
logreg= LogisticRegression(random_state=23)
logreg.fit(X_train_scaled, y_train)
y_pred = logreg.predict(X_test_scaled)


conf_matrix_log = confusion_matrix(y_test, y_pred)
classification_rep_log = classification_report(y_test, y_pred)

print("\nConfusion Matrix:\n", conf_matrix_log)
print("\nClassification Report:\n", classification_rep_log)





# import xgboost as xgb

# # Initialize the XGBoost model
# xgb_model = xgb.XGBClassifier(random_state=23)

# # Train the model on the training data
# xgb_model.fit(X_train_scaled, y_train)

# # Make predictions on the testing data
# y_pred_xgb = xgb_model.predict(X_test_scaled)

# # Evaluate the model
# conf_matrix_xgb = confusion_matrix(y_test, y_pred_xgb)
# classification_rep_xgb = classification_report(y_test, y_pred_xgb)

# # Display the evaluation metrics for XGBoost
# print("\nXGBoost:")
# print("\nConfusion Matrix:\n", conf_matrix_xgb)
# print("\nClassification Report:\n", classification_rep_xgb)


     
