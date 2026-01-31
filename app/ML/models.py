import numpy as np
import pandas as pd
import joblib
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


from preprocess import pre_precessing_dengue as ppd
df = pd.read_csv("/Users/eazdanmostafarafin/All Work/Clinexa/Dataset/CBC Report.csv")

X_train_scaled, X_test_scaled, X_cv_scaled, y_train, y_test, y_cv = ppd(df)


# Logistic Regression Model
logreg= LogisticRegression(random_state=23)
logreg.fit(X_train_scaled, y_train)
y_pred = logreg.predict(X_test_scaled)


conf_matrix_log = confusion_matrix(y_test, y_pred)
classification_rep_log = classification_report(y_test, y_pred)

print("\nConfusion Matrix:\n", conf_matrix_log)
print("\nClassification Report:\n", classification_rep_log)





# Random Forest Classifier Model

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=23)
rf_model.fit(X_train_scaled, y_train)

y_pred_rf = rf_model.predict(X_test_scaled)

conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
classification_rep_rf = classification_report(y_test, y_pred_rf)

print("\nConfusion Matrix:\n", conf_matrix_rf)
print("\nClassification Report:\n", classification_rep_rf)

     



     
# Decision Tree Classifier Model

from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier(random_state=23)
dt_model.fit(X_train_scaled, y_train)

y_pred_dt = dt_model.predict(X_test_scaled)

conf_matrix_dt = confusion_matrix(y_test, y_pred_dt)
classification_rep_dt = classification_report(y_test, y_pred_dt)

print("\nConfusion Matrix:\n", conf_matrix_dt)
print("\nClassification Report:\n", classification_rep_dt)

     
# Save the models
joblib.dump(logreg, './artifacts/logistic_regression_model.pkl')
joblib.dump(rf_model, './artifacts/random_forest_model.pkl')
joblib.dump(dt_model, './artifacts/decision_tree_model.pkl')