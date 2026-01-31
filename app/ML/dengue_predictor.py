import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer


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

# Encoding Gender Column using ColumnTransformer
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(drop='first'), ['Gender'])],
    remainder='passthrough'
)
data = pd.DataFrame(ct.fit_transform(data))

print(data.head())
