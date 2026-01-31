import joblib

logistic_regression_model = joblib.load('logistic_regression_model.pkl')
random_forest_model = joblib.load('random_forest_model.pkl')
decision_tree_model = joblib.load('decision_tree_model.pkl')


# Example: new CBC report
new_data = pd.DataFrame([{
    "Serial": 1,
    "Date": "2025-01-01",
    "ESR": 12,
    "Lymphocyte": 30,
    "Monocyte": 6,
    "Eosinophil": 2,
    "Basophil": 1,
    "RBC": 4.8,
    "Neutrophil": 60,
    "Gender": "Male"
}])

X_new = preprocess_data(new_data, fit=False, artifacts=artifacts)

prediction = models["random_forest"].predict(X_new)

print("Prediction:", prediction)