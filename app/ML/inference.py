import joblib

random_forest_model = joblib.load('./artifacts/random_forest_model.pkl')
logistic_regression_model = joblib.load('./artifacts/logistic_regression_model.pkl')
decision_tree_model = joblib.load('./artifacts/decision_tree_model.pkl')

def predict_dengue(input_data, model_type='random_forest'):
    """
    Predict dengue infection using the specified model.

    Parameters:
    input_data (array-like): Preprocessed input data for prediction.
    model_type (str): Type of model to use for prediction. Options are 'random_forest', 'logistic_regression', 'decision_tree'.

    Returns:
    int: Predicted class label (0 or 1).
    """

    if model_type == 'random_forest':
        model = random_forest_model
    elif model_type == 'logistic_regression':
        model = logistic_regression_model
    elif model_type == 'decision_tree':
        model = decision_tree_model
    else:
        raise ValueError("Invalid model_type. Choose from 'random_forest', 'logistic_regression', 'decision_tree'.")

    prediction = model.predict(input_data)
    return prediction[0]

input_data_example = [[65,11,60,21.9,82,15,3,0,0,3.77,285,0]]  # Example input data

predict = predict_dengue(input_data_example, model_type='decision_tree')
print(f"Predicted Dengue Infection (Decision Tree): {predict}")