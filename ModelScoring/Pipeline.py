'''
The final pipeline goes here (competition model) and its evaluation.
'''
import pickle 
import pandas as pd

def read_sample():
    '''
    A read_sample function for when the model is to be evaluated
    '''

    x_data = pd.read_csv('test.csv')

    # extract only the numerical features
    cont_feats = [feat for feat in x_data.columns if type(x_data.iloc[0, x_data.columns.get_loc(feat)]) != str]
    x_data = x_data[cont_feats]
    for feat in x_data.columns:
        x_data[feat] = x_data[feat].astype(float)

    # standardize the numerical features
    mean_dict = {    
    "Age":              24.290420,
    "Height":            1.701602,
    "Weight":           86.542290,
    "Veg_Consump":       2.421912,
    "Water_Consump":     2.005120,
    "Meal_Count":        2.682104,
    "Phys_Act":          1.023106,
    "Time_E_Dev":        0.650672,
    }
    std_dict = {
    "Age":               6.323081,
    "Height":            0.094654,
    "Weight":           26.278277,
    "Veg_Consump":       0.540711,
    "Water_Consump":     0.620307,
    "Meal_Count":        0.790751,
    "Phys_Act":          0.844840,
    "Time_E_Dev":        0.605199,
    }

    for feat in x_data.columns:
        x_data[feat] = (x_data[feat] - mean_dict[feat]) / std_dict[feat]
    
    return x_data

def load_model(model_path):
    '''
    Loads the model from the given path.
    '''
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def predict(model, x_test):
    '''
    Predicts the target variable for the given data.
    '''
    y_test = model.predict(x_test)
    body_levels = ['Body Level 1', 'Body Level 2', 'Body Level 3', 'Body Level 4']
    y_pred = [body_levels[i] for i in y_test]
    return y_pred


# Read the data
x_test = read_sample()

# Load the model
model = load_model('StackingEnsemble.pkl')

# Predict the target variable
y_pred = predict(model, x_test)

# write the predictions to a txt file
with open('preds.txt', 'w') as f:
    for i, item in enumerate(y_pred):
        if i == len(y_pred) - 1:
            f.write("%s" % item)
        else:
            f.write("%s\n" % item)