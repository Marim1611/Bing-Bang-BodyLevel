'''
The final pipeline goes here (competition model) and its evaluation.
'''
import pickle 
import sys; sys.path.append('../')
from DataPreparation.DataPreparation import read_sample


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
x_test = read_sample('../ModelScoring/test.csv')

# Load the model
model = load_model('../ModelScoring/StackingEnsemble.pkl')

# Predict the target variable
y_pred = predict(model, x_test)

# write the predictions to a txt file
with open('preds.txt', 'w') as f:
    for i, item in enumerate(y_pred):
        if i == len(y_pred) - 1:
            f.write("%s" % item)
        else:
            f.write("%s\n" % item)