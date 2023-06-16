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
    means=[24.30154547138047, 1.7044246599326598, 86.24393004377106, 2.4137310067340065, 1.9985143434343435, 2.603423063973064, 1.0657414511784513, 0.6401021212121212] 
    stds=[6.187403093300774, 0.09328631635951697, 25.765476944060072, 0.5586174649270286, 0.6404876859634282, 0.8226938923003604, 0.8170331197353868, 0.5959943002074906,0.98989898989899]
    for i,feat in enumerate(x_data.columns):
        x_data[feat] = (x_data[feat] - means[i])/stds[i]

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