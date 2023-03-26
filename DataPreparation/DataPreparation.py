import os
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm
import pandas as pd

def read_data(kind=None, dummy=False, display=False):
    '''
    reads the dataset from the folder and return it. 
    If kind is specified, it returns only the categorical or numerical features.
    dummy is a boolean that specifies if the categorical features should be one-hot encoded into numerical features.
    '''
    
    module_dir = os.path.dirname(__file__)
    path = os.path.join(module_dir, '../DataFiles/dataset.csv')
    
    ds = pd.read_csv(path)
    # all columns except Body_Level go to x_data
    x_data = ds.drop('Body_Level', axis=1)
    
    if kind == "Categorical":
        # extract only the categorical features
        disc_feats = [feat for feat in x_data.columns if type(x_data.iloc[0, x_data.columns.get_loc(feat)]) == str]
        x_data = x_data[disc_feats]
    elif kind == "Numerical":
        # extract only the numerical features
        cont_feats = [feat for feat in x_data.columns if type(x_data.iloc[0, x_data.columns.get_loc(feat)]) != str]
        x_data = x_data[cont_feats]
    
    if dummy:   x_data = pd.get_dummies(x_data)
    
    # Body_Level goes to y_data
    y_data = ds['Body_Level']
    
    if display: print(x_data.head())
    
    return x_data, y_data

    
def basic_info(x_data, y_data):
    '''
    prints basic info about the dataset like the number of rows, columns, features and possible classes
    '''
    # print the number of samples in the dataset
    print('\nNumber of samples in the dataset: ', len(x_data))
    
    # print the number of features in the dataset and their names in a table
    print('\nNumber of features in the dataset: ', len(x_data.columns))
    features = pd.DataFrame(x_data.columns)
    print('\nFeatures in the dataset: ', features)
    
    # print the number of classes in the dataset
    print('\nNumber of classes in the dataset: ', len(y_data.unique()))


def prior_distribution(y_data):
    '''
    plots the prior distribution of the dataset which is helpful for class imbalance
    '''
    # plot the prior distribution of the dataset
    plt.figure(figsize=(10, 5))
    plt.hist(y_data, bins=4, color='blue', edgecolor='black', alpha=0.7)
    plt.title('Prior distribution of the dataset')
    plt.xlabel('Body Level')
    plt.ylabel('Number of samples')
    plt.show()
    
    # print the number of samples in each class
    print('\nNumber of samples in each class:\n')
    print(y_data.value_counts())


def features_histograms(x_data):
    '''
    Plot a 4x4 grid of histograms for each feature in the dataset (there are 16 features).
    Also print the number of unique values of each feature and its kind.
    '''
    # plot a 4x4 grid of histograms for each feature in the dataset
    fig, axs = plt.subplots(4, 4, figsize=(20, 20))
    for i in range(4):
        for j in range(4):
            axs[i, j].hist(x_data.iloc[:, i*4+j], bins=20, color='blue', edgecolor='black', alpha=0.7)
            axs[i, j].set_title(x_data.columns[i*4+j])
    plt.show()
    
    # print number of unique values of each feature
    print('\nNumber of unique values of each feature:\n')
    c = 0
    for i in range(len(x_data.columns)):
        feature_type = type(x_data.iloc[0, i])
        if feature_type == str:
            print(x_data.columns[i], ': ', len(x_data.iloc[:, i].unique()))
            c+=1
        else:
            print(x_data.columns[i], ':', '(numerical)')
                        
        
        
    print("Number of categorical features: ", c)
    print("Number of numerical features: ", len(x_data.columns)-c)
        

def visualize_continuous_data(x_data, y_data):
    '''
    Plot a 4x7 grid of scatter plots for each pair of continuous features.
    '''
    # get only the continuous features
    cont_feats = [feat for feat in x_data.columns if type(x_data.iloc[0, x_data.columns.get_loc(feat)]) != str]
    x_data_cont = x_data[cont_feats]
    
    # get all possible combinations of 2 features
    combinations = list(itertools.combinations(x_data_cont.columns, 2))
    
    # craft c as integers for each unique val from ydata
    c = pd.factorize(y_data)[0]
    
    # plot each combination of 2 features in grid of 8C2 = 28 plots
    fig, axs = plt.subplots(4, 7, figsize=(20, 20))
    for i in range(4):
        for j in range(7):
            axs[i, j].scatter(x_data_cont[combinations[i*7+j][0]], x_data_cont[combinations[i*7+j][1]], c=c, cmap='coolwarm')
            axs[i, j].set_title(combinations[i*7+j][0] + ' vs ' + combinations[i*7+j][1])
    plt.show()
    

def numerical_correlation_matrix(x_data):
    '''
    plot a correlation matrix for the continuous features in the dataset
    '''
    # get only the continuous features
    cont_feats = [feat for feat in x_data.columns if type(x_data.iloc[0, x_data.columns.get_loc(feat)]) != str]
    x_data_cont = x_data[cont_feats]
    
    # get the correlation matrix
    corr = x_data_cont.corr()
    
    # plt an 8x8 heatmap representing the correlation matrix. The for loop only prints the values in the heatmap.
    fig, ax = plt.subplots(figsize=(10, 10))
    for i in range(len(corr)):
        for j in range(len(corr)):
             ax.text(j, i, '{:.2f}'.format(corr.values[i, j]), ha="center", va="center", color="w")
    ax.matshow(corr, cmap='bwr')
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.show()
    

    

def read_sample(path):
    '''
    A read_sample function for when the model is to be evaluated
    '''
    pass

