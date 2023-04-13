import os
import matplotlib.pyplot as plt
import itertools
import pandas as pd
import numpy as np
import seaborn as sns



def read_data(kind=None, encode=None, split="train", display=False):
    '''
    reads the dataset from the folder and return it. 
    If kind is specified, it returns only the categorical or numerical features.
    dummy is a boolean that specifies if the categorical features should be one-hot encoded into numerical features.
    '''
    module_dir = os.path.dirname(__file__)
    if split == "train":    path = os.path.join(module_dir, '../DataFiles/train.csv')
    elif split == "val":    path = os.path.join(module_dir, '../DataFiles/val.csv')
    elif split == "all":   path = os.path.join(module_dir, '../DataFiles/dataset.csv')
    
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
        for feat in x_data.columns:
            x_data[feat] = x_data[feat].astype(float)
    
    if encode=='Label':
        # convert categorical features to integer labels
        for feat in x_data.columns:
            if type(x_data.iloc[0, x_data.columns.get_loc(feat)]) == str:
                    x_data[feat] = pd.factorize(x_data[feat])[0]
    if encode=='OneHot':
        # convert categorical features to one-hot encoded features
        for feat in x_data.columns:
            if type(x_data.iloc[0, x_data.columns.get_loc(feat)]) == str:
                x_data = pd.concat([x_data, pd.get_dummies(x_data[feat], prefix=feat)], axis=1)
                x_data = x_data.drop(feat, axis=1)
    if encode=='Frequency':
        # convert categorical features to frequency encoded features
        for feat in x_data.columns:
            if type(x_data.iloc[0, x_data.columns.get_loc(feat)]) == str:
                x_data[feat] = x_data[feat].map(x_data[feat].value_counts())/len(x_data)
        
                    
        
                
    
    # Body_Level goes to y_data
    y_data = ds['Body_Level']
    # transform the classes into integers
    y_data = pd.factorize(y_data)[0]
    
    
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
    print('\nNumber of classes in the dataset: ', len(np.unique(y_data)))


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
    for i in range(len(np.unique(y_data))): print('Class', i, ':', len(y_data[y_data == i]))


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

def visualize_categorical_data(x_data, y_data, normalize=True):
    '''
    For each categorical feature, plot a bar chart for each class.
    '''
    # get only the categorical features
    disc_feats = [feat for feat in x_data.columns if type(x_data.iloc[0, x_data.columns.get_loc(feat)]) == str]
    x_data_disc = x_data[disc_feats]
    
    # get unique values of y_data and use it to partition the dataset
    unique_y = np.unique(y_data)
    x_data_0 = x_data_disc[y_data == unique_y[0]]
    x_data_1 = x_data_disc[y_data == unique_y[1]]
    x_data_2 = x_data_disc[y_data == unique_y[2]]
    x_data_3 = x_data_disc[y_data == unique_y[3]]
    
    
    # plot each categorical feature (there are 8) in a bar chart with colors representing the classes
    fig, axs = plt.subplots(2, 4, figsize=(20, 20))
    for i, feature in enumerate(disc_feats):
        # get the unique values for the feature
        unique_vals = x_data_disc[feature].unique()

        # how many values in each class for each unique value
        counts = []
        for x_data_class in [x_data_0, x_data_1, x_data_2, x_data_3]:
            counts_class = [np.sum(x_data_class[feature] == val) for val in unique_vals]
            # normalize the class counts by the total number of samples in the class
            if normalize: counts_class = [count/len(x_data_class) for count in counts_class]
            counts.append(counts_class)                                 # list of value feature counts for each class
        
        # plot the grouped bar chart with colors for each class
        ax = axs.flatten()[i]                                           #  to select the ith subplot
        ax.set_title(feature)
        sns.barplot(x=np.repeat(unique_vals, 4), y=np.array(counts).flatten(), hue=np.tile(np.arange(4), len(unique_vals)), ax=ax, palette='tab10')
        
    plt.show()
    
    
  