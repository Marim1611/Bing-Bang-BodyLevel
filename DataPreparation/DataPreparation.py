import os
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as ss


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
        for feat in x_data.columns:
            x_data[feat] = x_data[feat].astype(float)
    
    if dummy:   x_data = pd.get_dummies(x_data)
    
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
    


def cramers_v(data, col1, col2):
    """ calculate Cramers V statistic for categorial-categorial association.
        Like lift from Big data but more sophisticated.
        This was modified from SO: https://stackoverflow.com/questions/46498455/categorical-features-correlation/46498792#46498792
    """
    confusion_matrix = pd.crosstab(data[col1], data[col2]).values
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))
    
def categorical_correlation_matrix(x_data):
    '''
    plot a correlation matrix for the categorical features in the dataset
    '''
    disc_feats = [feat for feat in x_data.columns if type(x_data.iloc[0, x_data.columns.get_loc(feat)]) == str]
    x_data_disc = x_data[disc_feats]
    
    corr = np.zeros((len(disc_feats), len(disc_feats)))
    for i in range(len(disc_feats)):
        for j in range(len(disc_feats)):
            corr[i, j] = cramers_v(x_data_disc, disc_feats[i], disc_feats[j])
    
    # now plot the correlation matrix
    fig, ax = plt.subplots(figsize=(10, 10))
    for i in range(len(corr)):
        for j in range(len(corr)):
             ax.text(j, i, '{:.2f}'.format(corr[i, j]), ha="center", va="center", color="w")
    ax.matshow(corr, cmap='bwr')
    plt.xticks(range(len(corr)), disc_feats, rotation=90)
    plt.yticks(range(len(corr)), disc_feats)
    plt.show()
    


def correlation_ratio(x_data, col1, col2):
    '''
    A measure of association between a categorical variable and a continuous variable.
    - Divide the continuous variable into N groups, based on the categories of the categorical variable.
    - Find the mean of the continuous variable in each group.
    - Compute a weighted variance of the means where the weights are the size of each group.
    - Divide the weighted variance by the variance of the continuous variable.
    
    It asks the question: If the category changes are the values of the continuous variable on average different?
    If this is zero then the average is the same over all categories so there is no association.
    '''
    categories = np.array(x_data[col1])
    values = np.array(x_data[col2])
    
    group_variances = 0
    for category in set(categories):
        group = values[np.where(categories == category)[0]]
        group_variances += len(group)*(np.mean(group)-np.mean(values))**2
    total_variance = sum((values-np.mean(values))**2)

    return (group_variances / total_variance)**.5

def mix_correlation_matrix(x_data):
    '''
    plot a correlation matrix for the categorical and continuous features in the dataset
    '''
    disc_feats = [feat for feat in x_data.columns if type(x_data.iloc[0, x_data.columns.get_loc(feat)]) == str]
    cont_feats = [feat for feat in x_data.columns if type(x_data.iloc[0, x_data.columns.get_loc(feat)]) != str]
    
    corr = np.zeros((len(disc_feats), len(cont_feats)))
    for i in range(len(disc_feats)):
        for j in range(len(cont_feats)):
            corr[i, j] = correlation_ratio(x_data, disc_feats[i], cont_feats[j])
    
    # now plot the correlation matrix
    fig, ax = plt.subplots(figsize=(10, 10))
    for i in range(len(corr)):
        for j in range(len(corr)):
             ax.text(j, i, '{:.2f}'.format(corr[i, j]), ha="center", va="center", color="w")
    ax.matshow(corr, cmap='bwr')
    plt.xticks(range(len(corr)), cont_feats, rotation=90)
    plt.yticks(range(len(corr)), disc_feats)
    plt.show()

def read_sample(path):
    '''
    A read_sample function for when the model is to be evaluated
    '''
    pass

