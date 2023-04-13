import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as ss          


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

