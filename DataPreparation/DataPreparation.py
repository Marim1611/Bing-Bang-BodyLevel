import os
import matplotlib.pyplot as plt
import itertools
import pandas as pd
import numpy as np
import seaborn as sns
from IPython.display import display, HTML, Markdown, Latex
import sys
sys.path.append('../')
from utils import nice_table


def read_data(kind=None, encode=None, split="all", standardize=True ,**kwargs):
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
    
    
    if standardize:
        # standardize the numerical features
        for feat in x_data.columns:
            if type(x_data.iloc[0, x_data.columns.get_loc(feat)]) != str:
                x_data[feat] = (x_data[feat] - x_data[feat].mean())/x_data[feat].std()
                    
    
    # Body_Level goes to y_data
    y_data = ds['Body_Level']
    # transform the classes into integers
    y_data = pd.factorize(y_data)[0]
        
    return x_data, y_data

    
def basic_info(x_data, y_data):
    '''
    prints basic info about the dataset like the number of rows, columns, features and possible classes
    '''
    dic = {'Number of samples': len(x_data), 'Number of features': len(x_data.columns), 'Number of classes': len(np.unique(y_data))}
    display(HTML(nice_table(dic, title='Basic Counts')))
    column_dict = {}
    for column in x_data.columns:   column_dict[column] = ''
    display(HTML(nice_table(column_dict, title='Features')))


def prior_distribution(y_data):
    '''
    plots the prior distribution of the dataset which is helpful for class imbalance
    '''
    # plot the prior distribution of the dataset
    plt.figure(figsize=(10, 5))
    # make a bar chart for the unique values of y_data
    plt.bar(np.unique(y_data), np.bincount(y_data), color='aqua', edgecolor='black', alpha=0.7)
    plt.title('Prior distribution of the dataset')
    plt.xlabel('Body Level')
    plt.ylabel('Number of samples')
    plt.show()
    
    class_dict = {}
    for i in range(len(np.unique(y_data))):
        class_dict['Class '+str(i)] = len(y_data[y_data == i])
    display(HTML(nice_table(class_dict, title='Number of samples in each class')))


def features_histograms(x_data):
    '''
    Plot a 4x4 grid of histograms for each feature in the dataset (there are 16 features).
    Also print the number of unique values of each feature and its kind.
    '''
    # plot a 4x4 grid of histograms for each feature in the dataset
    plt.style.use('dark_background')
    fig, axs = plt.subplots(4, 4, figsize=(20, 20))
    plt.rcParams['figure.dpi'] = 200
    for i in range(4):
        for j in range(4):
            # check if its a categorical or numerical feature
            if type(x_data.iloc[0, i*4+j]) == str:
                # if categorical, plot a bar chart
                names_num = x_data.iloc[:, i*4+j].value_counts().index
                axs[i, j].bar(names_num, x_data.iloc[:, i*4+j].value_counts(), color='aqua', edgecolor='black', alpha=0.7)
            else:
                sns.kdeplot(x_data.iloc[:, i*4+j], color='aqua',  alpha=0.7, ax=axs[i, j])       
                axs[i, j].set_xlabel('')
                axs[i, j].set_ylabel('')

                         
            axs[i, j].set_title(x_data.columns[i*4+j])
    plt.show()
    
    # print number of unique values of each feature
    feats = {}
    c = 0
    for i in range(len(x_data.columns)):
        feature_type = type(x_data.iloc[0, i])
        if feature_type == str:
            feats[x_data.columns[i]] = str(len(x_data.iloc[:, i].unique()))
            c+=1
        else:
            feats[x_data.columns[i]] = "numerical"
        
        feats = dict(sorted(feats.items(), key=lambda item: item[1]))
    display(HTML(nice_table(feats, title='Number of unique values of each feature')))
    
    stats = {"Number of Categorical": c, "Number of Numerical": len(x_data.columns)-c}
    display(HTML(nice_table(stats, title='Features Statistics')))

        

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
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
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
        
        # plot the grouped bar chart with same color for each class
        ax = axs.flatten()[i]                                           #  to select the ith subplot
        ax.set_title(feature)
        sns.barplot(x=np.repeat(unique_vals, 4), y=np.array(counts).flatten(), hue=np.tile(np.arange(4), len(unique_vals)), ax=ax, palette="dark:aqua")
        
    plt.show()
    
    
def HoeffdingCheck(dataset, ratio=None, ϵ=None, δ=None):
    '''
    Given a two of the three parameters:
    N (validation set size), ϵ (allowed gen. error), σ (upper bound probability of deviation) , 
    this function returns the third.
    '''
    if ratio:
        N = int(len(dataset) * ratio)
    
    assert [ratio, ϵ, δ].count(None) == 1, "You must provide two of the three parameters: N, ϵ, δ"
    if δ is None:
        δ = 2 * np.exp(-2 * ϵ**2 * N)
        if δ >=1: 
            N, ϵ, δ = int(N), round(ϵ, 3), round(δ, 3)
            analysis = f'''<font size=4>Hoeffding's Inequality states:
                    $$P[|E_{{out}}(g)-E_{{test}}(g)| \leq \epsilon] \geq 1-2e^{{-2N_{{test}}\epsilon^2}}$$
                    If we use validation set of size ${ratio}N_{{train}}={N}$ then with $\epsilon={ϵ}$ we have 
                    $$P[|E_{{out}}(g)-E_{{test}}(g)| \leq {ϵ}] \geq {1-δ}$$
                    In other words, 
                    There are no generalization guarantees.
                    </font>
                    '''
            display(Markdown(analysis))
            return None
    
    if ϵ is None:
        ϵ = np.sqrt(np.log(2/δ)/(2*N))
    
    if ratio is None:
        N = np.log(2/δ)/(2*ϵ**2)
        ratio = N/len(dataset)
    
    # round to 3 decimal places
    N, ϵ, δ = int(N), round(ϵ, 3), round(δ, 3)
    
    analysis = f'''<font size=4>Hoeffding's Inequality states:
                    $$P[|E_{{out}}(g)-E_{{test}}(g)| \leq \epsilon] \geq 1-2e^{{-2N_{{test}}\epsilon^2}}$$
                    If we use validation set of size ${ratio}N_{{train}}={N}$ then with $\epsilon={ϵ}$ we have 
                    $$P[|E_{{out}}(g)-E_{{test}}(g)| \leq {ϵ}] \geq {1-δ}$$
                    In other words, 
                    with probability at least ${1-δ}$, the generalization error of our model will be at most {ϵ} given a validation set of size {N}.
                    </font>
                    '''
    display(Markdown(analysis))


                

def convey_insights(bullets_arr):
    '''
    Give it a bullet points array, give you bullet points in markdown for insights.
    '''
    # make a markdown string with the bullets
    markdown_str = '<h3><font color="pink" size=5>Insights</font></h3> <font size=4>\n'
    
    for bullet in bullets_arr:
        markdown_str += '<font color="pink">✦</font> ' + bullet + '<br><br>'
    # display the markdown string
    markdown_str += '</font>'
    display(Markdown(markdown_str))
    

def read_sample(path):
    '''
    A read_sample function for when the model is to be evaluated
    '''
    pass
