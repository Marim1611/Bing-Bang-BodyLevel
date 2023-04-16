from imblearn.over_sampling import SMOTE,SMOTENC,SMOTEN,BorderlineSMOTE
from imblearn.under_sampling import NearMiss,RandomUnderSampler
import numpy as np

def handle_class_imbalance(X,y, method=None,k=None, sampling_strategy=None):
    '''
    - this function handles the class imbalance problem in the dataset
    - takes the dataset as input X, y
    - takes required class imbalance handling method as input =>'SMOTE','SMOTENC',
      'SMOTEN',"BorderlineSMOTE","under","cost"
    - k nearest neighbors used to define the neighborhood of samples in case of oversampling
    - sampling_strategy: float corresponds to the desired ratio of the number of samples in the minority class over
      the number of samples in the majority class after resampling. 
    - returns balanced data set bal_X, bal_y or return weights of classes in case of cost sensitive 
    '''
    over_sampling_methods = ['SMOTE','SMOTENC','SMOTEN',"BorderlineSMOTE"]
    if method in over_sampling_methods:
        bal_X, bal_y = over_sampling(X,y,k, sampling_strategy,method)
        return bal_X,bal_y
    
    elif method == 'under':
        bal_X, bal_y = under_sampling(X,y)
        return bal_X,bal_y
        
    elif method == 'cost':
        weights = cost_sensitive(y)
        return weights
    else:
        return X,y
def over_sampling( X,y,k,sampling_strategy,method ):

    if method == "SMOTE":
        sm = SMOTE(k_neighbors=k , sampling_strategy=sampling_strategy)

    elif method == "SMOTEN":
        sm = SMOTEN(k_neighbors=5 ,sampling_strategy=sampling_strategy)

    elif method == "SMOTENC":
        sm = SMOTENC(k_neighbors=k, sampling_strategy=sampling_strategy, categorical_features=[0,4,7,8,10,11,12,15])

    elif method == "BorderlineSMOTE":
        sm = BorderlineSMOTE(k_neighbors=k, sampling_strategy=sampling_strategy)

    X_sm, y_sm = sm.fit_resample(X, y)
    return X_sm, y_sm


def under_sampling(X,y):
    rus = RandomUnderSampler(random_state=42)
    X_res, y_res = rus.fit_resample(X, y)
    return X_res, y_res
    
def cost_sensitive(y):
    weights=dict()
    for i in range(len(np.unique(y))): 
        weights[i]=1 /len(y[y == i])
    return weights

def show_difference(y,y_bal):
    # plt.pie(y)
    # plt.title("before balancing")
    # plt.show()
    # plt.pie(y_bal)
    # plt.title("after balancing")
    # plt.show()
    print('\nNumber of samples in each class before resampling:\n')
    for i in range(len(np.unique(y))): print('Class', i, ':', len(y[y == i]))
    print('\nNumber of samples in each class after resampling:\n')
    for i in range(len(np.unique(y_bal))): print('Class', i, ':', len(y_bal[y_bal == i]))