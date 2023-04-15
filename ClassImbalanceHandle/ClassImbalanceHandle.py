from imblearn.over_sampling import SMOTE,SMOTENC,SMOTEN
from imblearn.under_sampling import NearMiss,RandomUnderSampler
import numpy as np

def handle_class_imbalance(X,y,kind=None, method=None,k=None):
    '''
    this function handles the class imbalance problem in the dataset
    takes the dataset as input X, y
    takes kind of features as input => 'Categorical', 'Numerical', default: mixed
    takes type of class imbalance handling method as input => 'over', 'under', 'cost'
    k nearest neighbors used to define the neighborhood of samples in case of oversampling
    returns balanced data set bal_X, bal_y
     or return weights of classes in case of cost sensitive 
    '''
    if method == 'over':
        bal_X, bal_y = over_sampling(X,y,k, kind)
        return bal_X,bal_y
    
    elif method == 'under':
        bal_X, bal_y = under_sampling(X,y)
        return bal_X,bal_y
        
    elif method == 'cost':
        weights = cost_sensitive(y)
        return weights
    
def over_sampling( X,y,k,kind ):
    
    if kind == "Numerical":
        sm = SMOTE(k_neighbors=k)
        X_sm, y_sm = sm.fit_resample(X, y)

    elif kind == "Categorical":
        sm = SMOTEN(k_neighbors=5)
        X_sm, y_sm = sm.fit_resample(X, y)

    else:
        # categorical_features=X.loc[:,["Food_Between_Meals","Gender","Smoking","Alcohol_Consump","H_Cal_Consump","H_Cal_Burn", "Fam_Hist","Transport"]] 
        sm = SMOTENC(k_neighbors=k, categorical_features=[0,4,7,8,10,11,12,15])
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