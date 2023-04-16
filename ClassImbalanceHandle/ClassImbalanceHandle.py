from imblearn.over_sampling import SMOTE,SMOTENC,SMOTEN,BorderlineSMOTE
from imblearn.under_sampling import NearMiss,RandomUnderSampler
import numpy as np


def handle_class_imbalance(X,y, method=None,k=None, sampling_ratio=[0,0,0]):
    '''
    - this function handles the class imbalance problem in the dataset
    - takes the dataset as input X, y
    - takes required class imbalance handling method as input =>'SMOTE','SMOTENC',
      'SMOTEN',"BorderlineSMOTE","under","cost"
    - k nearest neighbors used to define the neighborhood of samples in case of oversampling
    - sampling_ratio list contains ratios of samples in each class over majority class after resampling
      if  ratio is 0 then the number of samples in that class will be the same as majority class
    - returns balanced data set bal_X, bal_y or return weights of classes in case of cost sensitive 
    '''
    over_sampling_methods = ['SMOTE','SMOTENC','SMOTEN',"BorderlineSMOTE"]
    if method in over_sampling_methods:
        bal_X, bal_y = over_sampling(X,y,k, sampling_ratio,method)
        return bal_X,bal_y
    
    elif method == 'under':
        bal_X, bal_y = under_sampling(X,y)
        return bal_X,bal_y
        
    elif method == 'cost':
        weights = cost_sensitive(y)
        return weights
    else:
        return X,y
def over_sampling( X,y,k,sampling_ratio,method ):
    categorical_features=['Gender', 'H_Cal_Consump', 'Alcohol_Consump', 'Smoking','Food_Between_Meals', 'Fam_Hist', 'H_Cal_Burn', 'Transport']
    numerical_features=['Age', 'Height', 'Weight', 'Veg_Consump', 'Water_Consump', 'Meal_Count','Phys_Act', 'Time_E_Dev']

    #----- handling the sampling strategy for each class
    
    sampling_strategy= dict() 
    Nmj = np.max(np.bincount(y)) # number of samples in the majority class
    #check if ratio is reasnoble
    for i in range(len(sampling_ratio)):
        sampling_strategy[i]= int(Nmj * sampling_ratio[i]) if sampling_ratio[i] != 0 else Nmj
        if sampling_strategy[i] < np.bincount(y)[i]:
            print("incorrect ratio for class ",i)
            print("after over sampling he number of samples in a class should be >= to the original number of samples")
            return X,y
    sampling_strategy[3]=Nmj #unchanged
    #------------------------
        
    if method == "SMOTE":
        if X.columns.tolist() != numerical_features:
            print("SMOTE used with numerical features only")
            return X,y
        sm = SMOTE(k_neighbors=k , sampling_strategy=sampling_strategy)

    elif method == "SMOTEN":
        if X.columns.tolist() != categorical_features:
            print("SMOTE used with categorical features only")
            return X,y
        sm = SMOTEN(k_neighbors=5 ,sampling_strategy=sampling_strategy)

    elif method == "SMOTENC":            
        sm = SMOTENC(k_neighbors=k, sampling_strategy=sampling_strategy, categorical_features=[0,4,7,8,10,11,12,15])

    elif method == "BorderlineSMOTE":
        if X.columns.tolist() != numerical_features:
            print("BorderlineSMOTE used with numerical features only")
            return X,y
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