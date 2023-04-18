from imblearn.over_sampling import SMOTE,SMOTENC,SMOTEN,BorderlineSMOTE
from imblearn.under_sampling import NearMiss,RandomUnderSampler
import numpy as np
from utils import nice_table
from IPython.display import display
import pandas as pd
from sklearn.model_selection import cross_val_predict
from mlpath import mlquest as mlq

# columns names for the dataset and thier types 
CATEGORICAL=['Gender', 'H_Cal_Consump', 'Alcohol_Consump', 'Smoking','Food_Between_Meals', 'Fam_Hist', 'H_Cal_Burn', 'Transport']
NUMERICAL=['Age', 'Height', 'Weight', 'Veg_Consump', 'Water_Consump', 'Meal_Count','Phys_Act', 'Time_E_Dev']
MIXED=['Gender', 'Age', 'Height', 'Weight', 'H_Cal_Consump', 'Veg_Consump','Water_Consump', 'Alcohol_Consump', 'Smoking', 'Meal_Count','Food_Between_Meals', 'Fam_Hist', 'H_Cal_Burn', 'Phys_Act','Time_E_Dev', 'Transport']

def handle_class_imbalance(X,y, method=None,k=None, sampling_ratio=[1,1,1]):
    '''
    - this function handles the class imbalance problem in the dataset
    - takes the dataset as input X, y
    - takes required class imbalance handling method as input =>'SMOTE','SMOTENC',
      'SMOTEN',"BorderlineSMOTE","Under Sampling","Cost Sensitive"
    - k nearest neighbors used to define the neighborhood of samples in case of oversampling
    - sampling_ratio list contains ratios of samples in each class over majority class after resampling
      if  ratio is 1 then the number of samples in that class will be the same as majority class
    - returns balanced data set bal_X, bal_y or return weights of classes in case of cost sensitive 
    '''
    over_sampling_methods = ['SMOTE','SMOTENC','SMOTEN',"BorderlineSMOTE"]
    if method in over_sampling_methods:
        bal_X, bal_y = over_sampling(X,y,k, sampling_ratio,method)
        return bal_X,bal_y
    
    elif method == 'Under Sampling':
        bal_X, bal_y = under_sampling(X,y)
        return bal_X,bal_y
        
    elif method == 'Cost Sensitive':
        weights = cost_sensitive(y)
        return weights
    else:
        return X,y
    
#--------------------------------- Resampling Functions ----------------------------------------------

def over_sampling( X,y,k,sampling_ratio,method ):

    #----- handling the sampling strategy for each class
    sampling_strategy= dict() 
    Nmj = np.max(np.bincount(y)) # number of samples in the majority class

    #------- check if ratio is reasnoble
    for i in range(len(sampling_ratio)):
        sampling_strategy[i]= int(Nmj * sampling_ratio[i])
        if sampling_strategy[i] < np.bincount(y)[i]:
            print("incorrect ratio for class ",i)
            print("after over sampling he number of samples in a class should be >= to the original number of samples")
            return X,y
    sampling_strategy[3]=Nmj #unchanged
    #------------------------
    if method == "SMOTE":
        if X.columns.tolist() != NUMERICAL:
            print("SMOTE used with numerical features only")
            return X,y
        sm = SMOTE(k_neighbors=k , sampling_strategy=sampling_strategy)

    elif method == "SMOTEN":
        if X.columns.tolist() != CATEGORICAL:
            print("SMOTEN used with categorical features only")
            return X,y
        sm = SMOTEN(k_neighbors=k ,sampling_strategy=sampling_strategy)

    elif method == "SMOTENC": 
        if X.columns.tolist() != MIXED:
            print("SMOTENC used with both categorical and numerical features only")
            return X,y
        sm = SMOTENC(k_neighbors=k, sampling_strategy=sampling_strategy, categorical_features=[0,4,7,8,10,11,12,15])

    elif method == "BorderlineSMOTE":
        if X.columns.tolist() != NUMERICAL:
            print("BorderlineSMOTE used with numerical features only")
            return X,y
        sm = BorderlineSMOTE(k_neighbors=k, sampling_strategy=sampling_strategy)
    else:
        return X,y

    X_sm, y_sm = sm.fit_resample(X, y)
    return X_sm, y_sm

#------------------------------------------------------------
def under_sampling(X,y):
    rus = RandomUnderSampler(random_state=42)
    X_res, y_res = rus.fit_resample(X, y)
    return X_res, y_res

#------------------------------------------------------------
   
def cost_sensitive(y):
    weights=dict()
    for i in range(len(np.unique(y))): 
        weights[i]=1 /len(y[y == i])
    return weights

#--------------------------------Evaluation Functions --------------------------------

def show_difference(y,y_bal):
    #TODO: show the difference between the original and balanced data
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

#-----------------------------------------------------------------------------
def show_results(accuracies, methods=[], k=[] , sample_ratio=[], title=""):
    perf=dict()
    if len(methods): 
        perf["Method"]=[]
        for meth in methods: perf["Method"].append(meth)
    if len(k):
        perf["K"]=[]
        for i in k: perf["K"].append(i)
    if len(sample_ratio) != 0:
        perf["Sampling Ratio"]=[]
        for r in sample_ratio: perf["Sampling Ratio"].append(r)
    perf["Accuracy"]=[]
    for acc in accuracies: perf["Accuracy"].append(acc)
    
    df=pd.DataFrame(perf)
    #TODO write title for the data frame
    # df.style.set_table_attributes("style='display:inline'").set_caption(title)
    print(title)
    display(df)

#-----------------------------------------------------------------------------
def evaluate_class_imbalance_handler_over_methods(X,y ,clf , methods=[] , sample_ratio=[1,1,1], k=5):
    '''
    this function is used to evaluate the performance of the class imbalance handler over different methods
    and const value for k and sampling ratio
    '''
    accuracies = []
    for method in methods:
        if method != "Cost Sensitive":
            bal_x, bal_y = handle_class_imbalance(X, y, method=method,k=k, sampling_ratio=sample_ratio)
            clf.fit(bal_x, bal_y)
            y_pred = cross_val_predict(clf, bal_x, bal_y, cv=4)
            accuracies.append( np.mean(y_pred == bal_y))
        else:
            new_weights = handle_class_imbalance(X, y, method=method,k=k, sampling_ratio=sample_ratio)
            try:
                clf.set_params(class_weight=new_weights)
            except:
                print("this classifier has no parameter called class_weight")
            
            clf.fit(X, y)
            y_pred = cross_val_predict(clf, X, y, cv=4)
            accuracies.append( np.mean(y_pred == y))

    show_results(accuracies, methods, title="K = "+str(k)+", Sampling Ratio = "+str(sample_ratio))

#---------------------------------------------------------------------------------

def evaluate_const_k_diff_sample_ratios(X,y ,clf , method , k=5, sample_ratios=[]):
    '''
    this function is used to evaluate the performance of the class imbalance handler for one
      method, const value for k and multiple values of sampling ratio
    '''
    accuracies = []
    for r in sample_ratios:
        if method != "Cost Sensitive":
            bal_x, bal_y = handle_class_imbalance(X, y, method=method,k=k, sampling_ratio=r)
            clf.fit(bal_x, bal_y)
            y_pred = cross_val_predict(clf, bal_x, bal_y, cv=4)
            accuracies.append( np.mean(y_pred == bal_y))
        else:
            new_weights = handle_class_imbalance(X, y, method=method,k=k, sampling_ratio=r)
            try:
                clf.set_params(class_weight=new_weights)
            except:
                print("this classifier has no parameter called class_weight")
            
            clf.fit(X, y)
            y_pred = cross_val_predict(clf, X, y, cv=4)
            accuracies.append( np.mean(y_pred == y))        
    show_results(accuracies, sample_ratio=sample_ratios,title="Method = "+method+", K = "+str(k))

#------------------------------------------------------------------------------------

def evaluate_const_sample_ratios_diff_k(X,y ,clf , method , Ks, sample_ratio=[1,1,1]):
    '''
    this function is used to evaluate the performance of the class imbalance handler for one
      method, const value for sampling ratio and multiple values of k
    '''
    accuracies = []
    for k in Ks:
        if method != "Cost Sensitive":
            bal_x, bal_y = handle_class_imbalance(X, y, method=method,k=k, sampling_ratio=sample_ratio)
            clf.fit(bal_x, bal_y)
            y_pred = cross_val_predict(clf, bal_x, bal_y, cv=4)
            accuracies.append( np.mean(y_pred == bal_y))
        else:
            new_weights = handle_class_imbalance(X, y, method=method,k=k, sampling_ratio=sample_ratio)
            try:
                clf.set_params(class_weight=new_weights)
            except:
                print("this classifier has no parameter called class_weight")
            
            clf.fit(X, y)
            y_pred = cross_val_predict(clf, X, y, cv=4)
            accuracies.append( np.mean(y_pred == y))  

    show_results(accuracies, k=Ks, title="Method = "+method+", Sampling Ratio = "+str(sample_ratio))

    
 