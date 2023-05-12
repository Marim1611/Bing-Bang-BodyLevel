from imblearn.over_sampling import SMOTE,SMOTENC,SMOTEN,BorderlineSMOTE
from imblearn.under_sampling import NearMiss,RandomUnderSampler
import numpy as np
from utils import nice_table
from IPython.display import display
import pandas as pd
from sklearn.model_selection import cross_val_predict
from mlpath import mlquest as mlq
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score


# columns names for the dataset and thier types 
CATEGORICAL=['Gender', 'H_Cal_Consump', 'Alcohol_Consump', 'Smoking','Food_Between_Meals', 'Fam_Hist', 'H_Cal_Burn', 'Transport']
NUMERICAL=['Age', 'Height', 'Weight', 'Veg_Consump', 'Water_Consump', 'Meal_Count','Phys_Act', 'Time_E_Dev']
MIXED=['Gender', 'Age', 'Height', 'Weight', 'H_Cal_Consump', 'Veg_Consump','Water_Consump', 'Alcohol_Consump', 'Smoking', 'Meal_Count','Food_Between_Meals', 'Fam_Hist', 'H_Cal_Burn', 'Phys_Act','Time_E_Dev', 'Transport']
COLOR= '#ECAF93' # color for the plots

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

#-------------------------------- Visualization Functions --------------------------------

def show_difference(y,y_bal):
    labels=['Class 0', 'Class 1', 'Class 2', 'Class 3'] 
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,10)) #ax1,ax2 refer to your two pies
    ax1.pie(np.bincount(y),labels = labels,autopct = '%1.1f%%') #plot first pie
    # TODO ax1.title('Before Balancing')
    ax2.pie(np.bincount(y_bal) ,labels = labels,autopct = '%1.1f%%') #plot second pie
    # ax2.title('After Balancing')
    fig.show()
   
#-----------------------------------------------------------------------------
def plot_results(metric ,methods=None, k=None , sample_ratio=None, title=""):

    if methods: labels = methods
    if sample_ratio: 
        for i in range(len(sample_ratio)):
            sample_ratio[i]= map(str, sample_ratio[i])
            sample_ratio[i]=', '.join(sample_ratio[i])
        labels = sample_ratio
    if k: labels = k
    
    plt.style.use('dark_background')
    plt.bar(labels, metric, color =COLOR,width = 0.4)
    plt.xticks(fontsize=7)
    plt.title(title,fontsize=10)
    plt.ylabel('Weighted F1-Score',fontsize=10)
    plt.figure(figsize=(6,5))
    plt.show()
 
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
    # df.style.set_table_attributes("style='display:inline'").set_caption(title)
    print(title)
    display(df)

#------------------------------------- Evaluation Functions ----------------------------------------
def evaluate_class_imbalance_handler_over_methods(X,y ,clf , methods=[] , sample_ratio=[1,1,1], k=5):
    '''
    this function is used to evaluate the performance of the class imbalance handler over different methods
    and const value for k and sampling ratio
    '''
    accuracies = []
    weighted_f1_scores = []
    for method in methods:
        if method != "Cost Sensitive":
            bal_x, bal_y = handle_class_imbalance(X, y, method=method,k=k, sampling_ratio=sample_ratio)
            clf.fit(bal_x, bal_y)
            y_pred = cross_val_predict(clf, bal_x, bal_y, cv=4)
            accuracies.append( np.mean(y_pred == bal_y))
            weighted_f1_scores.append(f1_score(bal_y, y_pred, average='weighted'))
        else:
            new_weights = handle_class_imbalance(X, y, method=method,k=k, sampling_ratio=sample_ratio)
            try:
                clf.set_params(class_weight=new_weights)
            except:
                print("this classifier has no parameter called class_weight")
            
            clf.fit(X, y)
            y_pred = cross_val_predict(clf, X, y, cv=4)
            accuracies.append( np.mean(y_pred == y))
            weighted_f1_scores.append(f1_score(y, y_pred, average='weighted'))
    
    plot_results(weighted_f1_scores, methods, title="K = "+str(k)+", Sampling Ratio = "+str(sample_ratio))

#---------------------------------------------------------------------------------

def evaluate_const_k_diff_sample_ratios(X,y ,clf , method , k=5, sample_ratios=[]):
    '''
    this function is used to evaluate the performance of the class imbalance handler for one
      method, const value for k and multiple values of sampling ratio
    '''
    accuracies = []
    weighted_f1_scores = []
    for r in sample_ratios:
        if method != "Cost Sensitive":
            bal_x, bal_y = handle_class_imbalance(X, y, method=method,k=k, sampling_ratio=r)
            clf.fit(bal_x, bal_y)
            y_pred = cross_val_predict(clf, bal_x, bal_y, cv=4)
            accuracies.append( np.mean(y_pred == bal_y))
            weighted_f1_scores.append(f1_score(bal_y, y_pred, average='weighted'))  
        else:
            new_weights = handle_class_imbalance(X, y, method=method,k=k, sampling_ratio=r)
            try:
                clf.set_params(class_weight=new_weights)
            except:
                print("this classifier has no parameter called class_weight")
            
            clf.fit(X, y)
            y_pred = cross_val_predict(clf, X, y, cv=4)
            accuracies.append( np.mean(y_pred == y))
            weighted_f1_scores.append(f1_score(y, y_pred, average='weighted'))   

    return weighted_f1_scores

#------------------------------------------------------------------------------------

def evaluate_const_sample_ratios_diff_k(X,y ,clf , method , Ks, sample_ratio=[1,1,1]):
    '''
    this function is used to evaluate the performance of the class imbalance handler for one
      method, const value for sampling ratio and multiple values of k
    '''
    accuracies = []
    weighted_f1_scores = []
    for k in Ks:
        if method != "Cost Sensitive":
            bal_x, bal_y = handle_class_imbalance(X, y, method=method,k=k, sampling_ratio=sample_ratio)
            clf.fit(bal_x, bal_y)
            y_pred = cross_val_predict(clf, bal_x, bal_y, cv=4)
            accuracies.append( np.mean(y_pred == bal_y))
            weighted_f1_scores.append(f1_score(bal_y, y_pred, average='weighted'))
        else:
            new_weights = handle_class_imbalance(X, y, method=method,k=k, sampling_ratio=sample_ratio)
            try:
                clf.set_params(class_weight=new_weights)
            except:
                print("this classifier has no parameter called class_weight")
            
            clf.fit(X, y)
            y_pred = cross_val_predict(clf, X, y, cv=4)
            accuracies.append( np.mean(y_pred == y))  
            weighted_f1_scores.append(f1_score(y, y_pred, average='weighted'))
    return weighted_f1_scores

def plot_different_evaluations( X,y, clf, methods, sample_ratios , const_sample_ratio,const_k, Ks):
    '''
    This function is used to plot the results of the evaluation of the class imbalance handler
    over different methods, const value for k and different sampling ratios and const value for
    sampling ratio and different values of k
    '''
    Scores= []
    Labels = []
    titles = []
    x_labels = []

    for method in methods:
        scores1 =evaluate_const_sample_ratios_diff_k(X,y ,clf , method , Ks, const_sample_ratio)
        Scores.append( scores1)
        Labels.append( Ks)
        titles.append("Method: "+method+", Sampling Ratio = "+str(const_sample_ratio))
        x_labels.append("K")
        scores2 =evaluate_const_k_diff_sample_ratios(X,y ,clf , method , const_k, sample_ratios)
        Scores.append( scores2)
        Labels.append( sample_ratios)
        titles.append("Method: "+method+", K = "+str(const_k))
        x_labels.append("Sampling Ratio")
    
    #--- plotting
    plt.rcParams['figure.dpi'] = 300
    plt.style.use('dark_background')
    plt.suptitle("Performance of different class imbalance handling methods", fontsize=15)

    fig, axs = plt.subplots(nrows= len(methods), ncols=2, figsize=(20, 8))
    k=0
    for i in range(len(methods)):
        for j in range(2):
            axs[i, j].bar(range(len(Scores[k])), Scores[k], width=0.3, color=COLOR)
            # increase margin between the bar and the top of the plot
            axs[i, j].set_ylim(top=max(Scores[k])+0.1)
            axs[i, j].axhline(y=0)
            axs[i, j].set_title(titles[k], fontsize=12)
            axs[i, j].set_xlabel(x_labels[k])
            axs[i, j].set_ylabel("Weighted F1 Score")
            axs[i, j].set_xticks(range(len(Scores[k])))
            axs[i, j].set_xticklabels(Labels[k], rotation=90)
            axs[i, j].tick_params(axis='both', which='major', labelsize=8)
            # write the value on top of each bar
            for index, value in enumerate(Scores[k]):
                axs[i, j].text(index-0.1, value+0.01, str(round(value, 3)))
            plt.subplots_adjust(hspace=0.7)
            k+=1

    plt.show()

   


    


        
        


 