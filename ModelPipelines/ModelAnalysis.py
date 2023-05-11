from sklearn.model_selection import StratifiedKFold, validation_curve, learning_curve
from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("../../")
from utils import nice_table
from IPython.display import display, HTML, Markdown
import warnings

def recursive_feature_elimination(clf, min_feats, cv, x_data_d, y_data_d, display=True):
    '''
    Recursive feature elimination recursively removes the weakest feature as determined by the given classifier.
    It stops when the desired number of features is reached or accuracy is no longer improving.
    '''
    rfecv = RFECV(estimator=clf, cv=StratifiedKFold(cv), scoring="accuracy", min_features_to_select=min_feats)
    rfecv.fit(x_data_d, y_data_d)
    opt_feats = rfecv.get_feature_names_out(x_data_d.columns)
    opt_feats = [feat for _, feat in sorted(zip(rfecv.ranking_, opt_feats))]
    print(f"Features to keep {opt_feats} with ranks {sorted(rfecv.ranking_[0:len(opt_feats)])}")
    
    if display:
        plt.rcParams['figure.dpi'] = 300
        plt.style.use('dark_background')
        plt.figure(figsize=(10, 6))
        plt.xlabel("Number of features selected")
        plt.ylabel("Mean test accuracy")
        plt.errorbar(range(min_feats, len(rfecv.cv_results_["mean_test_score"])+min_feats), 
                    rfecv.cv_results_["mean_test_score"])
        # draw a dashed red line through the selected number of features
        plt.axvline(x=rfecv.n_features_, color='r', linestyle='--')
        
        plt.title("Recursive Feature Elimination")
   
        plt.show()
    
    # choose the best features
    x_data_d = x_data_d[rfecv.get_feature_names_out(x_data_d.columns)]
    return x_data_d


def test_log_linearity(clf, class_index,  x_data_d, y_data_d):
    '''
    Test if the log odds are linearly related to the features to assess logistic regression.
    '''
    # print prediction probabilities
    clf.fit(x_data_d, y_data_d)
    probs = clf.predict_proba(x_data_d)
    # extract first column
    probs = probs[:, 3]
    log_odds = np.log(probs / (1 - probs))
    
    # get the weights of the model
    weights = clf.coef_[class_index]
    # get the intercept
    intercept = clf.intercept_[class_index]

    # make a 2x4 plot for all continuous features in x_data_d where in each plot the x-axis is the feature and the y-axis is the log odds
    plt.rcParams['figure.dpi'] = 300
    plt.style.use('dark_background')
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    for i, col in enumerate(x_data_d.columns):
        # check if column is continuous
        if x_data_d[col].dtype == np.float64:
            # plot
            axs[i // 4, i % 4].scatter(x_data_d[col], log_odds, s=1.5)
            axs[i // 4, i % 4].set_title(col)
            axs[i // 4, i % 4].set_xlabel(col)
            axs[i // 4, i % 4].set_ylabel("Log Odds")
            # plot the logistic regression line
            x = np.linspace(x_data_d[col].min(), x_data_d[col].max(), 100)
            y = weights[i] * x + intercept
            axs[i // 4, i % 4].plot(x, y, color='yellow', linewidth=2, alpha=0.7)
            
    plt.show()
    

def log_weights_analysis(clf,x_data_d):
    '''
    Display weights of each class for logistic regression.
    '''
    # get the weights of the model
    weights= clf.coef_

    plt.rcParams['figure.dpi'] = 300
    plt.style.use('dark_background')
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 8))
    for i in range(2):
        for j in range(2):
            axs[i, j].bar(range(len(weights[i])), weights[i], width=0.3)
            axs[i, j].axhline(y=0)
            axs[i, j].set_title(f"Body Level {i}", fontsize=20)
            axs[i, j].set_xlabel("Feature")
            axs[i, j].set_ylabel("Importance")
            # make x-ticks be the feature names
            axs[i, j].set_xticks(range(len(weights[i])))
            cols = x_data_d.columns
            cols = [col.replace("_", "\n") for col in cols]
            axs[i, j].set_xticklabels(cols, rotation=90)
            axs[i, j].tick_params(axis='both', which='major', labelsize=7)
            plt.subplots_adjust(hspace=0.5)
    plt.show()


def vc_dimension_check(clf, x_data_d):
    '''
    Given a model that provides a coef_ and intercept_ attribute, check if the VC bound is satisfied.
    '''
    dvc =  (np.sum([param.size for param in clf.coef_]) +  np.sum([param.size for param in clf.intercept_])) + 1
    N = x_data_d.shape[0]
    if 10 * dvc > N:
        analysis = f'''<font size=4>By estimating the VC dimension of the model, 
                    we have $d_{{vc}}={dvc}$. 
                    Since, $N={N}$, here it holds that that 
                    $$N < 10d_{{vc}}$$ 
                    Hence, generalization is not guaranteed and its advised to reduce the model complexity.
                    </font>
                    '''
        display(Markdown(analysis))
    else:
        analysis = f'''<font size=4>By estimating the VC dimension of the model, 
                    we have $d_{{vc}}={dvc}$. 
                    Since, $N={N}$, it holds that 
                    $$N \\geq 10d_{{vc}}$$
                    Hence, model is expected to have no issues with generalization.
                    </font>
                    '''
        display(Markdown(analysis))

def show_hyperparams(clf):
    '''
    Print the hyperparameters of a model.
    '''
    return display(HTML(nice_table(clf.get_params(), title="Hyperparameters")))


def validation_curves(clf,x_data,y_data,cv, hyperparameters):
    '''
    Plot the validation curve for a given model and hyperparameter.
    '''

    categorical = False

    plt.rcParams['figure.dpi'] = 300
    plt.style.use('dark_background') 

    nrows = int(np.ceil(len(hyperparameters)/2))
    ncols= 2 if len(hyperparameters) > 1 else 1

    fig, axs = plt.subplots(nrows, ncols, figsize=(20, 10))
    fig.subplots_adjust(top=1.0)

    if len(hyperparameters) % 2 == 1 and len(hyperparameters) > 1:
        fig.delaxes(axs[-1, -1])

    if len(hyperparameters) == 1:
        fig.set_size_inches(10,5)

    for index, param_name in enumerate(hyperparameters.keys()):
        param_range = hyperparameters[param_name]

        if isinstance(param_range[0], str):
            categorical = True

        train_scores, test_scores = validation_curve(clf, x_data, y_data, param_name=param_name, param_range=param_range,
                                    cv=StratifiedKFold(cv), scoring="accuracy", n_jobs=4)
        
        train_scores= 1-np.mean(train_scores, axis=1)
        test_scores= 1-np.mean(test_scores, axis=1)

        if len(hyperparameters) == 1:
            ax= axs
        elif len(hyperparameters) == 2:
            ax = axs[index]
        else:
            ax = axs[index // 2, index % 2]

        if categorical:
            # check for nan values and remove them
            nan_indices= np.argwhere(np.isnan(train_scores)) 

            if len(nan_indices) > 0:
                nan_values= [param_range[i] for i in nan_indices.flatten()]
                
                warnings.warn(f"Validation curve for {param_name} contains NaN values for values {nan_values}. These values will be removed.")

                train_scores= np.delete(train_scores, nan_indices)
                test_scores= np.delete(test_scores, nan_indices)
                param_range= np.delete(param_range, nan_indices)

            x_axis= np.arange(len(param_range))            
            ax.bar(x_axis-0.2/2, train_scores,width=0.2, label="Training Error")
            ax.bar(x_axis+0.2/2, test_scores,width=0.2, label="Validation Error")
            ax.set_xticks(x_axis)
            ax.set_xticklabels(param_range)

        else:

            ax.plot(param_range, train_scores, label="Training Error")
            ax.plot(param_range, test_scores, label="Validation Error")
            
            optimal_param= optimal_hyperparameter(train_scores, test_scores, param_range)
            ax.axvline(optimal_param, color='red', linestyle='--', label="Optimal "+param_name+\
                       " is around "+str(np.round(optimal_param,2)))

        ax.set_title(f"Validation Curve for {param_name}")
        ax.set_xlabel(param_name)
        ax.set_ylabel("Error")
        ax.legend(loc="best")  

        
    plt.show()


def optimal_hyperparameter(train_scores, test_scores, parameter):
    '''
    Given the training and validation scores, find the most probable optimal hyperparameter:

    The difference between the current and previous train scores serves as an estimate of the gradient,
    while the difference between the previous and current test scores serves as an estimate of the curvature,

    If the gradient is small (less than or equal to 0.001) and the curvature is large (greater than or equal to 0.001),
    then the optimal hyperparameter is updated to the next value.

    This update is performed using the gradient information, which ensures that the algorithm moves in the direction
    of steepest descent towards the optimal parameter value.
    '''

    optimal_param = parameter[0]  
    train_score_prev = train_scores[0]  
    test_score_prev = test_scores[0]  
    train_score_curr = 0  
    test_score_curr = 0  

    patience_counter = 0
    patience=4

    for i in range(1, len(train_scores)):
        train_score_curr = np.mean(train_scores[:i])  
        test_score_curr = np.mean(test_scores[:i])  

        if train_score_curr <= train_score_prev and test_score_prev >= test_score_curr and i+1 < len(train_scores):
            if patience_counter < patience:
                patience_counter += 1
                optimal_param = parameter[i + 1] 
            else:
                break

        train_score_prev = train_score_curr  
        test_score_prev = test_score_curr 

    return optimal_param
    

def learning_curves(clf, x_data, y_data, cv,N):

    '''
    Plot the learning curve for a given model.
    '''
    train_sizes, train_scores, test_scores = learning_curve(clf, x_data, y_data, cv=StratifiedKFold(cv), n_jobs=4, 
                                                            train_sizes=N, scoring="accuracy")

    plt.rcParams['figure.dpi'] = 300
    plt.style.use('dark_background')
    plt.figure(figsize=(10,5))
    plt.xlabel("N")
    plt.ylabel("Error")
    plt.title("Learning Curve")
    plt.plot(train_sizes, 1- train_scores.mean(axis=1), markersize=5, label='Training Error' )
    plt.plot(train_sizes, 1- test_scores.mean(axis=1), markersize=5, label='Validation Error' )
    plt.legend(loc="best")
    plt.show()
