from sklearn.model_selection import StratifiedKFold, validation_curve, learning_curve
from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("../../")
from utils import nice_table
from IPython.display import display, HTML


def recursive_feature_elimination(clf, min_feats, cv, x_data_d, y_data_d, display=True):
    '''
    Recursive feature elimination recursively removes the weakest feature as determined by the given classifier.
    It stops when the desired number of features is reached or accuracy is no longer improving.
    '''
    rfecv = RFECV(estimator=clf, cv=StratifiedKFold(cv), scoring="accuracy", min_features_to_select=min_feats)
    rfecv.fit(x_data_d, y_data_d)
    print("Features to keep", rfecv.get_feature_names_out(x_data_d.columns))
    
    if display:
        plt.rcParams['figure.dpi'] = 300
        plt.style.use('dark_background')
        plt.figure()
        plt.xlabel("Number of features selected")
        plt.ylabel("Mean test accuracy")
        plt.errorbar(range(min_feats, len(rfecv.cv_results_["mean_test_score"])+min_feats), 
                    rfecv.cv_results_["mean_test_score"], yerr=rfecv.cv_results_["std_test_score"])
        plt.title("Recursive Feature Elimination")
   
        plt.show()
    


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
    

def vc_dimension_check(clf, x_data_d):
    '''
    Given a model that provides a coef_ and intercept_ attribute, check if the VC bound is satisfied.
    '''
    dvc =  (np.sum([param.size for param in clf.coef_]) +  np.sum([param.size for param in clf.intercept_])) + 1
    N = x_data_d.shape[0]
    if 10 * dvc > N:
        print(f"VC bound is violated. Either increase the number of samples or decrease parameters by atleast {10 * dvc - N}")
    else:
        print(f"Model generalization is safe. VC Bound is satisfied where 10dvc={10 * dvc} < N={N}")


def show_hyperparams(clf):
    '''
    Print the hyperparameters of a model.
    '''
    return display(HTML(nice_table(clf.get_params(), title="Hyperparameters")))



def validation_curves(clf,x_data,y_data,cv, param_name,param_range,scoring="neg_mean_squared_error",
                      scoring_name="Error", display=True, print_scores=False,categorical=False):
    '''
    Plot the validation curve for a given model and hyperparameter.
    '''
    train_scores, test_scores = validation_curve(clf, x_data, y_data, param_name=param_name, param_range=param_range,
                                  cv=StratifiedKFold(cv), scoring=scoring, n_jobs=4)
    

    display_curves(param_range, train_scores, test_scores, scoring_name, param_name,
                   "Validation Curve", display, print_scores, overfit_measure=True,
                   categorical=categorical)
    


def learning_curves(clf, x_data, y_data, cv,N,scoring="neg_mean_squared_error",
                    scoring_name="Error", display=True, print_scores=False):

    '''
    Plot the learning curve for a given model.
    '''
    train_sizes, train_scores, test_scores = learning_curve(clf, x_data, y_data, cv=StratifiedKFold(cv), n_jobs=4, 
                                                            train_sizes=N, scoring=scoring)

    
    display_curves(train_sizes, train_scores, test_scores, scoring_name, "N", 
                   "Learning Curve", display, print_scores)
    

def display_curves(parameter, train_score, test_score,scoring_name,x_label, title,
                   display=True, print_scores=False,overfit_measure=False, categorical=False):

    '''
    For learning or validation curves, display them and/or printing the scores if needed.
    '''

    if scoring_name == "Error": # The score measure is neg mean squared error,so we need to flip the sign
        train_score =-train_score.mean(axis=1)
        test_score = -test_score.mean(axis=1)
    else:
        train_score = train_score.mean(axis=1)
        test_score = test_score.mean(axis=1)

    if print_scores:
        print("Training Error", train_score) 
        print("Validation Error", test_score)

    if display:
        plt.rcParams['figure.dpi'] = 300
        plt.style.use('dark_background')
        plt.figure()
        plt.xlabel(x_label)
        plt.ylabel(scoring_name)
        

        if categorical: #print the points instead of lines
            plt.scatter(parameter, train_score,marker='x',  label='Training '+scoring_name )
            plt.scatter(parameter, test_score,marker='x', label='Validation '+scoring_name)

            OF= test_score-train_score 
            print("Overfitting measure")
            for i in range(len(parameter)): 
                print(parameter[i], ":" , OF[i])

        else:
            plt.plot(parameter, train_score, markersize=5, label='Training '+scoring_name )
            plt.plot(parameter, test_score, markersize=5, label='Validation '+scoring_name)
        
        # check the point at which test score will start to increase and training score will start to decrease
            if overfit_measure: 
                # check if the difference between the consecutive scores is less/more than 0.001
                test = test_score[1:] - test_score[:-1] > 0.001
                train = train_score[1:] - train_score[:-1] < 0.001     

                # print("test", test)
                # print("train", train)
                optimal_param = parameter[np.where(test & train)[0][0]] 
                print( "Optimal "+ x_label +" is around:", optimal_param)

                plt.axvline(optimal_param, color='red', linestyle='--', label='Optimal '+x_label)

        plt.title(title)
        plt.legend(loc="best")
        plt.show()