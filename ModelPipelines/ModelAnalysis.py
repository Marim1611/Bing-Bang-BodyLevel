from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt

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
    

