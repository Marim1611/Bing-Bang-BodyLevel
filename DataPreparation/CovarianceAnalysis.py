import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as ss


class CorrelationMatrix:
    def __init__(self, x_data):
        self.x_data = x_data
        self.disc_feats = [feat for feat in x_data.columns if type(x_data.iloc[0, x_data.columns.get_loc(feat)]) == str]
        self.cont_feats = [feat for feat in x_data.columns if type(x_data.iloc[0, x_data.columns.get_loc(feat)]) != str]

    def numerical_correlation_matrix(self):
        '''
        Calculate a correlation matrix for the continuous features in the dataset
        '''
        x_data_cont = self.x_data[self.cont_feats]
        corr = x_data_cont.corr()
        return corr

    def categorical_correlation_matrix(self):
        '''
        Calculate a correlation matrix for the categorical features in the dataset
        '''
        x_data_disc = self.x_data[self.disc_feats]
        corr = np.zeros((len(self.disc_feats), len(self.disc_feats)))
        for i in range(len(self.disc_feats)):
            for j in range(len(self.disc_feats)):
                corr[i, j] = self.cramers_v(x_data_disc, self.disc_feats[i], self.disc_feats[j])
        return corr

    def mix_correlation_matrix(self):
        '''
        Calculate a correlation matrix for the categorical and continuous features in the dataset
        '''
        corr = np.zeros((len(self.disc_feats), len(self.cont_feats)))
        for i in range(len(self.disc_feats)):
            for j in range(len(self.cont_feats)):
                corr[i, j] = self.correlation_ratio(self.x_data, self.disc_feats[i], self.cont_feats[j])
        return corr

    def cramers_v(self, data, col1, col2):
        '''
        Calculate Cramers V statistic for categorial-categorial association.
        Like lift from Big data but more sophisticated.
        This was modified from SO: https://stackoverflow.com/questions/46498455/categorical-features-correlation/46498792#46498792
        '''
        confusion_matrix = pd.crosstab(data[col1], data[col2]).values
        chi2 = ss.chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum()
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
        rcorr = r - ((r-1)**2)/(n-1)
        kcorr = k - ((k-1)**2)/(n-1)
        return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

    def correlation_ratio(self, x_data, col1, col2):
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

    def plot_correlation_matrices(self):
        '''
            Plot all three correlation matrices (numerical, categorical, and mixed) side by side
        '''
        num_corr = self.numerical_correlation_matrix()
        cat_corr = self.categorical_correlation_matrix()
        mix_corr = self.mix_correlation_matrix()

        plt.style.use('dark_background')
        plt.rcParams['figure.dpi'] = 400
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(35, 18))
        fontsize = 15
        plt.rcParams.update({'font.size': fontsize})
        
        # Plot numerical correlation matrix
        for i in range(len(num_corr)):
            for j in range(len(num_corr)):
                if num_corr.values[i,j] > 0.4:
                    ax1.text(j, i, '{:.2f}'.format(num_corr.values[i, j]), ha="center", va="center", color="k", fontsize=fontsize)
                else:
                    ax1.text(j, i, '{:.2f}'.format(num_corr.values[i, j]), ha="center", va="center", color="w", fontsize=fontsize)
        ax1.matshow(num_corr, cmap='plasma')
        ax1.set_xticks(range(len(num_corr.columns)))
        ax1.set_xticklabels(num_corr.columns, rotation=90)
        ax1.set_yticks(range(len(num_corr.columns)))
        ax1.set_yticklabels(num_corr.columns)
        for tick in ax1.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize)
        ax1.set_title("Numerical Correlation Matrix")

        # Plot categorical correlation matrix
        for i in range(len(cat_corr)):
            for j in range(len(cat_corr)):
                if cat_corr[i,j] > 0.4:
                    ax2.text(j, i, '{:.2f}'.format(cat_corr[i, j]), ha="center", va="center", color="k", fontsize=fontsize)
                else:
                    ax2.text(j, i, '{:.2f}'.format(cat_corr[i, j]), ha="center", va="center", color="w", fontsize=fontsize)
        ax2.matshow(cat_corr, cmap='plasma')
        ax2.set_xticks(range(len(cat_corr)))
        ax2.set_xticklabels(self.disc_feats, rotation=90)
        ax2.set_yticks(range(len(cat_corr)))
        ax2.set_yticklabels(self.disc_feats)
        for tick in ax2.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize)
        ax2.set_title("Categorical Correlation Matrix")

        # Plot mix correlation matrix
        for i in range(len(mix_corr)):
            for j in range(len(mix_corr)):
                if mix_corr[i,j] > 0.4:
                    ax3.text(j, i, '{:.2f}'.format(mix_corr[i, j]), ha="center", va="center", color="k", fontsize=fontsize)
                else:
                    ax3.text(j, i, '{:.2f}'.format(mix_corr[i, j]), ha="center", va="center", color="w", fontsize=fontsize)
        ax3.matshow(mix_corr, cmap='plasma')
        ax3.set_xticks(range(len(mix_corr)))
        ax3.set_xticklabels(self.cont_feats, rotation=90)
        ax3.set_yticks(range(len(mix_corr)))
        ax3.set_yticklabels(self.disc_feats)
        for tick in ax3.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize)
        ax3.set_title("Mix Correlation Matrix")

        plt.show()

