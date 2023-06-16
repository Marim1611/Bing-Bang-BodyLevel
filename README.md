# 👩‍🦯 Body Level Classification 🏃‍♂️
<p align="justify"> 
The project comprises tackling the supervised problem of body level classification: given numerical and categorical features regarding an individual’s health such as their height, weight, family history, age, and eleven others, the objective is to predict the body level of the individual (out of four possible levels). </p>

## 🚀 Pipeline
Our solution to said problem utilitizes the following pipeline
<img width="1275" alt="image" src="https://github.com/Marim1611/ML_Project/assets/49572294/7a63d80f-5aa0-4e0f-a3b3-242c297764b0">

## 📂 Folder Structure
The following is the implied folder structure:
```
.
├── DataFiles
│ ├── dataset.csv
│ ├── train.csv
│ └── val.csv
├── DataPreparation
│ ├── CovarianceAnalysis.py
│ ├── DataPreparation.ipynb
│ └── DataPreparation.py
├── HandleClassImbalance
│ ├── HandleClassImbalance.ipynb
│ ├── HandleClassImbalance.py
├── ModelBaselines
│ └── Baseline.ipynb
├── Model Pipelines
│ ├── AdaBoost
│ │ └── Adaboost.ipynb
│ ├── Bagging
│ │ ├── Analysis.ipynb
│ │ └── SVMBagging.ipynb
│ ├── LogisticRegression
│ │ ├── Analysis.ipynb
│ │ └── LogisticRegression.ipynb
│ ├── Perceptron
│ │ ├── Analysis.ipynb
│ │ └── Perceptron.ipynb
│ ├── RandomForest
│ │ ├── Analysis.ipynb
│ │ └── RandomForest.ipynb
│ ├── SVM
│ │ ├── Analysis.ipynb
│ │ └── SVM.ipynb
│ ├── StackingEnsemble
│ │ └── StackingEnsemble.ipynb
│ ├── VotingEnsemble
│ │ └── VotingEnsemble.ipynb
│ ├── ModelAnalysis.py
│ ├── ModelVisualization.py
├── ModelScoring
│ └── Pipeline.py
├── References
│ └── ML Project Document.pdf
├── Saved
├── Quests
├── README.md
└── utils.py
```

## 🚁 Running the Project

```python
pip install requirements.txt
# To run any stage of the pipeline, consider the stage's folder. There will always be a demonstration notebook.
```

## 📝 Executive Summary

<p align="justify"> 
We started by designing and running a dataset analysis pipeline (i.e., studying the target function) which has lead to our initiation of SVM, LR, GNB, RF, Perceptron and Adaboost models. Then we proceeded by designing a model analysis cycle that we implemented for each of these models with the objective of studying the model's performance and tuning over the best hyperparameters, set of features and data preparation choices.
</p>

Best results for WF1 are as follows under 10-Repeated-10-Fold Cross Validation (which get better under other cval approaches)
<table>
  <tr>
    <th>SVM</th>
    <th>Logistic Regression</th>
    <th>Random Forest</th>
  </tr>
  <tr>
    <td>98.65%</td>
    <td>98.38%</td>
    <td>97.63%</td>
  </tr>
</table>

The following visually depicts SVM's over the most important features
![output](https://github.com/Marim1611/ML_Project/assets/49572294/3d76dc31-b253-4806-9c8b-40a58fa6c57b)

Which we did not end up choosing in the end, instead we considered forming Ensembles of these models via Voting, Bagging and Stacking. Yielding the following
<table>
  <tr>
    <th>Bagging SVM</th>
    <th>Voting</th>
    <th><font color="yellow">Stacked Generalization</font></th>
  </tr>
  <tr>
    <td>98.37%</td>
    <td>96.5%</td>
    <td>99.12%</td>
  </tr>
</table>

By this, our final model was stacking. 

We shall illustrate the whole pipeline including the analysis stages in the rest of the README. For an extensive overview of the the <b>insights extracted</b> and analysis results for the rest of the models please check the [report](https://github.com/Marim1611/ML_Project/tree/main/Report.pdf/) or the demonstration notebooks herein.

## 🌊 Data Preparation
Data preparation involves reading the data and putting in a suitable form. Options employed in this stage beyond reading the data are:
- To read specific splits of the data (by default train)
- To read only columns of numerical or categorical types (or both)
- Label,one-hot or frequency encoding for categorical features
- To standardize the data

This module was used to ingest the data for all subsequent models and analysis.

## 🎨 Dataset Analysis
In light of guiding model initiation by studying the population and the target function we have performed the following analyses:

### ◉ Basic Counts & Variables Involved
<table>
    <th>Number of Samples</th>
    <th>Number of Features</th>
    <th>Number of Classes</th>
  </tr>
  <tr>
    <td>1180</td>
    <td>16</td>
    <td>4</td>
  </tr>
</table>

| Variable | Gender | H_Cal_Consump | Smoking | Fam_Hist | H_Cal_Burn | Alcohol_Consump | Food_Between_Meals | Transport | Age | Height | Weight | Veg_Consump | Water_Consump | Meal_Count | Phys_Act | Time_E_Dev |
|---------|--------|---------------|---------|----------|------------|-----------------|--------------------|-----------|-----|--------|--------|-------------|---------------|------------|----------|------------|
| <b>#Uniques</b>   | 2      | 2             | 2       | 2        | 2          | 3               | 4                  | 5         | Numerical | Numerical | Numerical | Numerical    | Numerical      | Numerical   | Numerical | Numerical    |

### ◉ Variable Distributions
In this, we analyzed the distribution of each variable via a bar chart or kernel estimated density depending on whether its numerical or categorical respectively.

![image](https://github.com/Marim1611/ML_Project/assets/49572294/5fc3c033-2f12-4008-a923-76c8daeb1142)

### ◉ Prior Distribution
For purposes of studying imbalance between classes.

![image](https://github.com/Marim1611/ML_Project/assets/49572294/7816f812-c989-4b1d-a5d5-0bedcb49d53d)

### ◉ Variable Correlations
We analyzed correlations among all pairs of numerical variables using Pearson's, then all pairs of categorical variables using Crammer's V and then all pairs of numerical and categorical variables using Pearson's correlation ratio.

![image](https://github.com/Marim1611/ML_Project/assets/49572294/0d05bd12-7f01-4b77-98f9-db05c63d30dd)

### ◉ Separability with Numerical Variables
Here, the separability of the target is analyzed under all possible pairs of numerical variables. 

![image](https://github.com/Marim1611/ML_Project/assets/49572294/c37bd89d-1010-45ce-a916-254c9da8a12d)

### ◉ Separability with Categorical Variables
In this, we study the separability of the target under different categories of each categorical variable.

![image](https://github.com/Marim1611/ML_Project/assets/49572294/37875233-8c3c-40a8-ad7d-7c3587b93d17)

### ◉ Theoritical Generalization Guarantees
We automated a generalization check given any two of validation set size, maximum allowed error and probability of violating that error. This was used to inspire the decision of the number of splits for cross validation (tuning each model) and the size of the validation set (choosing between models).

<font size=4>Hoeffding's Inequality states:
                    $$P[|E_{out}(g)-E_{test}(g)| \leq \epsilon] \geq 1-2e^{-2N_{test}\epsilon^2}$$
                    If we use validation set of size $0.2N_{train}=295$ then with $\epsilon=0.06$ we have 
                    $$P[|E_{out}(g)-E_{test}(g)| \leq 0.06] \geq 0.761$$
                    In other words, 
                    with probability at least $0.761$, the generalization error of our model will be at most 0.06 given a validation set of size 295.
                    </font>

## 🤖 Model Initiation

We considered two trivial baselines (MostFrequent and UniformRandom) and another nontrivial baseline (Gaussian Naive Bayes) so that we can set the bar regarding the bias of further models we consider. We then initiated and analyzed the following models:
- SupportVectorMachines
- LogisticRegression
- Perceptron
- RandomForest
- AdaptiveBoosting



## 🛸 Model Analysis
We designed a unified analysis cycle that applies to any of the models as demonstrated in the [report](https://github.com/Marim1611/ML_Project/tree/main/Report.pdf/) and the notebooks. It consists of the following stages at no particular order:

<table>
  <tr>
    <th>Analysis Stage</th>
    <th>Components</th>
  </tr>
  <tr>
    <td rowspan="2">Model Greetings</td>
    <td >Initiating Model and Viewing Hyperparameters</td>
  </tr>
  <tr>
    <td>Studying the Hyperparameters and their Importance (documentation) </td>
  </tr>

  <tr>
    <td rowspan="4">Basic Model Analysis</td>
    <td>Testing Model Assumptions (if any)</td>
  </tr>
  <tr>
    <td>VC Dimension Check for Generalization </td>
  </tr>
  <tr>
    <td>Bias Variance Analysis </td>
  </tr>
  <tr>
    <td>Learning Curve</td>
  </tr>

  <tr>
    <td rowspan="3">Hyperparameter Analysis</td>
    <td>Validation Curves</td>
  </tr>
  <tr>
    <td>Hyperparameter Search </td>
  </tr>
  <tr>
    <td>Hyperparameter Logging </td>
  </tr>

  <tr>
    <td rowspan="2">Feature Analysis</td>
    <td>Feature Importance</td>
  </tr>
  <tr>
    <td>Recursive Feature Elimination</td>
  </tr>

  <tr>
    <td rowspan="2">Class Imbalance Analysis</td>
    <td>Analyzing Different Methods</td>
  </tr>
  <tr>
    <td>Analysis Different Hyperparameters</td>
  </tr>

</table>

We will demonstrate this for Logistic Regression, for the <b>extracted insights</b> and other models [report](https://github.com/Marim1611/ML_Project/tree/main/Report.pdf/) or the notebooks.

### 🤝 Model Greetings
| C       | class_weight | dual  | fit_intercept | intercept_scaling | l1_ratio | max_iter | multi_class | n_jobs | penalty | random_state | solver    | tol    | verbose | warm_start |
|---------|--------------|-------|---------------|-------------------|----------|----------|-------------|--------|----------|--------------|------------|--------|---------|------------|
| 40.074  | balanced     | False | True          | 1                 | None     | 100      | multinomial | None   | l2       | None         | newton-cg  | 0.0001 | 0       | False      |

The purpose of this stage is to get familiar with the model and its hyperparameters which involved research or reading the documentation.

### 💡 Basic Model Analysis
#### ◉ Testing Log-Linearity Assumption
![image](https://github.com/Marim1611/ML_Project/assets/49572294/1149fad5-be5d-4642-bd80-5c18dde356fd)


#### ◉ VC Dimension Analysis
<font size=4>By estimating the VC dimension of the model, 
                    we have $d_{vc}=37$. 
                    Since, $N=1477$, it holds that 
                    $$N \geq 10d_{vc}$$
                    Hence, model is expected to have no issues with generalization.
                    </font>

In this, the number of parameters of the model were used to estimate its generalization ability using the VC bound rule of thumb.

#### ◉ Bias-Variance Analysis
| Train WF1 | Val WF1 | Avoidable Bias | Variance |
|-----------|---------|----------------|----------|
|   0.986   |  0.981  |     0.014      |  0.005   |

The bias and variance of the model were heuristically computed here (Andrew NG's style) 

#### ◉ Learning Curve
![image](https://github.com/Marim1611/ML_Project/assets/49572294/5b4b8559-db26-4073-9087-2e12795ab2a9)

This helps indicate the bias of the model and sheds light on whether it would benefit from adding more data.

### 🔎 Hyperparameter Analysis
#### Validation Curves
![image](https://github.com/Marim1611/ML_Project/assets/49572294/c78537e5-448b-4bdd-8f11-5f25294b9dcb)

To study the effects of specific hyperparameters on the model's performance (in-sample and out-of-sample error) and mark the point where the model starts to overfit.

#### ◉ Random Hyperparameter Search
| C       | class_weight | multi_class  | penalty | solver     | WF1     |
|---------|--------------|--------------|---------|------------|---------|
| 40.074  | balanced     | multinomial  | l2      | newton-cg  | 0.98104 |

Here we used random search to find an optimal set of hyperparameters.

### 🚦 Features Analysis
#### ◉ Feature Importance Analysis
![image](https://github.com/Marim1611/ML_Project/assets/49572294/01294529-8ad7-4ea4-b691-b517b05f5c58)

In this, we analyzed the importance of each feature as assigned by the model's weights.

#### ◉ Recursive Feature Elimination
![image](https://github.com/Marim1611/ML_Project/assets/49572294/3dda1615-1faa-4818-87c0-dfbcf7a556c4)
Top 3 Features
| Veg_Consump | Height | Weight  |
|-------------|--------|---------|
|   0.32722   | 7.841  | 26.777  |

As suggested by one of Vapnik’s papers, a decent feature selection strategy is to remove the least important feature until a minimum number of features is reached or the metric is no longer improving.

### ⚖️ Class Imbalance Analysis
#### ◉ Analyzing Different Methods
![image](https://github.com/Marim1611/ML_Project/assets/49572294/c8b1d62e-9934-4152-ac44-e543fc6e642b)

The purpose of this was to compare different resampling approaches and class-weighting.

#### ◉ Analyzing Different Hyperparameters
![image](https://github.com/Marim1611/ML_Project/assets/49572294/cfea0718-263f-4ec2-b459-366be6788e5e)

Different hyperparameters within specific resampling approach(es) were analyzed here.

## 🏁 Model Evaluation, Ensemble and Final Delivery
As illustrated above.

## 📜 Conventions
We have set the following set of working [standards](https://github.com/Marim1611/ML_Project/tree/main/MLDIR.md/) before undertaking the project.


## 👥 Collaborators
<!-- readme: contributors -start -->
<table>
<tr>
    <td align="center">
        <a href="https://github.com/mariemzayn18">
            <img src="https://avatars.githubusercontent.com/u/76264155?v=4" width="100;" alt="mariemzayn18"/>
            <br />
            <sub><b>Mariem Muhammed</b></sub>
        </a>
    </td>
    <td align="center">
        <a href="https://github.com/Marim1611">
            <img src="https://avatars.githubusercontent.com/u/76243256?v=4" width="100;" alt="Marim1611"/>
            <br />
            <sub><b>Marim Naser</b></sub>
        </a>
    </td>
    <td align="center">
        <a href="https://github.com/EssamWisam">
            <img src="https://avatars.githubusercontent.com/u/49572294?v=4" width="100;" alt="EssamWisam"/>
            <br />
            <sub><b>Essam</b></sub>
        </a>
    </td>
    <td align="center">
        <a href="https://github.com/Muhammad-saad-2000">
            <img src="https://avatars.githubusercontent.com/u/61880555?v=4" width="100;" alt="Muhammad-saad-2000"/>
            <br />
            <sub><b>MUHAMMAD SAAD</b></sub>
        </a>
    </td></tr>
</table>
<!-- readme: contributors -end -->

## 📈 Progress Tracking
We have utilized [Notion](https://www.notion.so/) for progress tracking and task assignment among the team.

