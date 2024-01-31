## Classifiers

The following classifiers will be found within the files below

#### Model 0: Naive Bayes

Default Scikit-Learn implementation of GaussianNB. Initial theta = default

#### Model 1: Logistic Regression; custom

Custom-built Python implementation which uses batch gradient descent algorithm described in lectures to update weights. Max epochs = 100, Initial weights vector = [0.1 for all thetas 1-n], Bias = 0.0, Learning rate = 0.001

#### Model 2: Logistic Regression; default

Default Scikit-Learn implementation LogisticRegression. Max epochs = 300, Initial weights = default, Bias = default

#### Model 3: Logistic Regression; SGD

Uses Scikit-Learn implementation of SGDClassifier with “log-loss” penalty. Performs Logistic Regression using a stochastic gradient descent algorithm to update weights. Learning rate is constant. Max epochs = 300, Learning rate type = constant, Learning rate = 0.01

#### Model 3.1: Logistic Regression; SGD with variable learning rate

Uses Scikit-Learn implementation of SGDClassifier with “log-loss” penalty. Performs Logistic Regression using a stochastic gradient descent algorithm to update weights. Learning rate is optimised according to an internal heuristic. Max epochs = 300, Learning rate type = optimal

#### Model 3.2: Logistic Regression; SGD with fast learning rate

Uses Scikit-Learn implementation of SGDClassifier with “log-loss” penalty. Performs Logistic Regression using a stochastic gradient descent algorithm to update weights. Learning rate is constant. Max epochs = 300, Learning rate type = constant, Learning rate = 0.1

#### Model 4: K-Means

Uses Scikit-Lean implementation of KMeans Num. Clusters = 6

#### Model 5: Ensemble

Majority Vote algorithm; collects results from 3 classifiers, then returns True if at least 2 classifiers return True. Else, returns False. Classifiers = [0, 2, 3]


## Files

All files except lib_processing contain a `main()` method, which will automatically run what is described below

`lib_processing.py`
Custom library with methods for pre-processing of labelled and unlabelled data from files, and evaluation of results

`conventional_testing.py`
Generates and runs initial tests on all sklearn-based classifiers used in the report

`semi_supervised.py`
Generates and runs full test suite, including semi-supervised learning, on all sklearn-based classifiers used in the report

`classifier_custom_logistic_regression.py`
Built-from-scratch logistic regression using batch gradient descent (Classifier 1 in the report)

`ablation.py`
Trains classifiers 0, 2 and 3 to perform single-pass ablation;
train on dataset with a single feature column removed, and evaluate results

`kmeans_elbow.py`
Generates KMeans models with increasing numbers of clusters. Used to generate inertia values for use in diagrams.ipynb

`learning_curves.py`
Generates learning curves using method demonstrated in IML lectures, prints resulting arrays for use in diagrams.ipynb

`diagrams.ipynb`
Used to generate all graphs used in the report. Data is sourced from above python files.
