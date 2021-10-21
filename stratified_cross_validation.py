import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

#################################################################
# Prepare Target 
#################################################################

def prepare_targets(y):
    """
    Encodes classification target in the training set for  
    the cross-validation/feature selection.

    Args:
        y ([pandas.core.series.Series]): 
            A Series of all the training targets (y_train).

    Returns:
        [numpy.ndarray]: 
            Returns an encoded Array of the target variable.
    """
    le = LabelEncoder()
    
    le.fit(y)
    
    y_enc = le.transform(y)
    
    return y_enc

##################################################################
# Stratified Cross Validation Fold Function
##################################################################

def stratified_kfold(X, y, n_splits = 10):
    """
    Generates a stratified number of folds and returns a list
    of dataframes of each of the respective folds. 

    Args:
        X ([pandas.DataFrame]): 
            A DataFrame of all features that are being considered for 
            a given model.
        y ([pandas.Series]): 
            A Series of the target.
        n_splits (int, optional): The number of stratified kfolds to generate. Defaults to 10.

    Returns:
        [list]: Four lists of Pandas DataFrames, one for every X_train, 
        y_train, X_test, and y_test. So the default generates four lists 
        with ten Pandas DataFrames each.
    """

    # Create List of All Features in X_train
    features = X.columns.tolist()
    
    # Convert X_train Pandas DataFrame to Numpy Array
    X = X.to_numpy()

    # Encodes y_train Pandas Series
    y = prepare_targets(y)

    # Define Splits
    n_splits = n_splits

    # Generate Stratified Splits
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=False)

    # Enumerate the Splits and Summarize the Distributions
    for train_ix, test_ix in kfold.split(X, y):

        # Select Rows
        train_X, test_X = X[train_ix], X[test_ix]
        train_y, test_y = y[train_ix], y[test_ix]

    train_0, train_1 = len(train_y[train_y==0]), len(train_y[train_y==1])
    test_0, test_1   = len(test_y[test_y==0]), len(test_y[test_y==1])
	
    # Create Index for Recording Splits
    folds = [next(kfold.split(X,y)) for i in range(n_splits)]

    # Extract Each Split

    # Fold 1
    X_train_1 = pd.DataFrame(X[folds[0][0]]).set_axis([features], axis=1, inplace=False)
    X_test_1  = pd.DataFrame(X[folds[0][1]]).set_axis([features], axis=1, inplace=False)
    y_train_1 = pd.DataFrame(y[folds[0][0]])
    y_test_1  = pd.DataFrame(y[folds[0][1]])

    # Fold 2
    X_train_2 = pd.DataFrame(X[folds[1][0]]).set_axis([features], axis=1, inplace=False)
    X_test_2  = pd.DataFrame(X[folds[1][1]]).set_axis([features], axis=1, inplace=False)
    y_train_2 = pd.DataFrame(y[folds[1][0]])
    y_test_2  = pd.DataFrame(y[folds[1][1]])

    # Fold 3
    X_train_3 = pd.DataFrame(X[folds[2][0]]).set_axis([features], axis=1, inplace=False)
    X_test_3  = pd.DataFrame(X[folds[2][1]]).set_axis([features], axis=1, inplace=False)
    y_train_3 = pd.DataFrame(y[folds[2][0]])
    y_test_3  = pd.DataFrame(y[folds[2][1]])

    # Fold 4
    X_train_4 = pd.DataFrame(X[folds[3][0]]).set_axis([features], axis=1, inplace=False)
    X_test_4  = pd.DataFrame(X[folds[3][1]]).set_axis([features], axis=1, inplace=False)
    y_train_4 = pd.DataFrame(y[folds[3][0]])
    y_test_4  = pd.DataFrame(y[folds[3][1]])

    # Fold 5
    X_train_5 = pd.DataFrame(X[folds[4][0]]).set_axis([features], axis=1, inplace=False)
    X_test_5  = pd.DataFrame(X[folds[4][1]]).set_axis([features], axis=1, inplace=False)
    y_train_5 = pd.DataFrame(y[folds[4][0]])
    y_test_5  = pd.DataFrame(y[folds[4][1]])

    # Fold 6
    X_train_6 = pd.DataFrame(X[folds[5][0]]).set_axis([features], axis=1, inplace=False)
    X_test_6  = pd.DataFrame(X[folds[5][1]]).set_axis([features], axis=1, inplace=False)
    y_train_6 = pd.DataFrame(y[folds[5][0]])
    y_test_6  = pd.DataFrame(y[folds[5][1]])

    # Fold 7
    X_train_7 = pd.DataFrame(X[folds[6][0]]).set_axis([features], axis=1, inplace=False)
    X_test_7  = pd.DataFrame(X[folds[6][1]]).set_axis([features], axis=1, inplace=False)
    y_train_7 = pd.DataFrame(y[folds[6][0]])
    y_test_7  = pd.DataFrame(y[folds[6][1]])

    # Fold 8
    X_train_8 = pd.DataFrame(X[folds[7][0]]).set_axis([features], axis=1, inplace=False)
    X_test_8  = pd.DataFrame(X[folds[7][1]]).set_axis([features], axis=1, inplace=False)
    y_train_8 = pd.DataFrame(y[folds[7][0]])
    y_test_8  = pd.DataFrame(y[folds[7][1]])

    # Fold 9
    X_train_9 = pd.DataFrame(X[folds[8][0]]).set_axis([features], axis=1, inplace=False)
    X_test_9  = pd.DataFrame(X[folds[8][1]]).set_axis([features], axis=1, inplace=False)
    y_train_9 = pd.DataFrame(y[folds[8][0]])
    y_test_9  = pd.DataFrame(y[folds[8][1]])

    # Fold 10
    X_train_10 = pd.DataFrame(X[folds[9][0]]).set_axis([features], axis=1, inplace=False)
    X_test_10  = pd.DataFrame(X[folds[9][1]]).set_axis([features], axis=1, inplace=False)
    y_train_10 = pd.DataFrame(y[folds[9][0]])
    y_test_10  = pd.DataFrame(y[folds[9][1]])

    # Lists of Pandas DataFrames 
    X_train = [X_train_1, X_train_2, X_train_3, X_train_4, X_train_5, X_train_6, X_train_7, X_train_8, X_train_9, X_train_10] 
    X_test  = [X_test_1, X_test_2, X_test_3, X_test_4, X_test_5, X_test_6, X_test_7, X_test_8, X_test_9, X_test_10]
    y_train = [y_train_1, y_train_2, y_train_3, y_train_4, y_train_5, y_train_6, y_train_7, y_train_8, y_train_9, y_train_10]
    y_test  = [y_test_1, y_test_2, y_test_3, y_test_4, y_test_5, y_test_6, y_test_7, y_test_8, y_test_9, y_test_10]

    return X_train, X_test, y_train, y_test
