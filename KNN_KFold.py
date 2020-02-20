# Group Project # 2
# K-fold cross-validation for hyper-parameter tuning and model comparison
#
# Implementation of K-Fold cross-validation
#   Used it with machine learning algorithms:
#       (1) train hyper-parameters
#       (2) compare prediction accuracy.

# imports
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as skKnn
from sklearn import preprocessing

def knn_wrap( X_train, y_train, X_new ) :
    lab_enc = preprocessing.LabelEncoder()
    Y_encoded = lab_enc.fit_transform(y_train)
    knn = skKnn(n_neighbors=2)
    knn.fit(X_train, Y_encoded)
    return knn.predict( X_new )

# Function: KFoldCV
# input arguments:
#   X_mat, a matrix of numeric inputs (one row for each observation, one column for each feature).
#   y_vec, a vector of binary outputs (the corresponding label for each observation, either 0 or 1).
#   ComputePredictions, a function that takes three inputs (X_train,y_train,X_new),
#                       trains a model using X_train,y_train,
#                       then outputs a vector of predictions (one element for every row of X_new).
#   fold_vec, a vector of integer fold ID numbers (from 1 to K).
def KFoldCV( X_mat, y_vec, ComputePredictions, fold_vec ):

    K = fold_vec.shape[0]
    X_size = X_mat.shape[0]

    # The function should begin by initializing a variable called error_vec, a numeric vector of size K.
    error_vec = np.empty( K )

    id_vec = np.random.randint( 1, K+1, X_mat.shape[ 0 ] )

    X_new = []
    Y_new = []
    X_train = []
    y_train = []

    # The function should have a for loop over the unique values k in fold_vec (should be from 1 to K).
    for fold_id in fold_vec :
        # first define X_new,y_new based on the observations for which the corresponding elements of fold_vec
        #   are equal to the current fold ID k.
        for index in range(X_mat.shape[0]):
            if (fold_id == id_vec[index]):
                X_new.append(X_mat[index].tolist())
                Y_new.append(y_vec[index].tolist())
                #X_new = np.append(X_new, X_mat[index])
                #Y_new = np.append(Y_new, y_vec[index])
            # then define X_train,y_train using all the other observations
            else:
                X_train.append(X_mat[index].tolist())
                y_train.append(y_vec[index].tolist())
                #X_train = np.append(X_train, X_mat[index])
                #y_train = np.append(y_train, y_vec[index])

        # then call ComputePredictions
        pred_new = ComputePredictions(X_train, y_train, X_new)

        # then compute the zero-one loss of pred_new with respect to y_new
        #   and store the mean (error rate) in the corresponding entry of error_vec.
        error_vec[fold_id-1] = np.mean(Y_new != pred_new)
        #error_vec[ fold_id-1 ] = 1 - knn.score( X_new, Y_encoded )

    # At the end of the algorithm you should return error_vec.
    return error_vec

fold_vec = np.array([1, 2, 3, 4, 5])
X_mat = np.array([[1,2,3],
                  [3,2,1],
                  [1,2,3],
                  [1,2,3],
                  [3,2,1],
                  [1,2,3]
                  ])
Y_vec = np.array([1,0,1,1,0,1])
print(KFoldCV( X_mat, Y_vec, knn_wrap, fold_vec))

# function: NearestNeighborsCV
# input arguments
# X_mat, a matrix of numeric inputs/features (one row for each observation, one column for each feature).
# y_vec, a vector of binary outputs (the corresponding label for each observation, either 0 or 1).
# X_new, a matrix of numeric inputs/features for which we want to compute predictions.
# num_folds, default value 5.
# max_neighbors, default value 20.
def NearestNeighborsCV( X_mat, y_vec, X_new, num_folds=5, max_neighbors=20 ):

    # randomly create a variable called validation_fold_vec, a vector with integer values from 1 to num_folds.
    validation_fold_vec = np.empty( num_folds )

    # initialize a variable called error_mat, a numeric matrix (num_folds x max_neighbors).
    error_mat = np.empty( num_folds, max_neighbors )

    # There should be a for loop over num_neighbors from 1 to max_neighbors.
    for index in range( max_neighbors ):

        # Inside the for loop you should call KFoldCV, and specify ComputePreditions=a function that uses your
        #   programming language’s default implementation of the nearest neighbors algorithm, with num_neighbors.
        #   e.g. scikit-learn neighbors in Python, class::knn in R.
        #   Store the resulting error rate vector in the corresponding column of error_mat.
        KFoldCV( X_mat, y_vec, None )

        # Compute a variable called mean_error_vec (size max_neighbors) by taking the mean of each column of error_mat.

        # Compute a variable called best_neighbors which is the number of neighbors with minimal error.

    # Your function should output
    #   (1) the predictions for X_new, using the entire X_mat,y_vec with best_neighbors;
    #   (2) the mean_error_mat for visualizing the validation error.
    #return ( X_pred, mean_error_mat )


# First scale the inputs (each column should have mean 0 and variance 1).
#   You can do this by subtracting away the mean and then dividing by the standard deviation of each column
#   (or just use a standard function like scale in R).
# Use NearestNeighborsCV on the whole data set, then plot validation error as a function of the number of neighbors,
#   separately for each fold.
# Draw a bold line for the mean validation error, and draw a point to emphasize the minimum.
# Randomly create a variable test_fold_vec which is a vector with one element for each observation,
#   and elements are integers from 1 to 4. In your report please include a table of counts with a row for each fold (1/2/3/4) and a column for each class (0/1).
# Use KFoldCV with three algorithms:
#   (1) baseline/underfit – predict most frequent class,
#   (2) NearestNeighborsCV,
#   (3) overfit 1-nearest neighbors model. Plot the resulting test error values as a function of the data set,
#       in order to show that the NearestNeighborsCV is more accurate than the other two models.